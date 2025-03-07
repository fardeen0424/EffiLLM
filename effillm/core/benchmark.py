# EffiLLM/effillm/core/benchmark.py
import time
import torch
import psutil
import gc
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import logging

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMBenchmark:
    """Core benchmarking class for measuring LLM performance."""
    
    def __init__(
        self, 
        model_id: str, 
        device: str = "auto",
        batch_sizes: List[int] = [1, 4, 16],
        sequence_lengths: List[int] = [128, 512, 1024],
        warmup_runs: int = 3,
        num_runs: int = 10,
        quantization_config: Optional[Dict] = None,
    ):
        self.model_id = model_id
        
        # Auto-detect device if set to auto
        self.device = self._resolve_device(device)
        
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
        self.quantization_config = quantization_config
        self.model = None
        self.tokenizer = None
        self.results = {}
        
        # Initialize memory tracking
        if self.device == "cuda" and PYNVML_AVAILABLE:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming first GPU
        
    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use (CPU or CUDA)."""
        if device != "auto":
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self):
        """Load model and tokenizer with appropriate quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        start_time = time.time()
        logger.info(f"Loading model: {self.model_id}")
        
        # Basic memory usage before model loading
        initial_memory = self._measure_memory()
        
        # Handle quantization settings
        if self.quantization_config:
            # Import and apply quantization method
            from effillm.quantization.quantizers import apply_quantization
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = apply_quantization(self.model_id, self.quantization_config, self.device)
        else:
            # Standard loading
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
        
        # Measure loading time and memory impact
        loading_time = time.time() - start_time
        final_memory = self._measure_memory()
        
        self.results["loading"] = {
            "time_seconds": loading_time,
            "memory_before": initial_memory,
            "memory_after": final_memory,
            "memory_impact": {
                k: final_memory[k] - initial_memory[k] for k in initial_memory.keys()
            }
        }
        
        logger.info(f"Model loaded in {loading_time:.2f} seconds")
        return self.model
        
    def _measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage (RAM and VRAM if available)."""
        memory_stats = {
            "ram_used_gb": psutil.Process().memory_info().rss / (1024 ** 3),
            "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        }
        
        if self.device == "cuda" and PYNVML_AVAILABLE:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_stats.update({
                "vram_used_gb": mem_info.used / (1024 ** 3),
                "vram_total_gb": mem_info.total / (1024 ** 3),
            })
            
        return memory_stats
    
    def prepare_inputs(self, sample_texts: Optional[List[str]] = None, max_length: int = 512) -> Dict:
        """Prepare tokenized inputs for benchmarking."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
            
        if not sample_texts:
            # Default sample text if none provided
            sample_texts = [
                "The quick brown fox jumps over the lazy dog. " * 20,
            ]
            
        # Tokenize the inputs
        encoded_inputs = self.tokenizer(
            sample_texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        if self.device == "cuda":
            encoded_inputs = {k: v.cuda() for k, v in encoded_inputs.items()}
            
        return encoded_inputs
    
    def measure_inference_speed(self, batch_size: int, seq_length: int, generation_length: int = 128) -> Dict:
        """Measure inference speed metrics for a specific configuration."""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Create dummy input data
        input_ids = torch.randint(
            100, 1000, 
            (batch_size, seq_length), 
            device=self.device
        )
        
        attention_mask = torch.ones_like(input_ids)
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=10,  # Short for warmup
                    do_sample=False
                )
        
        # Measure memory before inference
        pre_inference_memory = self._measure_memory()
        
        # Inference latency measurement (time to first token)
        time_to_first_token = []
        for _ in range(self.num_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(input_ids, attention_mask=attention_mask)
                
            torch.cuda.synchronize() if self.device == "cuda" else None
            time_to_first_token.append(time.time() - start_time)
            
        # Throughput measurement (full generation)
        generation_times = []
        generated_tokens = []
        for _ in range(self.num_runs):
            torch.cuda.synchronize() if self.device == "cuda" else None
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=generation_length,
                    do_sample=False
                )
                
            torch.cuda.synchronize() if self.device == "cuda" else None
            generation_times.append(time.time() - start_time)
            generated_tokens.append(outputs.shape[1] - input_ids.shape[1])  # New tokens
            
        # Measure memory after inference
        post_inference_memory = self._measure_memory()
        
        # Calculate metrics
        avg_first_token_time = np.mean(time_to_first_token)
        avg_generation_time = np.mean(generation_times)
        avg_tokens_generated = np.mean(generated_tokens)
        tokens_per_second = avg_tokens_generated / avg_generation_time * batch_size
        
        results = {
            "batch_size": batch_size,
            "sequence_length": seq_length,
            "time_to_first_token": {
                "mean": avg_first_token_time,
                "std": np.std(time_to_first_token),
                "min": np.min(time_to_first_token),
                "max": np.max(time_to_first_token),
            },
            "generation_time": {
                "mean": avg_generation_time,
                "std": np.std(generation_times),
                "min": np.min(generation_times),
                "max": np.max(generation_times),
            },
            "throughput": {
                "tokens_per_second": tokens_per_second,
                "tokens_per_second_per_instance": tokens_per_second / batch_size,
            },
            "memory": {
                "before": pre_inference_memory,
                "after": post_inference_memory,
                "impact": {
                    k: post_inference_memory[k] - pre_inference_memory[k] 
                    for k in pre_inference_memory.keys()
                }
            }
        }
        
        return results
    
    def run_benchmark(self) -> Dict:
        """Run the complete benchmark suite across configurations."""
        if not self.model:
            self.load_model()
            
        results = {"model_id": self.model_id, "device": self.device}
        results.update({"loading": self.results.get("loading", {})})
        
        # Benchmark configurations
        inference_results = {}
        
        for batch_size in self.batch_sizes:
            for seq_length in self.sequence_lengths:
                logger.info(f"Running benchmark: batch_size={batch_size}, seq_length={seq_length}")
                
                try:
                    # Skip configurations that would likely OOM
                    if self.device == "cuda":
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        if batch_size * seq_length > vram_gb * 1e9 / 4:  # Rough estimate
                            logger.warning(f"Skipping batch_size={batch_size}, seq_length={seq_length} to prevent OOM")
                            continue
                    
                    config_result = self.measure_inference_speed(batch_size, seq_length)
                    config_key = f"bs{batch_size}_seq{seq_length}"
                    inference_results[config_key] = config_result
                    
                    # Cleanup between runs
                    torch.cuda.empty_cache() if self.device == "cuda" else None
                    gc.collect()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"OOM error with batch_size={batch_size}, seq_length={seq_length}")
                    else:
                        logger.error(f"Error in benchmark: {e}")
                except Exception as e:
                    logger.error(f"Error in benchmark: {e}")
        
        results["inference"] = inference_results
        self.results = results
        return results
    
    def export_results(self, format="json", filepath=None):
        """Export benchmark results to the specified format."""
        if format == "json":
            import json
            result_str = json.dumps(self.results, indent=2)
            
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(result_str)
                logger.info(f"Results saved to {filepath}")
            return result_str
            
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # TODO: Implement CSV export formatting
            
            result_str = output.getvalue()
            if filepath:
                with open(filepath, 'w') as f:
                    f.write(result_str)
                logger.info(f"Results saved to {filepath}")
            return result_str
            
        else:
            raise ValueError(f"Unsupported export format: {format}")