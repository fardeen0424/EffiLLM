# EffiLLM/effillm/core/evaluation.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
import time
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Evaluates model quality across different quantization methods 
    and optimization settings.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}
    
    def evaluate_perplexity(
        self, 
        model, 
        eval_texts: List[str], 
        max_length: int = 512
    ) -> Dict[str, float]:
        """
        Evaluate model perplexity on a set of test texts.
        Lower perplexity is better.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(eval_texts, desc="Evaluating perplexity"):
                # Tokenize input
                tokenized = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=max_length
                )
                
                input_ids = tokenized.input_ids.to(self.device)
                attention_mask = tokenized.attention_mask.to(self.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=input_ids
                )
                
                # Get loss and update totals
                loss = outputs.loss.item()
                tokens = attention_mask.sum().item()
                
                total_loss += loss * tokens
                total_tokens += tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            "perplexity": perplexity,
            "loss": avg_loss,
            "num_tokens": total_tokens
        }
    
    def evaluate_quality_vs_speed(
        self,
        original_model,
        quantized_model,
        eval_texts: List[str],
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Compare quality vs speed tradeoff between original and quantized model.
        """
        # Get quality metrics for both models
        logger.info("Evaluating original model quality...")
        original_metrics = self.evaluate_perplexity(original_model, eval_texts, max_length)
        
        logger.info("Evaluating quantized model quality...")
        quantized_metrics = self.evaluate_perplexity(quantized_model, eval_texts, max_length)
        
        # Speed comparison (tokens per second)
        logger.info("Measuring original model speed...")
        original_speed = self._measure_inference_speed(original_model, max_length)
        
        logger.info("Measuring quantized model speed...")
        quantized_speed = self._measure_inference_speed(quantized_model, max_length)
        
        # Memory usage comparison
        original_memory = self._estimate_model_memory(original_model)
        quantized_memory = self._estimate_model_memory(quantized_model)
        
        # Calculate relative differences
        quality_ratio = quantized_metrics["perplexity"] / original_metrics["perplexity"]
        speed_ratio = quantized_speed["tokens_per_second"] / original_speed["tokens_per_second"]
        memory_ratio = quantized_memory["model_size_mb"] / original_memory["model_size_mb"]
        
        return {
            "original": {
                "quality": original_metrics,
                "speed": original_speed,
                "memory": original_memory
            },
            "quantized": {
                "quality": quantized_metrics,
                "speed": quantized_speed,
                "memory": quantized_memory
            },
            "comparison": {
                "quality_ratio": quality_ratio,  # Lower is worse
                "speed_ratio": speed_ratio,      # Higher is better
                "memory_ratio": memory_ratio,    # Lower is better
                "efficiency_score": speed_ratio / (quality_ratio * memory_ratio)  # Higher is better
            }
        }
    
    def _measure_inference_speed(
        self, 
        model, 
        max_length: int = 512,
        batch_sizes: List[int] = [1, 4, 8],
        num_repeats: int = 5
    ) -> Dict[str, Any]:
        """Measure inference speed in tokens per second."""
        model.eval()
        results = defaultdict(list)
        
        # Generate random input IDs (dictionary words tend to be more representative than random tokens)
        input_length = max_length // 2  # Use half max_length as input, generate the rest
        
        for batch_size in batch_sizes:
            # Create random input data
            input_ids = torch.randint(
                100, 1000, 
                (batch_size, input_length),
                device=self.device
            )
            attention_mask = torch.ones_like(input_ids)
            
            # Warm-up runs
            for _ in range(2):
                with torch.no_grad():
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=10,
                        do_sample=False
                    )
            
            # Timed runs
            torch.cuda.synchronize() if self.device == "cuda" else None
            
            for _ in range(num_repeats):
                start_time = time.time()
                
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_length - input_length,
                        do_sample=False
                    )
                
                torch.cuda.synchronize() if self.device == "cuda" else None
                end_time = time.time()
                
                # Calculate speed
                generation_time = end_time - start_time
                new_tokens = output.shape[1] - input_length
                total_new_tokens = new_tokens * batch_size
                
                tokens_per_second = total_new_tokens / generation_time
                results[f"batch_{batch_size}"].append(tokens_per_second)
        
        # Calculate averages
        averages = {}
        for k, v in results.items():
            averages[k] = np.mean(v)
        
        # Get overall average and best result
        all_speeds = [item for sublist in results.values() for item in sublist]
        avg_speed = np.mean(all_speeds)
        max_speed = np.max(list(averages.values()))
        optimal_batch = max(averages, key=averages.get)
        
        return {
            "tokens_per_second": avg_speed,
            "max_tokens_per_second": max_speed,
            "optimal_batch_size": optimal_batch,
            "detailed": dict(averages)
        }
    
    def _estimate_model_memory(self, model) -> Dict[str, float]:
        """Estimate model memory usage."""
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate size based on dtype
        param_size = 0
        for param in model.parameters():
            if param.dtype == torch.float16 or param.dtype == torch.half:
                param_size += param.numel() * 2  # 2 bytes per parameter
            elif param.dtype == torch.float32 or param.dtype == torch.float:
                param_size += param.numel() * 4  # 4 bytes per parameter
            elif param.dtype == torch.int8:
                param_size += param.numel() * 1  # 1 byte per parameter
            elif param.dtype == torch.int4 or "int4" in str(param.dtype).lower():
                param_size += param.numel() * 0.5  # 0.5 bytes per parameter (approximation)
            else:
                # Default to float32
                param_size += param.numel() * 4
        
        # Convert to MB
        param_size_mb = param_size / (1024 * 1024)
        
        return {
            "total_params": total_params,
            "model_size_mb": param_size_mb,
            "model_size_gb": param_size_mb / 1024
        }