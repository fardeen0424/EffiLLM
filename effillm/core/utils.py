# EffiLLM/effillm/core/utils.py
import os
import logging
import platform
import subprocess
import torch
import psutil
import time
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, str]:
    """Collect system information for benchmark context."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pytorch_version": torch.__version__,
    }
    
    # GPU information if available
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory
            gpu_info.append(f"{name} ({memory / (1024**3):.2f} GB)")
        
        info["gpu_devices"] = gpu_info
    
    return info

def estimate_memory_requirements(
    model_id: str,
    batch_size: int,
    seq_length: int,
    quantization: Optional[str] = None
) -> Dict[str, float]:
    """Estimate memory requirements for a benchmark configuration."""
    # Get model info from Hugging Face if possible
    params = None
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        # Use model description to estimate parameters
        # This is approximate as the actual parameter count may vary
        if hasattr(info, "model_size") and info.model_size:
            size_str = info.model_size.lower()
            if "B" in size_str:  # Billion
                params = float(size_str.replace("B", "").strip()) * 1e9
            elif "M" in size_str:  # Million
                params = float(size_str.replace("M", "").strip()) * 1e6
    except:
        # Fallback to name-based heuristics if API doesn't work
        if "1.3b" in model_id.lower():
            params = 1.3e9
        elif "7b" in model_id.lower():
            params = 7e9
        elif "13b" in model_id.lower():
            params = 13e9
    
    # Use default estimate if we couldn't determine
    if params is None:
        # Default to 1B parameters as a fallback
        params = 1e9
        logger.warning(f"Couldn't determine model size for {model_id}, using default estimate of 1B parameters")
    
    # Calculate memory requirements
    bytes_per_param = 2  # fp16 by default
    
    if quantization == "int8":
        bytes_per_param = 1
    elif quantization == "int4":
        bytes_per_param = 0.5
    
    model_size_gb = params * bytes_per_param / (1024**3)
    
    # Activation memory (rough estimate - depends on model architecture)
    # Typically proportional to batch_size * seq_length * hidden_size
    # Using a simple heuristic based on model size and sequence length
    activation_memory_factor = 2.5  # Empirical factor
    activation_memory_gb = (batch_size * seq_length * model_size_gb) / (1e6) * activation_memory_factor
    
    # KV cache for generation (matters for longer generations)
    kv_cache_gb = (batch_size * seq_length * model_size_gb) / (100) * 0.2  # Rough estimate
    
    total_estimate_gb = model_size_gb + activation_memory_gb + kv_cache_gb
    
    return {
        "model_parameters": params,
        "model_size_gb": model_size_gb,
        "activation_memory_gb": activation_memory_gb,
        "kv_cache_gb": kv_cache_gb,
        "total_estimate_gb": total_estimate_gb
    }

def is_configuration_feasible(
    batch_size: int,
    seq_length: int, 
    model_id: str,
    device: str = "cuda",
    quantization: Optional[str] = None
) -> Tuple[bool, str]:
    """Check if a benchmark configuration is feasible on the current hardware."""
    if device != "cuda" or not torch.cuda.is_available():
        # CPU configurations are generally memory-bound but not GPU-bound
        # We'll assume it's feasible but might be slow
        return True, "CPU configuration should work but may be slow"
    
    # Estimate requirements
    estimate = estimate_memory_requirements(model_id, batch_size, seq_length, quantization)
    total_required = estimate["total_estimate_gb"]
    
    # Get available GPU memory
    available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Check if we have enough memory with a safety margin
    safety_factor = 0.8  # Use 80% of available memory as our limit
    available_with_safety = available_memory_gb * safety_factor
    
    if total_required > available_with_safety:
        return False, f"Estimated memory requirement ({total_required:.2f} GB) exceeds available GPU memory ({available_with_safety:.2f} GB)"
    
    return True, "Configuration should be feasible"

def get_optimal_configurations(
    model_id: str, 
    device: str = "cuda"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate optimal benchmark configurations based on model and hardware.
    Returns a set of recommended batch sizes and sequence lengths.
    """
    configs = {
        "recommended": [],
        "aggressive": []  # Configurations that might work but are pushing limits
    }
    
    # Base sequence lengths to test
    seq_lengths = [128, 256, 512, 1024]
    
    # For very large models, limit sequence lengths to save memory
    if "65b" in model_id.lower() or "70b" in model_id.lower():
        seq_lengths = [128, 256, 512]
    
    if device == "cuda" and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Small GPU (<=8GB)
        if vram_gb <= 8:
            for seq_len in [128, 256]:
                for bs in [1, 2]:
                    feasible, _ = is_configuration_feasible(bs, seq_len, model_id, device)
                    if feasible:
                        configs["recommended"].append({"batch_size": bs, "seq_length": seq_len})
            
            # Aggressive settings
            for seq_len in [512, 1024]:
                feasible, _ = is_configuration_feasible(1, seq_len, model_id, device)
                if feasible:
                    configs["aggressive"].append({"batch_size": 1, "seq_length": seq_len})
        
        # Medium GPU (8GB-16GB)
        elif vram_gb <= 16:
            for seq_len in [128, 256, 512]:
                for bs in [1, 2, 4]:
                    feasible, _ = is_configuration_feasible(bs, seq_len, model_id, device)
                    if feasible:
                        configs["recommended"].append({"batch_size": bs, "seq_length": seq_len})
            
            # Aggressive settings
            for seq_len in [1024, 2048]:
                for bs in [1, 2]:
                    feasible, _ = is_configuration_feasible(bs, seq_len, model_id, device)
                    if feasible:
                        configs["aggressive"].append({"batch_size": bs, "seq_length": seq_len})
        
        # Large GPU (>16GB)
        else:
            for seq_len in seq_lengths:
                for bs in [1, 2, 4, 8, 16]:
                    feasible, _ = is_configuration_feasible(bs, seq_len, model_id, device)
                    if feasible:
                        configs["recommended"].append({"batch_size": bs, "seq_length": seq_len})
            
            # Aggressive settings
            if vram_gb >= 24:  # Very large GPU
                for seq_len in [1024, 2048]:
                    for bs in [16, 32]:
                        feasible, _ = is_configuration_feasible(bs, seq_len, model_id, device)
                        if feasible:
                            configs["aggressive"].append({"batch_size": bs, "seq_length": seq_len})
    else:
        # CPU configurations - more limited but should work
        configs["recommended"] = [
            {"batch_size": 1, "seq_length": 128},
            {"batch_size": 1, "seq_length": 256},
            {"batch_size": 2, "seq_length": 128}
        ]
        
        configs["aggressive"] = [
            {"batch_size": 2, "seq_length": 256},
            {"batch_size": 4, "seq_length": 128},
            {"batch_size": 1, "seq_length": 512}
        ]
    
    return configs

class Timer:
    """Simple timer context manager for benchmarking code blocks."""
    
    def __init__(self, name=None):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
            logger.info(f"{self.name} took {self.interval:.4f} seconds")