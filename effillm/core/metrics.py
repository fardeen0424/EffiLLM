# EffiLLM/effillm/core/metrics.py
import time
import torch
import numpy as np
import psutil
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryMetrics:
    """Store memory-related metrics during benchmark runs."""
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
        }
        if self.vram_used_gb is not None:
            result["vram_used_gb"] = self.vram_used_gb
        if self.vram_total_gb is not None:
            result["vram_total_gb"] = self.vram_total_gb
        return result

@dataclass
class LatencyMetrics:
    """Store latency-related metrics during benchmark runs."""
    measurements: List[float] = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        """Get mean latency."""
        return np.mean(self.measurements) if self.measurements else 0.0
    
    @property
    def min(self) -> float:
        """Get minimum latency."""
        return np.min(self.measurements) if self.measurements else 0.0
    
    @property
    def max(self) -> float:
        """Get maximum latency."""
        return np.max(self.measurements) if self.measurements else 0.0
    
    @property
    def std(self) -> float:
        """Get standard deviation of latency."""
        return np.std(self.measurements) if len(self.measurements) > 1 else 0.0
    
    @property
    def median(self) -> float:
        """Get median latency."""
        return np.median(self.measurements) if self.measurements else 0.0
    
    @property
    def p95(self) -> float:
        """Get 95th percentile latency."""
        return np.percentile(self.measurements, 95) if len(self.measurements) > 1 else self.max
    
    @property
    def p99(self) -> float:
        """Get 99th percentile latency."""
        return np.percentile(self.measurements, 99) if len(self.measurements) > 1 else self.max
    
    def add(self, latency: float) -> None:
        """Add a latency measurement."""
        self.measurements.append(latency)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "median": self.median,
            "p95": self.p95,
            "p99": self.p99,
            "samples": len(self.measurements)
        }

@dataclass
class ThroughputMetrics:
    """Store throughput-related metrics during benchmark runs."""
    tokens_per_second: List[float] = field(default_factory=list)
    batch_size: int = 1
    
    @property
    def mean_throughput(self) -> float:
        """Get mean throughput."""
        return np.mean(self.tokens_per_second) if self.tokens_per_second else 0.0
    
    @property
    def mean_per_instance(self) -> float:
        """Get mean throughput per instance (per batch item)."""
        return self.mean_throughput / self.batch_size if self.batch_size > 0 else 0.0
    
    def add(self, tokens: int, seconds: float) -> None:
        """Add a throughput measurement."""
        if seconds > 0:
            self.tokens_per_second.append(tokens / seconds)
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "tokens_per_second": self.mean_throughput,
            "tokens_per_second_per_instance": self.mean_per_instance,
            "samples": len(self.tokens_per_second)
        }

class MetricsCollector:
    """Collects and analyzes performance metrics during benchmarking."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Initialize PYNVML for GPU metrics if available
        if self.device == "cuda" and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_metrics_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize GPU metrics: {e}")
                self.gpu_metrics_available = False
        else:
            self.gpu_metrics_available = False
    
    def measure_memory(self) -> MemoryMetrics:
        """Measure current memory usage."""
        metrics = MemoryMetrics(
            ram_used_gb=psutil.Process().memory_info().rss / (1024 ** 3),
            ram_total_gb=psutil.virtual_memory().total / (1024 ** 3)
        )
        
        if self.device == "cuda" and self.gpu_metrics_available:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                metrics.vram_used_gb = mem_info.used / (1024 ** 3)
                metrics.vram_total_gb = mem_info.total / (1024 ** 3)
            except Exception as e:
                logger.warning(f"Failed to get GPU memory info: {e}")
        
        return metrics
    
    def measure_memory_impact(
        self, 
        baseline: Optional[MemoryMetrics] = None
    ) -> Dict[str, float]:
        """
        Measure memory impact compared to a baseline.
        If baseline is None, returns current memory usage.
        """
        current = self.measure_memory()
        
        if baseline is None:
            return current.as_dict()
        
        impact = {
            "ram_used_gb": current.ram_used_gb - baseline.ram_used_gb,
        }
        
        if current.vram_used_gb is not None and baseline.vram_used_gb is not None:
            impact["vram_used_gb"] = current.vram_used_gb - baseline.vram_used_gb
        
        return impact
    
    def measure_inference_latency(
        self,
        model,
        inputs,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> LatencyMetrics:
        """
        Measure inference latency for a model.
        Returns time to generate first output token.
        """
        metrics = LatencyMetrics()
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(**inputs)
        
        # Synchronize before starting measurements
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Actual measurement runs
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(**inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.time()
            metrics.add(end_time - start_time)
        
        return metrics
    
    def measure_generation_throughput(
        self,
        model,
        input_ids,
        attention_mask,
        generation_params: Dict[str, Any],
        num_runs: int = 5,
        warmup_runs: int = 2
    ) -> Tuple[ThroughputMetrics, LatencyMetrics]:
        """
        Measure generation throughput and latency.
        Returns tokens per second and generation time.
        """
        batch_size = input_ids.shape[0]
        inputs_length = input_ids.shape[1]
        
        throughput = ThroughputMetrics(batch_size=batch_size)
        latency = LatencyMetrics()
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=10)
        
        # Synchronize before starting measurements
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Actual measurement runs
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    **generation_params
                )
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                
            end_time = time.time()
            
            # Calculate metrics
            generation_time = end_time - start_time
            output_length = outputs.shape[1]
            new_tokens = output_length - inputs_length
            total_new_tokens = new_tokens * batch_size
            
            throughput.add(total_new_tokens, generation_time)
            latency.add(generation_time)
        
        return throughput, latency
    
    def measure_peak_memory_usage(
        self,
        task,
        baseline: Optional[MemoryMetrics] = None,
        interval: float = 0.1,
        max_time: float = 60.0
    ) -> Dict[str, float]:
        """
        Monitor and measure peak memory usage during a task.
        
        Args:
            task: Callable that performs the task to monitor
            baseline: Optional baseline memory metrics
            interval: Sampling interval in seconds
            max_time: Maximum time to monitor in seconds
            
        Returns:
            Dictionary with peak memory usage statistics
        """
        if baseline is None:
            baseline = self.measure_memory()
        
        peak_ram = 0.0
        peak_vram = 0.0
        
        # Start memory monitoring in a separate thread
        import threading
        import queue
        
        should_stop = threading.Event()
        memory_queue = queue.Queue()
        
        def memory_monitor():
            while not should_stop.is_set():
                current = self.measure_memory()
                
                ram_usage = current.ram_used_gb - baseline.ram_used_gb
                peak_ram_local = max(peak_ram, ram_usage)
                
                vram_usage = 0.0
                if current.vram_used_gb is not None and baseline.vram_used_gb is not None:
                    vram_usage = current.vram_used_gb - baseline.vram_used_gb
                peak_vram_local = max(peak_vram, vram_usage)
                
                memory_queue.put((peak_ram_local, peak_vram_local))
                
                time.sleep(interval)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=memory_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute the task with timeout
        task_thread = threading.Thread(target=task)
        task_thread.daemon = True
        
        start_time = time.time()
        task_thread.start()
        
        # Wait for task to complete or timeout
        while task_thread.is_alive() and time.time() - start_time < max_time:
            try:
                peak_ram_new, peak_vram_new = memory_queue.get(timeout=0.5)
                peak_ram = max(peak_ram, peak_ram_new)
                peak_vram = max(peak_vram, peak_vram_new)
            except queue.Empty:
                pass
        
        # Stop monitoring
        should_stop.set()
        
        # Wait for threads to finish
        if task_thread.is_alive():
            logger.warning(f"Task did not complete within {max_time} seconds")
        else:
            task_thread.join()
        
        monitor_thread.join(timeout=1.0)
        
        # Process any remaining items in the queue
        while not memory_queue.empty():
            peak_ram_new, peak_vram_new = memory_queue.get_nowait()
            peak_ram = max(peak_ram, peak_ram_new)
            peak_vram = max(peak_vram, peak_vram_new)
        
        result = {
            "peak_ram_gb": peak_ram,
        }
        
        if self.gpu_metrics_available:
            result["peak_vram_gb"] = peak_vram
        
        return result