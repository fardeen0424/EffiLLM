# EffiLLM/effillm/reporting/visualize.py
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os

def plot_throughput_comparison(results: Dict, output_path: Optional[str] = None):
    """Plot throughput comparison across different configurations."""
    if "inference" not in results:
        raise ValueError("No inference results found")
        
    configs = []
    throughputs = []
    
    for config, data in results["inference"].items():
        configs.append(config)
        throughputs.append(data["throughput"]["tokens_per_second"])
    
    # Sort by throughput
    sorted_indices = np.argsort(throughputs)
    configs = [configs[i] for i in sorted_indices]
    throughputs = [throughputs[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(configs, throughputs)
    plt.xlabel("Tokens per second")
    plt.ylabel("Configuration")
    plt.title(f"Throughput Comparison: {results['model_id']}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
        
def plot_memory_usage(results: Dict, output_path: Optional[str] = None):
    """Plot memory usage across different configurations."""
    if "inference" not in results:
        raise ValueError("No inference results found")
        
    configs = []
    ram_usage = []
    vram_usage = []
    
    for config, data in results["inference"].items():
        configs.append(config)
        ram_impact = data["memory"]["impact"].get("ram_used_gb", 0)
        ram_usage.append(ram_impact)
        
        vram_impact = data["memory"]["impact"].get("vram_used_gb", 0)
        if vram_impact:
            vram_usage.append(vram_impact)
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    plt.bar(x, ram_usage, width, label='RAM')
    if vram_usage:
        plt.bar(x + width, vram_usage, width, label='VRAM')
    
    plt.xlabel('Configuration')
    plt.ylabel('Memory Usage (GB)')
    plt.title(f'Memory Impact: {results["model_id"]}')
    plt.xticks(x + width/2, configs, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def generate_summary_report(results: Dict, output_dir: str):
    """Generate a complete visual summary report of benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate throughput plot
    throughput_path = os.path.join(output_dir, "throughput.png")
    plot_throughput_comparison(results, throughput_path)
    
    # Generate memory usage plot
    memory_path = os.path.join(output_dir, "memory_usage.png")
    plot_memory_usage(results, memory_path)
    
    # Generate latency plot (first token time)
    # TODO: Implement latency visualization
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EffiLLM Benchmark Report: {results["model_id"]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .figure {{ margin: 20px 0; }}
            .figure img {{ max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>EffiLLM Benchmark Report: {results["model_id"]}</h1>
        <p>Device: {results["device"]}</p>
        
        <h2>Throughput Comparison</h2>
        <div class="figure">
            <img src="throughput.png" alt="Throughput Comparison">
        </div>
        
        <h2>Memory Usage</h2>
        <div class="figure">
            <img src="memory_usage.png" alt="Memory Usage">
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_report)
        
    return os.path.join(output_dir, "report.html")