# EffiLLM/effillm/reporting/compare.py
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import List, Dict, Optional
from pathlib import Path
import os

def load_results(filepath: str) -> Dict:
    """Load benchmark results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_models(result_files: List[str], metric: str = "throughput", output_path: Optional[str] = None):
    """
    Compare multiple models on a specific metric.
    
    Args:
        result_files: List of paths to result JSON files
        metric: Metric to compare ("throughput", "latency", "memory")
        output_path: Path to save the comparison plot
    """
    results = []
    model_names = []
    
    for file in result_files:
        result = load_results(file)
        results.append(result)
        
        # Extract model name
        model_id = result.get("model_id", Path(file).stem)
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        model_names.append(model_name)
    
    # Get common configurations
    configs = set()
    for result in results:
        if "inference" in result:
            configs.update(result["inference"].keys())
    
    configs = sorted(list(configs))
    
    # Extract metric data
    model_data = []
    
    for result in results:
        config_data = []
        
        for config in configs:
            if "inference" in result and config in result["inference"]:
                if metric == "throughput":
                    value = result["inference"][config]["throughput"]["tokens_per_second"]
                elif metric == "latency":
                    value = result["inference"][config]["time_to_first_token"]["mean"] * 1000  # ms
                elif metric == "memory":
                    value = result["inference"][config]["memory"]["impact"].get("vram_used_gb", 
                           result["inference"][config]["memory"]["impact"].get("ram_used_gb", 0))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                    
                config_data.append(value)
            else:
                config_data.append(0)  # Missing data
                
        model_data.append(config_data)
    
    # Create grouped bar chart
    fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
    
    x = np.arange(len(configs))
    width = 0.8 / len(results)
    
    for i, (data, name) in enumerate(zip(model_data, model_names)):
        offset = i - len(results)/2 + 0.5
        ax.bar(x + offset*width, data, width, label=name)
    
    # Add labels and legend
    metric_labels = {
        "throughput": "Tokens per second",
        "latency": "Latency (ms)",
        "memory": "Memory Usage (GB)"
    }
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f'{metric.title()} Comparison Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return fig

def generate_comparison_report(result_files: List[str], output_dir: str):
    """Generate a complete comparison report for multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparison plots
    throughput_path = os.path.join(output_dir, "throughput_comparison.png")
    compare_models(result_files, "throughput", throughput_path)
    
    latency_path = os.path.join(output_dir, "latency_comparison.png")
    compare_models(result_files, "latency", latency_path)
    
    memory_path = os.path.join(output_dir, "memory_comparison.png")
    compare_models(result_files, "memory", memory_path)
    
    # Generate HTML report
    model_names = []
    for file in result_files:
        result = load_results(file)
        model_id = result.get("model_id", Path(file).stem)
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        model_names.append(model_name)
        
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EffiLLM Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .figure {{ margin: 20px 0; }}
            .figure img {{ max-width: 100%; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>EffiLLM Model Comparison</h1>
        <p>Models compared: {', '.join(model_names)}</p>
        
        <h2>Throughput Comparison</h2>
        <div class="figure">
            <img src="throughput_comparison.png" alt="Throughput Comparison">
        </div>
        
        <h2>Latency Comparison</h2>
        <div class="figure">
            <img src="latency_comparison.png" alt="Latency Comparison">
        </div>
        
        <h2>Memory Usage Comparison</h2>
        <div class="figure">
            <img src="memory_comparison.png" alt="Memory Usage Comparison">
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "comparison_report.html"), "w") as f:
        f.write(html_report)
        
    return os.path.join(output_dir, "comparison_report.html")