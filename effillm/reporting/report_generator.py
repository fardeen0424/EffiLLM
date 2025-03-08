# EffiLLM/effillm/reporting/report_generator.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import base64
from io import BytesIO
import datetime
from pathlib import Path
import pandas as pd

# Set Matplotlib styling
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8')
sns.set_context("talk")

class BenchmarkReportGenerator:
    """Generates comprehensive HTML benchmark reports with visualizations."""
    
    def __init__(self, 
                 results_files: List[str] = None, 
                 results_data: List[Dict] = None,
                 output_dir: str = "reports"):
        """Initialize with either result files or direct result data."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = []
        if results_files:
            for file in results_files:
                with open(file, 'r') as f:
                    self.results.append(json.load(f))
        elif results_data:
            self.results = results_data
            
        self.model_names = []
        self.model_short_names = []
        self.quantization_info = {}
        
        for result in self.results:
            model_id = result.get("model_id", "unknown")
            full_name = model_id.split('/')[-1] if '/' in model_id else model_id
            
            # Store quantization info
            quant_text = ""
            if "quantization" in result:
                quant = result["quantization"]
                bits = quant.get("bits", 16)
                self.quantization_info[full_name] = f"{bits}bit"
                quant_text = f" ({bits}-bit)"
            else:
                self.quantization_info[full_name] = "fp16"
                quant_text = " (FP16)"
                
            self.model_names.append(full_name + quant_text)
            self.model_short_names.append(full_name)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64-encoded string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)  # Close figure to free memory
        return f"data:image/png;base64,{img_str}"
    
    def generate_summary_stats(self) -> str:
        """Generate a summary of the benchmark results."""
        summary_lines = []
        
        # Get hardware info from the first result
        if self.results and "system_info" in self.results[0]:
            system = self.results[0]["system_info"]
            gpu_info = system.get("gpu_devices", ["Unknown GPU"])[0] if "gpu_devices" in system else "CPU only"
            summary_lines.append(f"• Benchmark run on {gpu_info}")
            
            # Add more system details
            cpu_info = f"{system.get('cpu_count', 'Unknown')} cores, {system.get('processor', 'Unknown')}"
            summary_lines.append(f"• CPU: {cpu_info}")
            
            if "cuda_version" in system:
                summary_lines.append(f"• CUDA: {system.get('cuda_version', 'Unknown')}")
        
        # Summarize models
        if self.model_names:
            summary_lines.append(f"• Models tested: {', '.join(self.model_names)}")
            
        # Summarize configurations
        configs = set()
        for result in self.results:
            if "inference" in result:
                configs.update(result["inference"].keys())
        
        if configs:
            config_count = len(configs)
            batch_sizes = set()
            seq_lengths = set()
            
            for config in configs:
                if "bs" in config and "seq" in config:
                    parts = config.split("_")
                    if len(parts) >= 2:
                        bs_part = parts[0].replace("bs", "")
                        seq_part = parts[1].replace("seq", "")
                        if bs_part.isdigit():
                            batch_sizes.add(int(bs_part))
                        if seq_part.isdigit():
                            seq_lengths.add(int(seq_part))
            
            if batch_sizes:
                summary_lines.append(f"• Batch sizes: {', '.join(map(str, sorted(batch_sizes)))}")
            if seq_lengths:
                summary_lines.append(f"• Sequence lengths: {', '.join(map(str, sorted(seq_lengths)))}")
        
        # Average throughput improvement or comparison
        if len(self.results) > 1:
            # Compare models based on throughput
            avg_throughputs = {}
            
            for i, result in enumerate(self.results):
                model_name = self.model_names[i]
                total_throughput = 0
                count = 0
                
                if "inference" in result:
                    for config, data in result["inference"].items():
                        if "throughput" in data and "tokens_per_second" in data["throughput"]:
                            total_throughput += data["throughput"]["tokens_per_second"]
                            count += 1
                
                if count > 0:
                    avg_throughputs[model_name] = total_throughput / count
            
            # Find best and worst performers
            if avg_throughputs:
                sorted_models = sorted(avg_throughputs.items(), key=lambda x: x[1], reverse=True)
                best_model, best_value = sorted_models[0]
                worst_model, worst_value = sorted_models[-1]
                
                if worst_value > 0:
                    speedup = best_value / worst_value
                    summary_lines.append(f"• Best performer ({best_model}) is {speedup:.1f}x faster than {worst_model}")
                    
                # Quantization impact if available
                quant_models = [name for name in self.model_short_names if name in self.quantization_info and self.quantization_info[name] != "fp16"]
                base_models = [name for name in self.model_short_names if name in self.quantization_info and self.quantization_info[name] == "fp16"]
                
                if quant_models and base_models:
                    matching_models = set(quant_models).intersection(set(base_models))
                    if matching_models:
                        summary_lines.append(f"• Quantization effects measured for {len(matching_models)} models")
        
        return "<p>" + "</p><p>".join(summary_lines) + "</p>"
    
    def generate_throughput_chart(self) -> Dict[str, Any]:
        """Generate throughput comparison chart and data table."""
        # Extract throughput data by model and configuration
        throughputs = {}
        configs = set()
        
        for i, result in enumerate(self.results):
            model_name = self.model_names[i]
            throughputs[model_name] = {}
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "throughput" in data:
                        throughputs[model_name][config] = data["throughput"]["tokens_per_second"]
        
        configs = sorted(list(configs))
        
        # Create multiple visualizations for throughput
        
        # 1. Bar chart comparison by configuration
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        x = np.arange(len(configs))
        bar_width = 0.8 / len(throughputs) if throughputs else 0.8
        
        for i, (model, model_data) in enumerate(throughputs.items()):
            model_throughputs = [model_data.get(config, 0) for config in configs]
            offset = (i - len(throughputs)/2 + 0.5) * bar_width
            ax1.bar(x + offset, model_throughputs, bar_width, label=model)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Tokens per second')
        ax1.set_title('Throughput Comparison by Configuration')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        plt.tight_layout()
        
        # 2. Horizontal bar chart - average throughput by model
        fig2 = plt.figure(figsize=(10, 5))
        ax2 = fig2.add_subplot(111)
        
        avg_throughputs = {}
        for model, model_data in throughputs.items():
            if model_data:
                avg_throughputs[model] = sum(model_data.values()) / len(model_data)
        
        models_sorted = sorted(avg_throughputs.items(), key=lambda x: x[1])
        models = [x[0] for x in models_sorted]
        avgs = [x[1] for x in models_sorted]
        
        # Use color gradient based on performance
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.2, 0.8, len(models)))
        
        ax2.barh(models, avgs, color=colors)
        ax2.set_xlabel('Tokens per second (avg)')
        ax2.set_title('Average Throughput by Model')
        
        # Add values at end of bars
        for i, v in enumerate(avgs):
            ax2.text(v + 0.5, i, f"{v:.1f}", va='center')
        
        plt.tight_layout()
        
        # 3. Line plot - throughput scaling with batch size
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        
        # Extract batch sizes from configs
        batch_size_data = {}
        for model, model_data in throughputs.items():
            batch_size_data[model] = {}
            
            for config, value in model_data.items():
                if "bs" in config and "_seq" in config:
                    parts = config.split("_")
                    batch = int(parts[0].replace("bs", ""))
                    seq = int(parts[1].replace("seq", ""))
                    
                    if seq not in batch_size_data[model]:
                        batch_size_data[model][seq] = {}
                    
                    batch_size_data[model][seq][batch] = value
        
        # Plot scaling for a consistent sequence length
        seq_to_plot = None
        # Find the most common sequence length
        all_seqs = []
        for model_data in batch_size_data.values():
            all_seqs.extend(model_data.keys())
        
        if all_seqs:
            from collections import Counter
            seq_to_plot = Counter(all_seqs).most_common(1)[0][0]
            
            for model, model_data in batch_size_data.items():
                if seq_to_plot in model_data:
                    seq_data = model_data[seq_to_plot]
                    batches = sorted(seq_data.keys())
                    throughputs_by_batch = [seq_data[b] for b in batches]
                    
                    ax3.plot(batches, throughputs_by_batch, 'o-', label=model, linewidth=2, markersize=8)
            
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Tokens per second')
            ax3.set_title(f'Throughput Scaling with Batch Size (seq_len={seq_to_plot})')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
        
        plt.tight_layout()
        
        # Convert to base64 for HTML embedding
        chart_img1 = self._fig_to_base64(fig1)
        chart_img2 = self._fig_to_base64(fig2)
        chart_img3 = self._fig_to_base64(fig3)
        
        # Create data for the table
        table_data = {
            "headers": ["Model"] + configs,
            "rows": []
        }
        
        for model, model_data in throughputs.items():
            row = [model]
            for config in configs:
                value = model_data.get(config, "N/A")
                row.append(f"{value:.2f}" if isinstance(value, (int, float)) else value)
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = self._generate_throughput_summary(throughputs, configs)
        
        return {
            "charts": [chart_img1, chart_img2, chart_img3],
            "table": table_data,
            "summary": summary
        }
    
    def _generate_throughput_summary(self, throughputs, configs) -> str:
        """Generate a summary of throughput comparisons."""
        if not throughputs or not configs:
            return "<p>No throughput data available for comparison.</p>"
        
        summary_lines = []
        
        # Find best performing model and configuration
        best_model = None
        best_config = None
        best_throughput = 0
        
        for model, model_data in throughputs.items():
            for config, value in model_data.items():
                if value > best_throughput:
                    best_throughput = value
                    best_model = model
                    best_config = config
        
        if best_model:
            summary_lines.append(f"• Highest throughput: {best_throughput:.2f} tokens/sec with {best_model} on {best_config}")
        
        # Compare batch size impact
        batch_impacts = {}
        for config in configs:
            if "bs" in config and "_seq" in config:
                parts = config.split("_")
                batch = parts[0].replace("bs", "")
                seq = parts[1].replace("seq", "")
                
                for model, model_data in throughputs.items():
                    if config in model_data:
                        key = f"model={model}, seq={seq}"
                        if key not in batch_impacts:
                            batch_impacts[key] = []
                        batch_impacts[key].append((int(batch), model_data[config]))
        
        for key, values in batch_impacts.items():
            if len(values) > 1:
                values.sort(key=lambda x: x[0])  # Sort by batch size
                min_batch, min_throughput = values[0]
                max_batch, max_throughput = values[-1]
                
                if min_throughput > 0:
                    speedup = max_throughput / min_throughput
                    summary_lines.append(f"• Increasing batch size from {min_batch} to {max_batch} for {key}: {speedup:.2f}x speedup")
        
        # Compare models if multiple
        if len(throughputs) > 1:
            # Average across configs
            avg_throughputs = {}
            for model, model_data in throughputs.items():
                if model_data:
                    avg_throughputs[model] = sum(model_data.values()) / len(model_data)
            
            # Sort by throughout
            models_by_throughput = sorted(avg_throughputs.items(), key=lambda x: x[1], reverse=True)
            
            if len(models_by_throughput) >= 2:
                best_model, best_avg = models_by_throughput[0]
                second_model, second_avg = models_by_throughput[1]
                
                if second_avg > 0:
                    relative_perf = best_avg / second_avg
                    summary_lines.append(f"• {best_model} is {relative_perf:.2f}x faster than {second_model} on average")
        
        # Check for quantization impact
        quant_models = {}
        base_models = {}
        
        # Group models by their base name and quantization
        for model_name in throughputs.keys():
            short_name = model_name.split(" (")[0]
            if "(FP16)" in model_name:
                base_models[short_name] = model_name
            elif "bit)" in model_name:
                quant_models[short_name] = model_name
        
        # Compare performance for same model with different quantization
        for short_name in base_models.keys():
            if short_name in quant_models:
                base_model = base_models[short_name]
                quant_model = quant_models[short_name]
                
                if base_model in avg_throughputs and quant_model in avg_throughputs:
                    base_perf = avg_throughputs[base_model]
                    quant_perf = avg_throughputs[quant_model]
                    
                    if base_perf > 0:
                        perf_change = (quant_perf / base_perf - 1) * 100
                        direction = "faster" if perf_change > 0 else "slower"
                        summary_lines.append(f"• Quantization impact on {short_name}: {abs(perf_change):.1f}% {direction}")
        
        return "<p>" + "</p><p>".join(summary_lines) + "</p>"
    
    def generate_latency_chart(self) -> Dict[str, Any]:
        """Generate latency comparison chart and data table."""
        # Extract latency data by model and configuration
        latencies = {}
        configs = set()
        
        for i, result in enumerate(self.results):
            model_name = self.model_names[i]
            latencies[model_name] = {}
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "time_to_first_token" in data:
                        # Convert to milliseconds
                        latencies[model_name][config] = data["time_to_first_token"]["mean"] * 1000
        
        configs = sorted(list(configs))
        
        # Create multiple visualizations for latency
        
        # 1. Bar chart for latency by configuration
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        x = np.arange(len(configs))
        bar_width = 0.8 / len(latencies) if latencies else 0.8
        
        for i, (model, model_data) in enumerate(latencies.items()):
            model_latencies = [model_data.get(config, 0) for config in configs]
            offset = (i - len(latencies)/2 + 0.5) * bar_width
            ax1.bar(x + offset, model_latencies, bar_width, label=model)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Time to First Token Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        plt.tight_layout()
        
        # 2. Box plot comparing latency distributions
        # Create a DataFrame for the box plot
        latency_data = []
        
        for result_idx, result in enumerate(self.results):
            model_name = self.model_names[result_idx]
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    if "time_to_first_token" in data:
                        ttft_mean = data["time_to_first_token"]["mean"] * 1000
                        ttft_std = data["time_to_first_token"]["std"] * 1000
                        ttft_min = data["time_to_first_token"]["min"] * 1000
                        ttft_max = data["time_to_first_token"]["max"] * 1000
                        
                        latency_data.append({
                            "Model": model_name,
                            "Latency (ms)": ttft_mean
                        })
        
        if latency_data:
            df = pd.DataFrame(latency_data)
            
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            
            sns.boxplot(x="Model", y="Latency (ms)", data=df, ax=ax2)
            ax2.set_title('Latency Distribution by Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        else:
            # Create empty figure if no data
            fig2 = plt.figure(figsize=(10, 6))
            ax2 = fig2.add_subplot(111)
            ax2.text(0.5, 0.5, "No latency distribution data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
        
        # 3. Heatmap showing latency by batch size and sequence length
        heatmap_data = {}
        for model_name, model_latencies in latencies.items():
            batch_seq_data = {}
            for config, value in model_latencies.items():
                if "bs" in config and "_seq" in config:
                    parts = config.split("_")
                    batch = int(parts[0].replace("bs", ""))
                    seq = int(parts[1].replace("seq", ""))
                    batch_seq_data[(batch, seq)] = value
            
            if batch_seq_data:
                heatmap_data[model_name] = batch_seq_data
        
        # Create heatmap for the first model (most representative)
        fig3 = plt.figure(figsize=(12, 8))
        
        if heatmap_data:
            model_count = len(heatmap_data)
            rows = (model_count + 1) // 2  # Calculate rows needed
            
            for i, (model, data) in enumerate(heatmap_data.items()):
                ax = fig3.add_subplot(rows, 2, i+1)
                
                # Extract batch sizes and sequence lengths
                batch_sizes = sorted(list(set([b for b, _ in data.keys()])))
                seq_lengths = sorted(list(set([s for _, s in data.keys()])))
                
                # Create matrix for heatmap
                matrix = np.zeros((len(batch_sizes), len(seq_lengths)))
                for b_idx, b in enumerate(batch_sizes):
                    for s_idx, s in enumerate(seq_lengths):
                        if (b, s) in data:
                            matrix[b_idx, s_idx] = data[(b, s)]
                
                # Plot heatmap
                im = ax.imshow(matrix, cmap='YlOrRd')
                
                # Set ticks and labels
                ax.set_xticks(np.arange(len(seq_lengths)))
                ax.set_yticks(np.arange(len(batch_sizes)))
                ax.set_xticklabels(seq_lengths)
                ax.set_yticklabels(batch_sizes)
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Add value annotations
                for b_idx, b in enumerate(batch_sizes):
                    for s_idx, s in enumerate(seq_lengths):
                        if (b, s) in data:
                            text = ax.text(s_idx, b_idx, f"{data[(b, s)]:.1f}",
                                        ha="center", va="center", color="black")
                
                ax.set_title(f"Latency (ms): {model}")
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel("Batch Size")
            
            plt.tight_layout()
        else:
            ax = fig3.add_subplot(111)
            ax.text(0.5, 0.5, "No heatmap data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
        
        # Convert to base64 for HTML embedding
        chart_img1 = self._fig_to_base64(fig1)
        chart_img2 = self._fig_to_base64(fig2)
        chart_img3 = self._fig_to_base64(fig3)
        
        # Create data for the table
        table_data = {
            "headers": ["Model"] + configs,
            "rows": []
        }
        
        for model, model_data in latencies.items():
            row = [model]
            for config in configs:
                value = model_data.get(config, "N/A")
                row.append(f"{value:.2f}" if isinstance(value, (int, float)) else value)
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = self._generate_latency_summary(latencies, configs)
        
        return {
            "charts": [chart_img1, chart_img2, chart_img3],
            "table": table_data,
            "summary": summary
        }
    
    def _generate_latency_summary(self, latencies, configs) -> str:
        """Generate a summary of latency comparisons."""
        if not latencies or not configs:
            return "<p>No latency data available for comparison.</p>"
        
        summary_lines = []
        
        # Find lowest latency model and configuration
        best_model = None
        best_config = None
        best_latency = float('inf')
        
        for model, model_data in latencies.items():
            for config, value in model_data.items():
                if value < best_latency:
                    best_latency = value
                    best_model = model
                    best_config = config
        
        if best_model:
            summary_lines.append(f"• Lowest latency: {best_latency:.2f} ms with {best_model} on {best_config}")
        
        # Compare models if multiple
        if len(latencies) > 1:
            # Average across configs
            avg_latencies = {}
            for model, model_data in latencies.items():
                if model_data:
                    avg_latencies[model] = sum(model_data.values()) / len(model_data)
            
            # Sort by latency (lower is better)
            models_by_latency = sorted(avg_latencies.items(), key=lambda x: x[1])
            
            if len(models_by_latency) >= 2:
                best_model, best_avg = models_by_latency[0]
                second_model, second_avg = models_by_latency[1]
                
                if best_avg > 0:
                    relative_latency = second_avg / best_avg
                    summary_lines.append(f"• {best_model} has {relative_latency:.2f}x lower latency than {second_model} on average")
                    
        # Look at batch size impact on latency
        batch_impacts = {}
        for config in configs:
            if "bs" in config and "_seq" in config:
                parts = config.split("_")
                batch = parts[0].replace("bs", "")
                seq = parts[1].replace("seq", "")
                
                for model, model_data in latencies.items():
                    if config in model_data:
                        key = f"model={model}, seq={seq}"
                        if key not in batch_impacts:
                            batch_impacts[key] = []
                        batch_impacts[key].append((int(batch), model_data[config]))
        
        # See if increasing batch size affects latency
        for key, values in batch_impacts.items():
            if len(values) > 1:
                values.sort(key=lambda x: x[0])  # Sort by batch size
                min_batch, min_latency = values[0]
                max_batch, max_latency = values[-1]
                
                if min_latency > 0:
                    latency_change = (max_latency - min_latency) / min_latency
                    if abs(latency_change) > 0.1:  # Only report if change is significant
                        direction = "increases" if latency_change > 0 else "decreases"
                        summary_lines.append(f"• Increasing batch size from {min_batch} to {max_batch} for {key}: {direction} latency by {abs(latency_change):.1f}%")
        
        # Check for quantization impact on latency
        quant_models = {}
        base_models = {}
        
        # Group models by their base name and quantization
        for model_name in latencies.keys():
            short_name = model_name.split(" (")[0]
            if "(FP16)" in model_name:
                base_models[short_name] = model_name
            elif "bit)" in model_name:
                quant_models[short_name] = model_name
        
        # Compare latency for same model with different quantization
        for short_name in base_models.keys():
            if short_name in quant_models:
                base_model = base_models[short_name]
                quant_model = quant_models[short_name]
                
                if base_model in avg_latencies and quant_model in avg_latencies:
                    base_latency = avg_latencies[base_model]
                    quant_latency = avg_latencies[quant_model]
                    
                    if base_latency > 0:
                        latency_change = (quant_latency / base_latency - 1) * 100
                        direction = "higher" if latency_change > 0 else "lower"
                        summary_lines.append(f"• Quantization impact on {short_name} latency: {abs(latency_change):.1f}% {direction}")
        
        return "<p>" + "</p><p>".join(summary_lines) + "</p>"
    
    def generate_memory_chart(self) -> Dict[str, Any]:
        """Generate memory usage comparison chart and data table."""
        # Extract memory data by model and configuration
        memory_usage = {}
        configs = set()
        
        for i, result in enumerate(self.results):
            model_name = self.model_names[i]
            memory_usage[model_name] = {
                "loading": {},
                "inference": {}
            }
            
            # Get loading memory
            if "loading" in result and "memory_impact" in result["loading"]:
                mem_impact = result["loading"]["memory_impact"]
                ram = mem_impact.get("ram_used_gb", 0)
                vram = mem_impact.get("vram_used_gb", 0)
                memory_usage[model_name]["loading"] = {
                    "ram": ram,
                    "vram": vram if vram else 0
                }
            
            # Get inference memory
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "memory" in data and "impact" in data["memory"]:
                        mem_impact = data["memory"]["impact"]
                        ram = mem_impact.get("ram_used_gb", 0)
                        vram = mem_impact.get("vram_used_gb", 0)
                        memory_usage[model_name]["inference"][config] = {
                            "ram": ram, 
                            "vram": vram if vram else 0
                        }
        
        configs = sorted(list(configs))
        
        # Create multiple visualizations for memory usage
        
        # 1. Stacked bar chart for total memory (loading + inference)
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        x = np.arange(len(self.model_names))
        width = 0.35
        
        # Get loading and inference memory values
        loading_vram = []
        loading_ram = []
        inference_vram = []
        inference_ram = []
        
        for model in self.model_names:
            # Loading memory
            if model in memory_usage and "loading" in memory_usage[model]:
                loading_vram.append(memory_usage[model]["loading"].get("vram", 0))
                loading_ram.append(memory_usage[model]["loading"].get("ram", 0))
            else:
                loading_vram.append(0)
                loading_ram.append(0)
            
            # Average inference memory
            inf_vram = 0
            inf_ram = 0
            count = 0
            
            if model in memory_usage and "inference" in memory_usage[model]:
                for config, data in memory_usage[model]["inference"].items():
                    inf_vram += data.get("vram", 0)
                    inf_ram += data.get("ram", 0)
                    count += 1
                
                if count > 0:
                    inf_vram /= count
                    inf_ram /= count
            
            inference_vram.append(inf_vram)
            inference_ram.append(inf_ram)
        
        # Use VRAM if available, otherwise RAM
        loading_mem = loading_vram if any(loading_vram) else loading_ram
        inference_mem = inference_vram if any(inference_vram) else inference_ram
        
        # Create the stacked bars
        ax1.bar(x, loading_mem, width, label='Model Loading')
        ax1.bar(x, inference_mem, width, bottom=loading_mem, label='Inference (avg)')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Memory Usage (GB)')
        mem_type = "VRAM" if any(loading_vram) or any(inference_vram) else "RAM"
        ax1.set_title(f'{mem_type} Usage by Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.model_names, rotation=45, ha='right')
        ax1.legend()
        
        # Annotate bars with values
        for i, v in enumerate(loading_mem):
            if v > 0:
                ax1.text(i, v/2, f"{v:.1f}", ha='center', va='center')
        
        for i, v in enumerate(inference_mem):
            if v > 0:
                ax1.text(i, loading_mem[i] + v/2, f"{v:.1f}", ha='center', va='center')
        
        plt.tight_layout()
        
        # 2. Scatter plot: Memory vs Throughput (efficiency visualization)
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        # Extract throughput data
        throughputs = {}
        for i, result in enumerate(self.results):
            model_name = self.model_names[i]
            avg_throughput = 0
            count = 0
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    if "throughput" in data:
                        avg_throughput += data["throughput"]["tokens_per_second"]
                        count += 1
            
            if count > 0:
                throughputs[model_name] = avg_throughput / count
        
        # Create memory vs throughput data
        plot_data = []
        for model in self.model_names:
            if model in throughputs and model in memory_usage:
                total_mem = 0
                if "loading" in memory_usage[model]:
                    mem_type = "vram" if memory_usage[model]["loading"].get("vram", 0) > 0 else "ram"
                    total_mem += memory_usage[model]["loading"].get(mem_type, 0)
                
                plot_data.append({
                    "model": model,
                    "throughput": throughputs[model],
                    "memory": total_mem
                })
        
        if plot_data:
            # Extract data for plotting
            x_data = [d["memory"] for d in plot_data]
            y_data = [d["throughput"] for d in plot_data]
            labels = [d["model"] for d in plot_data]
            
            # Create scatter plot
            scatter = ax2.scatter(x_data, y_data, c=range(len(x_data)), cmap='viridis', 
                                s=100, alpha=0.7)
            
            # Add model labels
            for i, label in enumerate(labels):
                ax2.annotate(label, (x_data[i], y_data[i]),
                            xytext=(5, 5), textcoords='offset points')
            
            # Add trendline if we have enough points
            if len(x_data) > 2:
                try:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax2.plot(x_data, p(x_data), "r--", alpha=0.5)
                except np.linalg.LinAlgError:
                    pass
            
            ax2.set_xlabel('Memory Usage (GB)')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Memory Efficiency: Throughput vs Memory Usage')
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for memory-throughput comparison", 
                    horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # 3. Donut chart showing memory distribution
        fig3 = plt.figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        
        # Use the model with the most memory data
        model_to_plot = None
        best_config = None
        most_configs = 0
        
        for model, data in memory_usage.items():
            if "inference" in data and len(data["inference"]) > most_configs:
                model_to_plot = model
                most_configs = len(data["inference"])
                # Find config with highest memory usage
                best_mem = 0
                for config, mem_data in data["inference"].items():
                    total_mem = mem_data.get("vram", 0) + mem_data.get("ram", 0)
                    if total_mem > best_mem:
                        best_mem = total_mem
                        best_config = config
        
        if model_to_plot and best_config and best_config in memory_usage[model_to_plot]["inference"]:
            # Get memory before and after
            if "inference" in memory_usage[model_to_plot]:
                # Create the donut chart
                for result_idx, result in enumerate(self.results):
                    if self.model_names[result_idx] == model_to_plot:
                        if "inference" in result and best_config in result["inference"]:
                            config_data = result["inference"][best_config]
                            if "memory" in config_data:
                                mem_data = config_data["memory"]
                                if "before" in mem_data and "after" in mem_data:
                                    # Get memory values
                                    vram_before = mem_data["before"].get("vram_used_gb", 0)
                                    vram_after = mem_data["after"].get("vram_used_gb", 0)
                                    ram_before = mem_data["before"].get("ram_used_gb", 0)
                                    ram_after = mem_data["after"].get("ram_used_gb", 0)
                                    
                                    # Create donut chart
                                    if vram_before > 0 or vram_after > 0:
                                        # Use VRAM if available
                                        mem_type = "VRAM"
                                        mem_before = vram_before
                                        mem_after = vram_after
                                    else:
                                        mem_type = "RAM"
                                        mem_before = ram_before
                                        mem_after = ram_after
                                    
                                    # Calculate unused and active memory
                                    active_memory = mem_after
                                    memory_growth = max(0, mem_after - mem_before)
                                    
                                    # Create pie chart
                                    labels = ['Base Memory', 'Growth During Inference']
                                    sizes = [mem_before, memory_growth]
                                    colors = ['#66b3ff', '#ff9999']
                                    explode = (0.1, 0)  # Explode first slice for emphasis
                                    
                                    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', shadow=True, startangle=90)
                                    ax3.axis('equal')  # Equal aspect ratio ensures pie is circular
                                    ax3.set_title(f'Memory Distribution for {model_to_plot}\n({mem_type}, {best_config})')
                                break
        else:
            ax3.text(0.5, 0.5, "Insufficient data for memory distribution chart", 
                    horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
        
        # Convert to base64 for HTML embedding
        chart_img1 = self._fig_to_base64(fig1)
        chart_img2 = self._fig_to_base64(fig2)
        chart_img3 = self._fig_to_base64(fig3)
        
        # Create detailed memory table
        table_data = {
            "headers": ["Model", "Loading Memory (GB)"] + [f"Inference: {config} (GB)" for config in configs],
            "rows": []
        }
        
        for model in self.model_names:
            row = [model]
            
            # Loading memory
            load_mem = 0
            if model in memory_usage and "loading" in memory_usage[model]:
                load_mem_data = memory_usage[model]["loading"]
                load_mem = load_mem_data.get("vram", 0) if load_mem_data.get("vram", 0) > 0 else load_mem_data.get("ram", 0)
            
            row.append(f"{load_mem:.2f}")
            
            # Inference memory by config
            for config in configs:
                inf_mem = "N/A"
                if model in memory_usage and "inference" in memory_usage[model] and config in memory_usage[model]["inference"]:
                    inf_mem_data = memory_usage[model]["inference"][config]
                    inf_mem = inf_mem_data.get("vram", 0) if inf_mem_data.get("vram", 0) > 0 else inf_mem_data.get("ram", 0)
                    inf_mem = f"{inf_mem:.2f}"
                
                row.append(inf_mem)
                
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = self._generate_memory_summary(memory_usage)
        
        return {
            "charts": [chart_img1, chart_img2, chart_img3],
            "table": table_data,
            "summary": summary
        }
    
    def _generate_memory_summary(self, memory_usage) -> str:
        """Generate a summary of memory comparisons."""
        if not memory_usage:
            return "<p>No memory data available for comparison.</p>"
        
        summary_lines = []
        
        # Extract loading memory values
        loading_memory = []
        for model, data in memory_usage.items():
            if "loading" in data:
                load_data = data["loading"]
                mem_value = load_data.get("vram", 0) if load_data.get("vram", 0) > 0 else load_data.get("ram", 0)
                loading_memory.append((model, mem_value))
        
        # Compare loading memory
        loading_memory.sort(key=lambda x: x[1])
        
        if loading_memory:
            min_model, min_memory = loading_memory[0]
            max_model, max_memory = loading_memory[-1]
            
            if min_memory > 0:
                summary_lines.append(f"• {min_model} uses the least memory for loading ({min_memory:.2f} GB)")
                
                if len(loading_memory) > 1 and max_memory > 0:
                    memory_ratio = max_memory / min_memory
                    summary_lines.append(f"• {max_model} requires {memory_ratio:.1f}x more memory than {min_model} for loading")
        
        # Extract average inference memory 
        avg_inference_memory = []
        for model, data in memory_usage.items():
            if "inference" in data and data["inference"]:
                total_inf_mem = 0
                count = 0
                
                for config, inf_data in data["inference"].items():
                    mem_value = inf_data.get("vram", 0) if inf_data.get("vram", 0) > 0 else inf_data.get("ram", 0)
                    total_inf_mem += mem_value
                    count += 1
                
                if count > 0:
                    avg_inf_mem = total_inf_mem / count
                    avg_inference_memory.append((model, avg_inf_mem))
        
        # Compare inference memory
        avg_inference_memory.sort(key=lambda x: x[1])
        
        if avg_inference_memory:
            min_model, min_memory = avg_inference_memory[0]
            max_model, max_memory = avg_inference_memory[-1]
            
            if min_memory > 0:
                summary_lines.append(f"• {min_model} uses the least memory during inference ({min_memory:.2f} GB on average)")
                
                if len(avg_inference_memory) > 1 and max_memory > 0:
                    memory_ratio = max_memory / min_memory
                    summary_lines.append(f"• {max_model} requires {memory_ratio:.1f}x more memory than {min_model} during inference")
        
        # Check for quantization impact on memory
        quant_models = {}
        base_models = {}
        
        # Group models by their base name and quantization
        for model_name in memory_usage.keys():
            short_name = model_name.split(" (")[0]
            if "(FP16)" in model_name:
                base_models[short_name] = model_name
            elif "bit)" in model_name:
                quant_models[short_name] = model_name
        
        # Compare memory usage for same model with different quantization
        for short_name in base_models.keys():
            if short_name in quant_models:
                base_model = base_models[short_name]
                quant_model = quant_models[short_name]
                
                # Find memory values
                base_memory = 0
                quant_memory = 0
                
                for model, mem_value in loading_memory:
                    if model == base_model:
                        base_memory = mem_value
                    elif model == quant_model:
                        quant_memory = mem_value
                
                if base_memory > 0 and quant_memory > 0:
                    memory_saving = (1 - quant_memory / base_memory) * 100
                    summary_lines.append(f"• Quantization reduces {short_name} memory usage by {memory_saving:.1f}%")
        
        return "<p>" + "</p><p>".join(summary_lines) + "</p>"
    
    def save_html_template(template_path):
        """Save the HTML template for reports."""
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
        with open(template_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="">
<head>
<title></title>
<meta content="summary_large_image" name="twitter:card" />
<meta content="website" property="og:type" />
<meta content="" property="og:description" />
<meta content="https://fyi17hw2vf.preview-beefreedesign.com/ovXH" property="og:url" />
<meta content="https://pro-bee-beepro-thumbnail.getbee.io/messages/1358215/1344654/2365586/12468090_large.jpg" property="og:image" />
<meta content="" property="og:title" />
<meta content="" name="description" />
<meta charset="utf-8" />
<meta content="width=device-width" name="viewport" />
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@100;200;300;400;500;600;700;800;900" rel="stylesheet" type="text/css" />
<style>
.bee-row,
.bee-row-content {
position: relative
}
body {
background-color: #FFFFFF;
color: #000000;
font-family: Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif
}
a {
color: #0068A5
}
* {
box-sizing: border-box
}
body,
h1,
h2,
h3,
p {
margin: 0
}
.bee-row-content {
max-width: 1010px;
margin: 0 auto;
display: flex
}
.bee-row-content .bee-col-w3 {
flex-basis: 25%
}
.bee-row-content .bee-col-w6 {
flex-basis: 50%
}
.bee-row-content .bee-col-w9 {
flex-basis: 75%
}
.bee-row-content .bee-col-w12 {
flex-basis: 100%
}
.bee-icon .bee-icon-label-right a {
text-decoration: none
}
.bee-image {
overflow: auto
}
.bee-row-3 .bee-col-1 .bee-block-2,
.bee-row-4 .bee-col-1 .bee-block-2,
.bee-row-5 .bee-col-1 .bee-block-2 {
width: 100%
}
.bee-icon {
display: inline-block;
vertical-align: middle
}
.bee-icon .bee-content {
display: flex;
align-items: center
}
.bee-image img {
display: block;
width: 100%
}
.bee-paragraph {
overflow-wrap: anywhere
}
.bee-table table {
border-collapse: collapse;
width: 100%
}
.bee-table table tbody,
.bee-table table thead {
vertical-align: top
}
.bee-table table td,
.bee-table table th {
padding: 10px;
word-break: break-word
}
@media (max-width:768px) {
.bee-row-content:not(.no_stack) {
display: block
}
}
.bee-row-1,
.bee-row-6 {
background-color: #f2f2f2;
background-repeat: no-repeat
}
.bee-row-1 .bee-row-content,
.bee-row-6 .bee-row-content {
background-color: #f2f2f2;
background-repeat: no-repeat;
color: #000000
}
.bee-row-7,
.bee-row-7 .bee-row-content {
background-color: #ffffff;
background-repeat: no-repeat
}
.bee-row-1 .bee-col-1,
.bee-row-6 .bee-col-1 {
padding-left: 15px;
padding-right: 15px;
padding-top: 5px
}
.bee-row-1 .bee-col-1 .bee-block-1 {
padding-left: 10px;
padding-top: 10px;
width: 100%
}
.bee-row-1 .bee-col-2,
.bee-row-6 .bee-col-2,
.bee-row-7 .bee-col-1 {
padding-bottom: 5px;
padding-top: 5px
}
.bee-row-1 .bee-col-2 .bee-block-1,
.bee-row-2 .bee-col-1 .bee-block-1,
.bee-row-3 .bee-col-1 .bee-block-1,
.bee-row-4 .bee-col-1 .bee-block-1,
.bee-row-5 .bee-col-1 .bee-block-1 {
padding: 10px;
text-align: center;
width: 100%
}
.bee-row-2,
.bee-row-3,
.bee-row-4,
.bee-row-5 {
background-repeat: no-repeat
}
.bee-row-2 .bee-row-content,
.bee-row-3 .bee-row-content,
.bee-row-4 .bee-row-content,
.bee-row-5 .bee-row-content {
background-repeat: no-repeat;
color: #000000
}
.bee-row-2 .bee-col-1,
.bee-row-3 .bee-col-1,
.bee-row-4 .bee-col-1,
.bee-row-5 .bee-col-1 {
padding: 10px 15px 5px
}
.bee-row-2 .bee-col-1 .bee-block-2,
.bee-row-3 .bee-col-1 .bee-block-3,
.bee-row-3 .bee-col-1 .bee-block-4,
.bee-row-4 .bee-col-1 .bee-block-3,
.bee-row-4 .bee-col-1 .bee-block-4,
.bee-row-5 .bee-col-1 .bee-block-3,
.bee-row-5 .bee-col-1 .bee-block-4 {
padding: 10px
}
.bee-row-6 .bee-col-1 .bee-block-1 {
padding: 30px 10px 10px;
text-align: center;
width: 100%
}
.bee-row-7 .bee-row-content {
color: #000000
}
.bee-row-7 .bee-col-1 .bee-block-1 {
color: #1e0e4b;
font-family: Inter, sans-serif;
font-size: 15px;
padding-bottom: 5px;
padding-top: 5px;
text-align: center
}
.bee-row-1 .bee-col-2 .bee-block-1 h1,
.bee-row-2 .bee-col-1 .bee-block-2,
.bee-row-3 .bee-col-1 .bee-block-4,
.bee-row-4 .bee-col-1 .bee-block-4,
.bee-row-5 .bee-col-1 .bee-block-4,
.bee-row-6 .bee-col-1 .bee-block-1 h3 {
direction: ltr;
font-family: "Courier New", Courier, "Lucida Sans Typewriter", "Lucida Typewriter", monospace;
line-height: 120%
}
.bee-row-1 .bee-col-2 .bee-block-1 h1 {
color: #737373;
font-size: 40px;
font-weight: 400;
letter-spacing: normal;
text-align: right
}
.bee-row-2 .bee-col-1 .bee-block-1 h2,
.bee-row-3 .bee-col-1 .bee-block-1 h2,
.bee-row-4 .bee-col-1 .bee-block-1 h2,
.bee-row-5 .bee-col-1 .bee-block-1 h2 {
color: #0e3d00;
direction: ltr;
font-family: "Courier New", Courier, "Lucida Sans Typewriter", "Lucida Typewriter", monospace;
font-size: 35px;
font-weight: 700;
letter-spacing: normal;
line-height: 120%;
text-align: left
}
.bee-row-2 .bee-col-1 .bee-block-2,
.bee-row-3 .bee-col-1 .bee-block-4,
.bee-row-4 .bee-col-1 .bee-block-4,
.bee-row-5 .bee-col-1 .bee-block-4 {
color: #101112;
font-size: 18px;
font-weight: 400;
letter-spacing: 0;
text-align: left
}
.bee-row-2 .bee-col-1 .bee-block-2 a,
.bee-row-3 .bee-col-1 .bee-block-4 a,
.bee-row-4 .bee-col-1 .bee-block-4 a,
.bee-row-5 .bee-col-1 .bee-block-4 a {
color: #7747FF
}
.bee-row-2 .bee-col-1 .bee-block-2 p:not(:last-child),
.bee-row-3 .bee-col-1 .bee-block-4 p:not(:last-child),
.bee-row-4 .bee-col-1 .bee-block-4 p:not(:last-child),
.bee-row-5 .bee-col-1 .bee-block-4 p:not(:last-child) {
margin-bottom: 16px
}
.bee-row-7 .bee-col-1 .bee-block-1 .bee-icon-image {
padding: 5px 6px 5px 5px
}
.bee-row-7 .bee-col-1 .bee-block-1 .bee-icon:not(.bee-icon-first) .bee-content {
margin-left: 0
}
.bee-row-7 .bee-col-1 .bee-block-1 .bee-icon::not(.bee-icon-last) .bee-content {
margin-right: 0
}
.bee-row-7 .bee-col-1 .bee-block-1 .bee-icon-label a {
color: #1e0e4b
}
.bee-row-6 .bee-col-1 .bee-block-1 h3 {
color: #a5a5a5;
font-size: 16px;
font-weight: 700;
letter-spacing: normal;
text-align: left
}
</style>
</head>
<body>
<div class="bee-page-container">
<div class="bee-row bee-row-1">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w3">
<div class="bee-block bee-block-1 bee-image"><img alt="" class="bee-autowidth" src="https://0c26875212.imgdist.com/pub/bfra/bpovlfhu/mx6/2cf/k54/logoipsum-345.svg" style="max-width:168px;" /></div>
</div>
<div class="bee-col bee-col-2 bee-col-w9">
<div class="bee-block bee-block-1 bee-heading">
<h1><span class="tinyMce-placeholder">Inference Benchmark Report</span> </h1>
</div>
</div>
</div>
</div>
<div class="bee-row bee-row-2">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w12">
<div class="bee-block bee-block-1 bee-heading">
<h2><span class="tinyMce-placeholder">Summary</span> </h2>
</div>
<div class="bee-block bee-block-2 bee-paragraph">
<p>This should be a sample summary of whole report in 5 lines</p>
</div>
</div>
</div>
</div>
<div class="bee-row bee-row-3">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w12">
<div class="bee-block bee-block-1 bee-heading">
<h2><span class="tinyMce-placeholder">Throughput Comparison</span> </h2>
</div>
<div class="bee-block bee-block-2 bee-image">
<div></div>
</div>
<div class="bee-block bee-block-3 bee-table">
<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">
<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">
<tr>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
</tr>
</thead>
<tbody style="font-size:16px;line-height:120%;">
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
</tbody>
</table>
</div>
<div class="bee-block bee-block-4 bee-paragraph">
<p>This should be a sample summary of throughput</p>
</div>
</div>
</div>
</div>
<div class="bee-row bee-row-4">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w12">
<div class="bee-block bee-block-1 bee-heading">
<h2><span class="tinyMce-placeholder">Latency Comparison</span> </h2>
</div>
<div class="bee-block bee-block-2 bee-image">
<div></div>
</div>
<div class="bee-block bee-block-3 bee-table">
<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">
<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">
<tr>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
</tr>
</thead>
<tbody style="font-size:16px;line-height:120%;">
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
</tbody>
</table>
</div>
<div class="bee-block bee-block-4 bee-paragraph">
<p>This should be a sample summary of latency</p>
</div>
</div>
</div>
</div>
<div class="bee-row bee-row-5">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w12">
<div class="bee-block bee-block-1 bee-heading">
<h2><span class="tinyMce-placeholder">Memory Usage Comparison</span> </h2>
</div>
<div class="bee-block bee-block-2 bee-image">
<div></div>
</div>
<div class="bee-block bee-block-3 bee-table">
<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">
<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">
<tr>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>
</tr>
</thead>
<tbody style="font-size:16px;line-height:120%;">
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
<tr>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>
</tr>
</tbody>
</table>
</div>
<div class="bee-block bee-block-4 bee-paragraph">
<p>This should be a sample summary of latency</p>
</div>
</div>
</div>
</div>
<div class="bee-row bee-row-6">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w6">
<div class="bee-block bee-block-1 bee-heading">
<h3><span class="tinyMce-placeholder">Report generated by EffiLLM</span> </h3>
</div>
</div>
<div class="bee-col bee-col-2 bee-col-w6"></div>
</div>
</div>
<div class="bee-row bee-row-7">
<div class="bee-row-content">
<div class="bee-col bee-col-1 bee-col-w12">
<div class="bee-block bee-block-1 bee-icons">
<div class="bee-icon bee-icon-last">
<div class="bee-content">
<div class="bee-icon-image"><a href="http://designedwithbeefree.com/" target="_blank" title="Designed with Beefree"><img alt="Beefree Logo" height="32px" src="https://d1oco4z2z1fhwp.cloudfront.net/assets/Beefree-logo.png" width="auto" /></a></div>
<div class="bee-icon-label bee-icon-label-right"><a href="http://designedwithbeefree.com/" target="_blank" title="Designed with Beefree">Designed with Beefree</a></div>
</div>
</div>
</div>
</div>
</div>
</div>
</div>
</body>
</html>''')
    
    def generate_html_report(self, 
                            title: str = "EffiLLM Benchmark Report",
                            filename: str = None) -> str:
        """
        Generate a complete HTML benchmark report.
        
        Args:
            title: Title of the report
            filename: Filename to save the report (default: auto-generated)
            
        Returns:
            Path to the saved HTML report file
        """
        # Generate data for the report sections
        summary = self.generate_summary_stats()
        throughput_data = self.generate_throughput_chart()
        latency_data = self.generate_latency_chart()
        memory_data = self.generate_memory_chart()
        
        # Generate HTML from template
        with open(os.path.join(os.path.dirname(__file__), "report_template.html"), "r") as f:
            template = f.read()
        
        # Replace placeholders with data
        html = template
        html = html.replace('<span class="tinyMce-placeholder">Inference Benchmark Report</span>', title)
        html = html.replace('<p>This should be a sample summary of whole report in 5 lines</p>', summary)
        
        # Throughput section
        throughput_charts_html = ""
        for chart_img in throughput_data["charts"]:
            throughput_charts_html += f'<div class="bee-block bee-block-2 bee-image"><img src="{chart_img}" style="width:100%;"/></div>'
        
        html = html.replace('<div class="bee-block bee-block-2 bee-image"><div></div></div>', throughput_charts_html, 1)
        
        # Generate throughput table
        throughput_table = self._generate_html_table(throughput_data["table"])
        html = html.replace('<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>', throughput_table)
        
        html = html.replace('<p>This should be a sample summary of throughput</p>', throughput_data["summary"])
        
        # Latency section
        latency_charts_html = ""
        for chart_img in latency_data["charts"]:
            latency_charts_html += f'<div class="bee-block bee-block-2 bee-image"><img src="{chart_img}" style="width:100%;"/></div>'
        
        # Find the next occurrence of the placeholder
        placeholder = '<div class="bee-block bee-block-2 bee-image"><div></div></div>'
        start_idx = html.find(placeholder, html.find("Latency Comparison"))
        if start_idx != -1:
            end_idx = start_idx + len(placeholder)
            html = html[:start_idx] + latency_charts_html + html[end_idx:]
        
        # Generate latency table
        latency_table = self._generate_html_table(latency_data["table"])
        placeholder = '<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>'
        start_idx = html.find(placeholder, html.find("Latency Comparison"))
        if start_idx != -1:
            end_idx = start_idx + len(placeholder)
            html = html[:start_idx] + latency_table + html[end_idx:]
        
        html = html.replace('<p>This should be a sample summary of latency</p>', latency_data["summary"])
        
        # Memory section
        memory_charts_html = ""
        for chart_img in memory_data["charts"]:
            memory_charts_html += f'<div class="bee-block bee-block-2 bee-image"><img src="{chart_img}" style="width:100%;"/></div>'
        
        # Find the next occurrence of the placeholder
        placeholder = '<div class="bee-block bee-block-2 bee-image"><div></div></div>'
        start_idx = html.find(placeholder, html.find("Memory Usage Comparison"))
        if start_idx != -1:
            end_idx = start_idx + len(placeholder)
            html = html[:start_idx] + memory_charts_html + html[end_idx:]
        
        # Generate memory table
        memory_table = self._generate_html_table(memory_data["table"])
        placeholder = '<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>'
        start_idx = html.find(placeholder, html.find("Memory Usage Comparison"))
        if start_idx != -1:
            end_idx = start_idx + len(placeholder)
            html = html[:start_idx] + memory_table + html[end_idx:]
        
        html = html.replace('<p>This should be a sample summary of latency</p>', memory_data["summary"], 1)

        # Set the logo
        html = html.replace('<img alt="" class="bee-autowidth" src="https://0c26875212.imgdist.com/pub/bfra/bpovlfhu/mx6/2cf/k54/logoipsum-345.svg" style="max-width:168px;" />', 
                            '<h3 style="color:#737373;">EffiLLM</h3>')
        
        # Update footer
        html = html.replace('<span class="tinyMce-placeholder">Report generated by EffiLLM</span>',
                           f'Report generated by EffiLLM on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        # Remove Beefree logo
        html = html.replace('<div class="bee-block bee-block-1 bee-icons">\n<div class="bee-icon bee-icon-last">\n<div class="bee-content">\n<div class="bee-icon-image"><a href="http://designedwithbeefree.com/" target="_blank" title="Designed with Beefree"><img alt="Beefree Logo" height="32px" src="https://d1oco4z2z1fhwp.cloudfront.net/assets/Beefree-logo.png" width="auto" /></a></div>\n<div class="bee-icon-label bee-icon-label-right"><a href="http://designedwithbeefree.com/" target="_blank" title="Designed with Beefree">Designed with Beefree</a></div>\n</div>\n</div>\n</div>', '')
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"effillm_report_{timestamp}.html"
        
        # Save the HTML file
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(html)
        
        return filepath
    
    def _generate_html_table(self, table_data: Dict) -> str:
        """Generate HTML table from data dictionary."""
        headers = table_data["headers"]
        rows = table_data["rows"]
        
        # Generate header row
        header_html = '<tr>\n'
        for header in headers:
            header_html += f'<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">{header}</th>\n'
        header_html += '</tr>\n'
        
        # Generate data rows
        rows_html = ''
        for row in rows:
            rows_html += '<tr>\n'
            for cell in row:
                rows_html += f'<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">{cell}</td>\n'
            rows_html += '</tr>\n'
        
        # Complete table
        table_html = f'''<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">
<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">
{header_html}</thead>
<tbody style="font-size:16px;line-height:120%;">
{rows_html}</tbody>
</table>'''
        
        return table_html