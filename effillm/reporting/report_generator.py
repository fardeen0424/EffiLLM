# EffiLLM/effillm/reporting/report_generator.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import List, Dict, Any, Optional, Union
import base64
from io import BytesIO
import datetime
from pathlib import Path
import pandas as pd

# Set Matplotlib styling
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8-pastel')
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
        for result in self.results:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            self.model_names.append(model_name)
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64-encoded string for HTML embedding."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
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
        
        # Summarize models
        if self.model_names:
            summary_lines.append(f"• Models compared: {', '.join(self.model_names)}")
            
        # Summarize configurations
        configs = set()
        for result in self.results:
            if "inference" in result:
                configs.update(result["inference"].keys())
        
        if configs:
            config_count = len(configs)
            summary_lines.append(f"• {config_count} different configurations tested")
        
        # Average throughput improvement or comparison
        if len(self.results) > 1:
            # Compare the first and last model (assuming they might be base vs quantized)
            base_throughput = 0
            quant_throughput = 0
            
            for config in configs:
                if "inference" in self.results[0] and config in self.results[0]["inference"]:
                    base_throughput += self.results[0]["inference"][config]["throughput"]["tokens_per_second"]
                
                if "inference" in self.results[-1] and config in self.results[-1]["inference"]:
                    quant_throughput += self.results[-1]["inference"][config]["throughput"]["tokens_per_second"]
            
            if base_throughput > 0 and quant_throughput > 0:
                improvement = (quant_throughput / base_throughput) - 1
                if improvement > 0:
                    summary_lines.append(f"• Average throughput improvement: {improvement:.1%}")
                else:
                    summary_lines.append(f"• Average throughput change: {improvement:.1%}")
        
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
        
        # Create bar chart for throughput
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        x = np.arange(len(configs))
        bar_width = 0.8 / len(throughputs) if throughputs else 0.8
        
        # Plot each model's throughput
        for i, (model, model_data) in enumerate(throughputs.items()):
            model_throughputs = [model_data.get(config, 0) for config in configs]
            offset = (i - len(throughputs)/2 + 0.5) * bar_width
            ax.bar(x + offset, model_throughputs, bar_width, label=model)
        
        # Customize plot
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Tokens per second')
        ax.set_title('Throughput Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        
        # Convert to base64 for HTML embedding
        chart_img = self._fig_to_base64(fig)
        
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
            "chart": chart_img,
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
        
        # Create bar chart for latency
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        x = np.arange(len(configs))
        bar_width = 0.8 / len(latencies) if latencies else 0.8
        
        # Plot each model's latency
        for i, (model, model_data) in enumerate(latencies.items()):
            model_latencies = [model_data.get(config, 0) for config in configs]
            offset = (i - len(latencies)/2 + 0.5) * bar_width
            ax.bar(x + offset, model_latencies, bar_width, label=model)
        
        # Customize plot
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Time to First Token Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        
        # Convert to base64 for HTML embedding
        chart_img = self._fig_to_base64(fig)
        
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
            "chart": chart_img,
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
                        summary_lines.append(f"• Increasing batch size from {min_batch} to {max_batch} for {key}: {direction} latency by {abs(latency_change):.1%}")
        
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
                memory_usage[model_name]["loading"] = max(ram, vram)
            
            # Get inference memory
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "memory" in data and "impact" in data["memory"]:
                        mem_impact = data["memory"]["impact"]
                        ram = mem_impact.get("ram_used_gb", 0)
                        vram = mem_impact.get("vram_used_gb", 0)
                        # Use VRAM if available, otherwise RAM
                        memory_usage[model_name]["inference"][config] = vram if vram > 0 else ram
        
        configs = sorted(list(configs))
        
        # Create stacked bar chart for memory
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        x = np.arange(len(self.model_names))
        
        # Create bars for loading memory
        loading_memory = [memory_usage[model]["loading"] for model in self.model_names]
        loading_bars = ax.bar(x, loading_memory, label='Model Loading')
        
        # Calculate average inference memory
        inference_memory = []
        for model in self.model_names:
            inf_mem = memory_usage[model]["inference"]
            if inf_mem:
                avg_mem = sum(inf_mem.values()) / len(inf_mem)
            else:
                avg_mem = 0
            inference_memory.append(avg_mem)
        
        # Create bars for inference memory (stacked)
        ax.bar(x, inference_memory, bottom=loading_memory, label='Inference (avg)')
        
        # Customize plot
        ax.set_xlabel('Model')
        ax.set_ylabel('Memory Usage (GB)')
        ax.set_title('Memory Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names)
        ax.legend()
        
        # Add value annotations
        for i, v in enumerate(loading_memory):
            ax.text(i, v/2, f"{v:.1f}", ha='center')
        
        for i, v in enumerate(inference_memory):
            if v > 0:
                ax.text(i, loading_memory[i] + v/2, f"{v:.1f}", ha='center')
        
        plt.tight_layout()
        
        # Convert to base64 for HTML embedding
        chart_img = self._fig_to_base64(fig)
        
        # Create detailed memory table by configuration
        table_data = {
            "headers": ["Model", "Loading"] + [f"Inference ({config})" for config in configs],
            "rows": []
        }
        
        for model in self.model_names:
            row = [model, f"{memory_usage[model]['loading']:.2f}"]
            for config in configs:
                value = memory_usage[model]["inference"].get(config, "N/A")
                row.append(f"{value:.2f}" if isinstance(value, (int, float)) else value)
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = self._generate_memory_summary(memory_usage)
        
        return {
            "chart": chart_img,
            "table": table_data,
            "summary": summary
        }
    
    def _generate_memory_summary(self, memory_usage) -> str:
        """Generate a summary of memory comparisons."""
        if not memory_usage:
            return "<p>No memory data available for comparison.</p>"
        
        summary_lines = []
        
        # Compare loading memory
        loading_memory = [(model, data["loading"]) for model, data in memory_usage.items()]
        loading_memory.sort(key=lambda x: x[1])
        
        if loading_memory:
            min_model, min_memory = loading_memory[0]
            max_model, max_memory = loading_memory[-1]
            
            if min_memory > 0:
                summary_lines.append(f"• {min_model} uses the least memory for loading ({min_memory:.2f} GB)")
                
                if len(loading_memory) > 1 and max_memory > 0:
                    memory_ratio = max_memory / min_memory
                    summary_lines.append(f"• {max_model} requires {memory_ratio:.1f}x more memory than {min_model} for loading")
        
        # Compare inference memory (averaged across configs)
        avg_inference_memory = []
        for model, data in memory_usage.items():
            inf_data = data["inference"]
            if inf_data:
                avg_mem = sum(inf_data.values()) / len(inf_data)
                avg_inference_memory.append((model, avg_mem))
        
        avg_inference_memory.sort(key=lambda x: x[1])
        
        if avg_inference_memory:
            min_model, min_memory = avg_inference_memory[0]
            max_model, max_memory = avg_inference_memory[-1]
            
            if min_memory > 0:
                summary_lines.append(f"• {min_model} uses the least memory during inference ({min_memory:.2f} GB on average)")
                
                if len(avg_inference_memory) > 1 and max_memory > 0:
                    memory_ratio = max_memory / min_memory
                    summary_lines.append(f"• {max_model} requires {memory_ratio:.1f}x more memory than {min_model} during inference")
        
        return "<p>" + "</p><p>".join(summary_lines) + "</p>"
    
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
        html = html.replace('<div class="bee-block bee-block-2 bee-image"><div></div></div>',
                          f'<div class="bee-block bee-block-2 bee-image"><img src="{throughput_data["chart"]}" style="width:100%;"/></div>')
        
        # Generate throughput table
        throughput_table = self._generate_html_table(throughput_data["table"])
        html = html.replace('<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>', throughput_table)
        
        html = html.replace('<p>This should be a sample summary of throughput</p>', throughput_data["summary"])
        
        # Latency section
        html = html.replace('<div class="bee-block bee-block-2 bee-image"><div></div></div>',
                          f'<div class="bee-block bee-block-2 bee-image"><img src="{latency_data["chart"]}" style="width:100%;"/></div>', 1)  # Replace only first occurrence
        
        # Generate latency table
        latency_table = self._generate_html_table(latency_data["table"])
        html = html.replace('<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>', latency_table, 1)  # Replace only first occurrence
        
        html = html.replace('<p>This should be a sample summary of latency</p>', latency_data["summary"])
        
        # Memory section
        html = html.replace('<div class="bee-block bee-block-2 bee-image"><div></div></div>',
                          f'<div class="bee-block bee-block-2 bee-image"><img src="{memory_data["chart"]}" style="width:100%;"/></div>', 1)  # Replace only first occurrence
        
        # Generate memory table
        memory_table = self._generate_html_table(memory_data["table"])
        html = html.replace('<table style="table-layout:fixed;direction:ltr;background-color:transparent;font-family:Open Sans, Helvetica Neue, Helvetica, Arial, sans-serif;font-weight:400;color:#101112;text-align:left;letter-spacing:0px;">\n<thead style="background-color:#f2f2f2;color:#101112;font-size:14px;line-height:120%;text-align:center;">\n<tr>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add header text</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">​</th>\n</tr>\n</thead>\n<tbody style="font-size:16px;line-height:120%;">\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">Add text</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n<tr>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n<td style="border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">&amp;ZeroWidthSpace;</td>\n</tr>\n</tbody>\n</table>', memory_table, 1)  # Replace only first occurrence
        
        html = html.replace('<p>This should be a sample summary of latency</p>', memory_data["summary"], 1)  # Replace only first occurrence
        
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


# Save the template HTML file
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