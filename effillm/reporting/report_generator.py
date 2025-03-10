# test_report_generator.py
import os
import sys
import logging
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

def generate_throughput_charts(results_data):
    """Generate throughput visualization charts"""
    try:
        charts = []
        
        # Extract throughput data
        models = []
        configs = set()
        throughputs = {}
        
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            if model_name not in models:
                models.append(model_name)
            
            throughputs[model_name] = {}
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "throughput" in data:
                        throughputs[model_name][config] = data["throughput"]["tokens_per_second"]
        
        configs = sorted(list(configs))
        
        # Chart 1: Bar chart comparison
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(configs))
        width = 0.8 / len(models) if models else 0.8
        
        for i, model in enumerate(models):
            if model in throughputs:
                values = [throughputs[model].get(config, 0) for config in configs]
                offset = (i - len(models)/2 + 0.5) * width
                ax1.bar(x + offset, values, width, label=model)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Tokens per second')
        ax1.set_title('Throughput Comparison by Configuration')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        plt.tight_layout()
        charts.append(fig_to_base64(fig1))
        
        # Chart 2: Horizontal bar chart - average throughput
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        avg_throughputs = {}
        for model, model_data in throughputs.items():
            if model_data:
                avg_throughputs[model] = sum(model_data.values()) / len(model_data)
        
        if avg_throughputs:
            models_sorted = sorted(avg_throughputs.items(), key=lambda x: x[1])
            model_labels = [x[0] for x in models_sorted]
            avgs = [x[1] for x in models_sorted]
            
            cmap = plt.cm.viridis
            colors = cmap(np.linspace(0.2, 0.8, len(model_labels)))
            ax2.barh(model_labels, avgs, color=colors)
            ax2.set_xlabel('Tokens per second (avg)')
            ax2.set_title('Average Throughput by Model')
            
            for i, v in enumerate(avgs):
                ax2.text(v + 0.5, i, f"{v:.1f}", va='center')
            
            plt.tight_layout()
            charts.append(fig_to_base64(fig2))
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.text(0.5, 0.5, 'Insufficient data for average throughput', 
                     horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
            charts.append(fig_to_base64(fig2))
        
        # Chart 3: Scatter plot - throughput vs. batch size
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        batch_sizes = {}
        
        for config in configs:
            if "bs" in config and "seq" in config:
                try:
                    parts = config.split("_")  # Changed from "" to "_"
                    if len(parts) >= 2:
                        batch = int(parts[0].replace("bs", ""))
                        seq = int(parts[1].replace("seq", ""))
                        
                        for model in models:
                            if model in throughputs and config in throughputs[model]:
                                if model not in batch_sizes:
                                    batch_sizes[model] = {}
                                if seq not in batch_sizes[model]:
                                    batch_sizes[model][seq] = {}
                                batch_sizes[model][seq][batch] = throughputs[model][config]
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse config {config}: {e}")
        
        # Plot scaling for a specific sequence length
        for model in batch_sizes:
            for seq_len in sorted(batch_sizes[model].keys()):
                seq_data = batch_sizes[model][seq_len]
                batches = sorted(seq_data.keys())
                throughputs_by_batch = [seq_data[b] for b in batches]
                ax3.plot(batches, throughputs_by_batch, 'o-', 
                         label=f"{model} (seq={seq_len})", linewidth=2, markersize=8)
        
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Tokens per second')
        ax3.set_title('Throughput Scaling with Batch Size')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
        plt.tight_layout()
        charts.append(fig_to_base64(fig3))
        
        # Generate table data
        table_data = {
            "headers": ["Model"] + configs,
            "rows": []
        }
        
        for model in models:
            row = [model]
            for config in configs:
                if model in throughputs and config in throughputs[model]:
                    row.append(f"{throughputs[model][config]:.2f}")
                else:
                    row.append("N/A")
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = []
        
        if models and configs:
            # Find best performer
            best_model = None
            best_config = None
            best_throughput = 0
            
            for model in models:
                if model in throughputs:
                    for config, value in throughputs[model].items():
                        if value > best_throughput:
                            best_throughput = value
                            best_model = model
                            best_config = config
            
            if best_model and best_config:
                summary.append(f"• Highest throughput: {best_throughput:.2f} tokens/sec with {best_model} on {best_config}")
            
            # Compare models if multiple
            if len(avg_throughputs) > 1:
                sorted_models = sorted(avg_throughputs.items(), key=lambda x: x[1], reverse=True)
                best, best_avg = sorted_models[0]
                second, second_avg = sorted_models[1]
                
                if second_avg > 0:
                    ratio = best_avg / second_avg
                    summary.append(f"• {best} is {ratio:.2f}x faster than {second} on average")
        
        return {
            "charts": charts,
            "table": table_data,
            "summary": "<p>" + "</p><p>".join(summary) + "</p>" if summary else "<p>No throughput data available for comparison.</p>"
        }
    except Exception as e:
        logger.error(f"Error generating throughput charts: {e}")
        return {
            "charts": [],
            "table": {"headers": ["Error"], "rows": [["Failed to generate throughput data"]]},
            "summary": f"<p>Error generating throughput charts: {e}</p>"
        }

def generate_latency_charts(results_data):
    """Generate latency visualization charts"""
    try:
        charts = []
        
        # Extract latency data
        models = []
        configs = set()
        latencies = {}
        
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            if model_name not in models:
                models.append(model_name)
                
            latencies[model_name] = {}
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "time_to_first_token" in data:
                        # Convert to milliseconds
                        latencies[model_name][config] = data["time_to_first_token"]["mean"] * 1000
        
        configs = sorted(list(configs))
        
        # Chart 1: Bar chart for latency
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(configs))
        width = 0.8 / len(models) if models else 0.8
        
        for i, model in enumerate(models):
            if model in latencies:
                values = [latencies[model].get(config, 0) for config in configs]
                offset = (i - len(models)/2 + 0.5) * width  # Fixed syntax
                ax1.bar(x + offset, values, width, label=model)
        
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Time to First Token Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        ax1.legend()
        plt.tight_layout()
        charts.append(fig_to_base64(fig1))
        
        # Chart 2: Heatmap of latency by batch size and sequence length
        # Combine data from all models
        batch_seq_data = {}
        
        for model in models:
            if model in latencies:
                for config, value in latencies[model].items():
                    if "bs" in config and "seq" in config:
                        try:
                            parts = config.split("_")  # Changed from "" to "_"
                            if len(parts) >= 2:
                                batch = int(parts[0].replace("bs", ""))
                                seq = int(parts[1].replace("seq", ""))
                                key = (model, batch, seq)
                                batch_seq_data[key] = value
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse config {config}: {e}")
        
        if batch_seq_data:
            # Get unique batch sizes and sequence lengths
            all_models = set()
            all_batches = set()
            all_seqs = set()
            
            for (model, batch, seq) in batch_seq_data.keys():
                all_models.add(model)
                all_batches.add(batch)
                all_seqs.add(seq)
            
            all_models = sorted(list(all_models))
            all_batches = sorted(list(all_batches))
            all_seqs = sorted(list(all_seqs))
            
            # Create a figure with subplots for each model
            fig2 = plt.figure(figsize=(5*len(all_models), 4))  # Fixed syntax
            
            # Fix: Handle single model case differently
            if len(all_models) == 1:
                ax = fig2.add_subplot(111)
                axes = [ax]  # Use list of single axis
            else:
                axes = []
                for i in range(len(all_models)):
                    ax = fig2.add_subplot(1, len(all_models), i+1)
                    axes.append(ax)
            
            for i, model in enumerate(all_models):
                # Create matrix for heatmap
                matrix = np.zeros((len(all_batches), len(all_seqs)))
                
                for b_idx, b in enumerate(all_batches):
                    for s_idx, s in enumerate(all_seqs):
                        key = (model, b, s)
                        if key in batch_seq_data:
                            matrix[b_idx, s_idx] = batch_seq_data[key]
                
                # Plot heatmap
                im = axes[i].imshow(matrix, cmap='YlOrRd')
                
                # Set ticks and labels
                axes[i].set_xticks(np.arange(len(all_seqs)))
                axes[i].set_yticks(np.arange(len(all_batches)))
                axes[i].set_xticklabels(all_seqs)
                axes[i].set_yticklabels(all_batches)
                axes[i].set_title(f"{model}")
                
                # Add text annotations
                for b_idx, b in enumerate(all_batches):
                    for s_idx, s in enumerate(all_seqs):
                        key = (model, b, s)
                        if key in batch_seq_data:
                            text = axes[i].text(s_idx, b_idx, f"{batch_seq_data[key]:.1f}", 
                                               ha="center", va="center", color="black", fontsize=8)
            
            # Add colorbar - fix for both single and multiple axes cases
            fig2.colorbar(im, ax=axes, shrink=0.7, label='Latency (ms)')
            
            plt.suptitle('Latency Heatmap by Batch Size and Sequence Length', fontsize=16)
            
            # Add common labels as text - safer than fig.text
            for ax in axes:
                ax.set_xlabel('Sequence Length')
                ax.set_ylabel('Batch Size')
            
            # Use subplots_adjust instead of tight_layout
            plt.subplots_adjust(wspace=0.3, hspace=0.3, bottom=0.15, top=0.85)
            charts.append(fig_to_base64(fig2))
        else:
            # Create empty chart if no data
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.text(0.5, 0.5, 'Insufficient data for latency heatmap', 
                    horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
            charts.append(fig_to_base64(fig2))
        
        # Chart 3: Box plot for latency distribution
        latency_data = []
        
        for model in models:
            if model in latencies:
                for config, value in latencies[model].items():
                    latency_data.append({
                        "Model": model,
                        "Configuration": config,
                        "Latency (ms)": value
                    })
        
        if latency_data:
            df = pd.DataFrame(latency_data)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x="Model", y="Latency (ms)", data=df, ax=ax3)
            ax3.set_title('Latency Distribution by Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            charts.append(fig_to_base64(fig3))
        else:
            # Create empty chart
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.text(0.5, 0.5, 'Insufficient data for latency distribution', 
                    horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
            charts.append(fig_to_base64(fig3))
        
        # Generate table data
        table_data = {
            "headers": ["Model"] + configs,
            "rows": []
        }
        
        for model in models:
            row = [model]
            for config in configs:
                if model in latencies and config in latencies[model]:
                    row.append(f"{latencies[model][config]:.2f}")
                else:
                    row.append("N/A")
            table_data["rows"].append(row)
        
        # Generate summary text
        summary = []
        
        if models and configs:
            # Find lowest latency
            best_model = None
            best_config = None
            best_latency = float('inf')
            
            for model in models:
                if model in latencies:
                    for config, value in latencies[model].items():
                        if value < best_latency:
                            best_latency = value
                            best_model = model
                            best_config = config
            
            if best_model and best_config:
                summary.append(f"• Lowest latency: {best_latency:.2f} ms with {best_model} on {best_config}")
            
            # Calculate average latencies
            avg_latencies = {}
            for model in models:
                if model in latencies and latencies[model]:
                    avg_latencies[model] = sum(latencies[model].values()) / len(latencies[model])
            
            # Compare models if multiple
            if len(avg_latencies) > 1:
                sorted_models = sorted(avg_latencies.items(), key=lambda x: x[1])
                best, best_avg = sorted_models[0]
                second, second_avg = sorted_models[1]
                
                if best_avg > 0:
                    ratio = second_avg / best_avg
                    summary.append(f"• {best} has {ratio:.2f}x lower latency than {second} on average")
        
        return {
            "charts": charts,
            "table": table_data,
            "summary": "<p>" + "</p><p>".join(summary) + "</p>" if summary else "<p>No latency data available for comparison.</p>"
        }
    except Exception as e:
        logger.error(f"Error generating latency charts: {e}")
        return {
            "charts": [],
            "table": {"headers": ["Error"], "rows": [["Failed to generate latency data"]]},
            "summary": f"<p>Error generating latency charts: {e}</p>"
        }

def generate_memory_charts(results_data):
    """Generate memory usage visualization charts"""
    try:
        charts = []
        
        # Extract memory data
        models = []
        configs = set()
        memory_data = {}
        
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            if model_name not in models:
                models.append(model_name)
                
            memory_data[model_name] = {
                "loading": {},
                "inference": {}
            }
            
            # Get loading memory
            if "loading" in result and "memory_impact" in result["loading"]:
                mem_impact = result["loading"]["memory_impact"]
                ram = mem_impact.get("ram_used_gb", 0)
                vram = mem_impact.get("vram_used_gb", 0) if "vram_used_gb" in mem_impact else 0
                memory_data[model_name]["loading"] = max(ram, vram)
            
            # Get inference memory
            if "inference" in result:
                for config, data in result["inference"].items():
                    configs.add(config)
                    if "memory" in data and "impact" in data["memory"]:
                        mem_impact = data["memory"]["impact"]
                        ram = mem_impact.get("ram_used_gb", 0)
                        vram = mem_impact.get("vram_used_gb", 0) if "vram_used_gb" in mem_impact else 0
                        memory_data[model_name]["inference"][config] = max(ram, vram)
        
        configs = sorted(list(configs))
        
        # Chart 1: Stacked bar chart for memory (loading + inference)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35
        
        # Get loading and average inference memory
        loading_mem = []
        inference_mem = []
        
        for model in models:
            # Loading memory
            if model in memory_data and memory_data[model]["loading"]:
                loading_mem.append(memory_data[model]["loading"])
            else:
                loading_mem.append(0)
            
            # Average inference memory
            avg_inf_mem = 0
            count = 0
            
            if model in memory_data and memory_data[model]["inference"]:
                for config, value in memory_data[model]["inference"].items():
                    avg_inf_mem += value
                    count += 1
            
            if count > 0:
                avg_inf_mem /= count
            
            inference_mem.append(avg_inf_mem)
        
        # Create stacked bars
        loading_bars = ax1.bar(x, loading_mem, width, label='Model Loading')
        inference_bars = ax1.bar(x, inference_mem, width, bottom=loading_mem, label='Inference (avg)')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('Memory Usage by Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        
        # Add value annotations
        for i, v in enumerate(loading_mem):
            if v > 0:
                ax1.text(i, v/2, f"{v:.1f}", ha='center', va='center')
        
        for i, v in enumerate(inference_mem):
            if v > 0:
                ax1.text(i, loading_mem[i] + v/2, f"{v:.1f}", ha='center', va='center')
        
        plt.tight_layout()
        charts.append(fig_to_base64(fig1))
        
        # Chart 2: Scatter plot - Memory vs. Throughput
        # Extract throughput data
        throughputs = {}
        
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            avg_throughput = 0
            count = 0
            
            if "inference" in result:
                for config, data in result["inference"].items():
                    if "throughput" in data and "tokens_per_second" in data["throughput"]:
                        avg_throughput += data["throughput"]["tokens_per_second"]
                        count += 1
            
            if count > 0:
                avg_throughput /= count
                throughputs[model_name] = avg_throughput
        
        # Create scatter data
        scatter_data = []
        
        for model in models:
            if model in memory_data and model in throughputs:
                total_mem = memory_data[model]["loading"]
                scatter_data.append({
                    "model": model,
                    "memory": total_mem,
                    "throughput": throughputs[model]
                })
        
        if scatter_data:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            x = [d["memory"] for d in scatter_data]
            y = [d["throughput"] for d in scatter_data]
            labels = [d["model"] for d in scatter_data]
            
            scatter = ax2.scatter(x, y, c=range(len(x)), cmap='viridis', s=100, alpha=0.7)
            
            # Add model labels
            for i, label in enumerate(labels):
                ax2.annotate(label, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
            
            # Add trend line if we have enough points
            if len(x) > 2:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax2.plot(x, p(x), "r--", alpha=0.5)
                except Exception as e:
                    logger.warning(f"Could not generate trend line: {e}")
            
            ax2.set_xlabel('Memory Usage (GB)')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Memory Efficiency: Throughput vs. Memory Usage')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            charts.append(fig_to_base64(fig2))
        else:
            # Create empty chart
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.text(0.5, 0.5, 'Insufficient data for memory efficiency plot', 
                    horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
            charts.append(fig_to_base64(fig2))
        
        # Chart 3: Pie chart showing memory distribution for a model
        if models and models[0] in memory_data:
            model = models[0]  # Use the first model for the pie chart
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            
            # Get memory components
            loading_mem = memory_data[model]["loading"]
            
            # Calculate average inference memory
            inf_mem = 0
            if memory_data[model]["inference"]:
                inf_mem = sum(memory_data[model]["inference"].values()) / len(memory_data[model]["inference"])
            
            # Create pie chart
            sizes = [loading_mem, inf_mem]
            labels = ['Model Loading', 'Inference (avg)']
            explode = (0.1, 0)  # Explode the first slice
            ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', 
                    shadow=True, startangle=90)
            ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax3.set_title(f'Memory Distribution for {model}')
            
            plt.tight_layout()
            charts.append(fig_to_base64(fig3))
        else:
            # Create empty chart
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.text(0.5, 0.5, 'Insufficient data for memory distribution chart', 
                    horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
            charts.append(fig_to_base64(fig3))
        
        # Generate table data
        table_data = {
            "headers": ["Model", "Loading Memory (GB)"] + [f"Inference: {config} (GB)" for config in configs],
            "rows": []
        }
        
        for model in models:
            row = [model]
            
            # Loading memory
            if model in memory_data and memory_data[model]["loading"]:
                row.append(f"{memory_data[model]['loading']:.2f}")
            else:
                row.append("N/A")
            
            # Inference memory by config
            for config in configs:
                if model in memory_data and "inference" in memory_data[model] and config in memory_data[model]["inference"]:
                    row.append(f"{memory_data[model]['inference'][config]:.2f}")
                else:
                    row.append("N/A")
            
            table_data["rows"].append(row)
        
        # Generate summary
        summary = []
        
        if models:
            # Compare loading memory
            loading_mems = [(model, memory_data[model]["loading"]) for model in models 
                          if model in memory_data and memory_data[model]["loading"]]
            
            if loading_mems:
                loading_mems.sort(key=lambda x: x[1])
                min_model, min_mem = loading_mems[0]
                summary.append(f"• {min_model} uses the least memory for loading ({min_mem:.2f} GB)")
                
                if len(loading_mems) > 1:
                    max_model, max_mem = loading_mems[-1]
                    ratio = max_mem / min_mem if min_mem > 0 else 0
                    if ratio > 1:
                        summary.append(f"• {max_model} requires {ratio:.1f}x more memory than {min_model} for loading")
            
            # Compare inference memory
            avg_inference = []
            
            for model in models:
                if model in memory_data and "inference" in memory_data[model] and memory_data[model]["inference"]:
                    avg_mem = sum(memory_data[model]["inference"].values()) / len(memory_data[model]["inference"])
                    avg_inference.append((model, avg_mem))
            
            if avg_inference:
                avg_inference.sort(key=lambda x: x[1])
                min_model, min_mem = avg_inference[0]
                summary.append(f"• {min_model} uses the least memory during inference ({min_mem:.2f} GB on average)")
                
                if len(avg_inference) > 1:
                    max_model, max_mem = avg_inference[-1]
                    ratio = max_mem / min_mem if min_mem > 0 else 0
                    if ratio > 1:
                        summary.append(f"• {max_model} requires {ratio:.1f}x more memory than {min_model} during inference")
        
        return {
            "charts": charts,
            "table": table_data,
            "summary": "<p>" + "</p><p>".join(summary) + "</p>" if summary else "<p>No memory data available for comparison.</p>"
        }
    except Exception as e:
        logger.error(f"Error generating memory charts: {e}")
        return {
            "charts": [],
            "table": {"headers": ["Error"], "rows": [["Failed to generate memory data"]]},
            "summary": f"<p>Error generating memory charts: {e}</p>"
        }

def generate_system_info_section(results_data):
    """Generate detailed system information section"""
    try:
        if not results_data or "system_info" not in results_data[0]:
            return "<p>System information not available</p>"
        
        system = results_data[0]["system_info"]
        
        # GPU Details
        gpu_info = []
        if "gpu_devices" in system and system["gpu_devices"]:
            for i, gpu in enumerate(system["gpu_devices"]):
                vram = system.get("gpu_vram_gb", {}).get(str(i), "Unknown")
                gpu_info.append(f"<li>GPU {i}: {gpu} (VRAM: {vram} GB)</li>")
        
        # CPU Details
        cpu_details = f"{system.get('cpu_count', 'Unknown')} cores, {system.get('processor', 'Unknown')}"
        cpu_info = f"<li>CPU: {cpu_details}</li>"
        
        # RAM Details
        ram = system.get("total_ram_gb", "Unknown")
        ram_info = f"<li>System RAM: {ram} GB</li>"
        
        # Software Details
        software_info = []
        if "python_version" in system:
            software_info.append(f"<li>Python: {system['python_version']}</li>")
        if "cuda_version" in system:
            software_info.append(f"<li>CUDA: {system['cuda_version']}</li>")
        if "torch_version" in system:
            software_info.append(f"<li>PyTorch: {system['torch_version']}</li>")
        
        # Combine all sections
        html = f"""
        <div class="system-info">
            <h4>Hardware Configuration</h4>
            <ul>
                {cpu_info}
                {ram_info}
                {"".join(gpu_info)}
            </ul>
            
            <h4>Software Environment</h4>
            <ul>
                {"".join(software_info)}
            </ul>
        </div>
        """
        
        return html
    except Exception as e:
        logger.error(f"Error generating system info: {e}")
        return f"<p>Error generating system information: {e}</p>"

def generate_summary_stats(results_data):
    """Generate summary statistics from benchmark results"""
    try:
        summary = []
        
        # Get hardware info from the first result
        if results_data and "system_info" in results_data[0]:
            system = results_data[0]["system_info"]
            gpu_info = system.get("gpu_devices", ["Unknown GPU"])[0] if "gpu_devices" in system and system["gpu_devices"] else "CPU only"
            summary.append(f"• Benchmark run on {gpu_info}")
            
            # Add more system details
            cpu_info = f"{system.get('cpu_count', 'Unknown')} cores, {system.get('processor', 'Unknown')}"
            summary.append(f"• CPU: {cpu_info}")
            
            if "cuda_version" in system:
                summary.append(f"• CUDA: {system.get('cuda_version', 'Unknown')}")
        
        # Summarize models
        if results_data:
            model_names = []
            for result in results_data:
                model_id = result.get("model_id", "unknown")
                model_name = model_id.split('/')[-1] if '/' in model_id else model_id
                if model_name not in model_names:
                    model_names.append(model_name)
            
            summary.append(f"• Models tested: {', '.join(model_names)}")
        
        # Summarize configurations
        configs = set()
        for result in results_data:
            if "inference" in result:
                configs.update(result["inference"].keys())
        
        if configs:
            config_count = len(configs)
            batch_sizes = set()
            seq_lengths = set()
            
            for config in configs:
                if "bs" in config and "seq" in config:
                    try:
                        parts = config.split("_")  # Changed from "" to "_"
                        if len(parts) >= 2:
                            bs_part = parts[0].replace("bs", "")
                            seq_part = parts[1].replace("seq", "")
                            
                            if bs_part.isdigit():
                                batch_sizes.add(int(bs_part))
                            if seq_part.isdigit():
                                seq_lengths.add(int(seq_part))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse config {config}: {e}")
            
            if batch_sizes:
                summary.append(f"• Batch sizes: {', '.join(map(str, sorted(batch_sizes)))}")
            
            if seq_lengths:
                summary.append(f"• Sequence lengths: {', '.join(map(str, sorted(seq_lengths)))}")
        
        # Add benchmark date information if available
        if results_data and "timestamp" in results_data[0]:
            try:
                import datetime
                timestamp = results_data[0]["timestamp"]
                benchmark_time = datetime.datetime.fromtimestamp(timestamp)
                summary.append(f"• Benchmark run on: {benchmark_time.strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                logger.warning(f"Could not parse timestamp: {e}")
        
        # Add benchmark duration if available
        if results_data and "benchmark_duration_seconds" in results_data[0]:
            duration = results_data[0]["benchmark_duration_seconds"]
            if duration > 3600:
                summary.append(f"• Total benchmark duration: {duration/3600:.1f} hours")
            elif duration > 60:
                summary.append(f"• Total benchmark duration: {duration/60:.1f} minutes")
            else:
                summary.append(f"• Total benchmark duration: {duration:.1f} seconds")
        
        return "<p>" + "</p><p>".join(summary) + "</p>"
    except Exception as e:
        logger.error(f"Error generating summary stats: {e}")
        return f"<p>Error generating summary statistics: {e}</p>"

def generate_benchmark_summary(results_data):
    """Generate executive summary of benchmark results"""
    try:
        summary = []
        
        # Models overview
        model_names = []
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            if model_name not in model_names:
                model_names.append(model_name)
        
        # Get best model for each metric
        best_throughput = {"model": None, "value": 0, "config": None}
        best_latency = {"model": None, "value": float('inf'), "config": None}
        lowest_memory = {"model": None, "value": float('inf')}
        
        # Analyze all results
        for result in results_data:
            model_id = result.get("model_id", "unknown")
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            
            # Process memory
            if "loading" in result and "memory_impact" in result["loading"]:
                mem = result["loading"]["memory_impact"].get("vram_used_gb", 0)
                if mem < lowest_memory["value"] and mem > 0:
                    lowest_memory["value"] = mem
                    lowest_memory["model"] = model_name
            
            # Process throughput and latency
            if "inference" in result:
                for config, data in result["inference"].items():
                    # Throughput
                    if "throughput" in data and "tokens_per_second" in data["throughput"]:
                        tps = data["throughput"]["tokens_per_second"]
                        if tps > best_throughput["value"]:
                            best_throughput["value"] = tps
                            best_throughput["model"] = model_name
                            best_throughput["config"] = config
                    
                    # Latency
                    if "time_to_first_token" in data:
                        latency = data["time_to_first_token"]["mean"] * 1000  # ms
                        if latency < best_latency["value"] and latency > 0:
                            best_latency["value"] = latency
                            best_latency["model"] = model_name
                            best_latency["config"] = config
        
        # Compile summary text
        summary.append(f"<p>Benchmark comparing {len(model_names)} models: {', '.join(model_names)}</p>")
        
        if best_throughput["model"]:
            summary.append(f"<p><strong>Best Throughput:</strong> {best_throughput['model']} " 
                        f"({best_throughput['value']:.2f} tokens/sec with {best_throughput['config']})</p>")
        
        if best_latency["model"]:
            summary.append(f"<p><strong>Lowest Latency:</strong> {best_latency['model']} "
                        f"({best_latency['value']:.2f} ms with {best_latency['config']})</p>")
        
        if lowest_memory["model"]:
            summary.append(f"<p><strong>Most Memory Efficient:</strong> {lowest_memory['model']} "
                        f"({lowest_memory['value']:.2f} GB)</p>")
        
        return "".join(summary)
    except Exception as e:
        logger.error(f"Error generating benchmark summary: {e}")
        return f"<p>Error generating benchmark summary: {e}</p>"

def generate_html_table(table_data):
    """Generate HTML table from data"""
    try:
        headers = table_data["headers"]
        rows = table_data["rows"]
        
        # Header row
        header_html = '<tr>\n'
        for header in headers:
            header_html += f'<th style="font-weight:700;border-top:1px solid #dddddd;border-right:1px solid #dddddd;border-bottom:1px solid #dddddd;border-left:1px solid #dddddd;">{header}</th>\n'
        header_html += '</tr>\n'
        
        # Data rows
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
    except Exception as e:
        logger.error(f"Error generating HTML table: {e}")
        return "<table><tr><td>Error generating table</td></tr></table>"

def generate_html_report(results_data, title="EffiLLM Benchmark Report"):
    """Generate complete HTML report from benchmark results"""
    try:
        # Load the HTML template from the same directory as this script
        template_path = os.path.join(os.path.dirname(__file__), "template.html")
        try:
            with open(template_path, "r") as f:
                html_template = f.read()
        except FileNotFoundError:
            logger.error(f"Template file not found at {template_path}")
            raise FileNotFoundError(f"Template file not found at {template_path}. Make sure template.html is in the same directory as report_generator.py")
        
        # Generate sections
        summary = generate_summary_stats(results_data)
        system_info = generate_system_info_section(results_data)
        benchmark_summary = generate_benchmark_summary(results_data)
        throughput = generate_throughput_charts(results_data)
        latency = generate_latency_charts(results_data)
        memory = generate_memory_charts(results_data)
        
        # Create a fresh HTML document
        html = html_template
        
        # Replace template placeholders
        html = html.replace('<span class="tinyMce-placeholder">Inference Benchmark Report</span>', title)
        
        # Add system info and benchmark summary to the main summary
        enhanced_summary = f"""
        <div class="benchmark-overview" style="margin-bottom:20px;">
            <div style="margin-bottom:20px;">
                <h3 style="margin:15px 0;color:#0e3d00;">Benchmark Overview</h3>
                {benchmark_summary}
            </div>
            
            <div style="margin-bottom:20px;">
                <h4 style="margin:15px 0;color:#0e3d00;">System Configuration</h4>
                {system_info}
            </div>
            
            <div style="margin-bottom:20px;">
                <h4 style="margin:15px 0;color:#0e3d00;">Benchmark Details</h4>
                {summary}
            </div>
        </div>
        """
        
        # Replace summary - be more specific with the replacement to avoid duplicates
        summary_block = '<div class="bee-block bee-block-2 bee-paragraph">\n<p>This should be a sample summary of whole report in 5 lines</p>\n</div>'
        if summary_block in html:
            html = html.replace(summary_block, f'<div class="bee-block bee-block-2 bee-paragraph">\n{enhanced_summary}\n</div>')
        
        # THROUGHPUT SECTION
        # Create a proper HTML for throughput charts with full width
        throughput_charts_html = '<div style="margin: 15px 0;">'
        for i, chart in enumerate(throughput["charts"]):
            throughput_charts_html += f'<div style="width:100%; margin-bottom:25px;"><img src="{chart}" style="width:100%; display:block; margin:0 auto;"/></div>'
        throughput_charts_html += '</div>'
        
        # Replace the throughput image block
        throughput_image_block = '<div class="bee-block bee-block-2 bee-image">\n<div></div>\n</div>'
        throughput_section_start = html.find('<span class="tinyMce-placeholder">Throughput Comparison</span>')
        if throughput_section_start > 0:
            # Find the image block after this heading
            block_start = html.find(throughput_image_block, throughput_section_start)
            if block_start > 0:
                block_end = block_start + len(throughput_image_block)
                html = html[:block_start] + f'<div class="bee-block bee-block-2 bee-image">\n{throughput_charts_html}\n</div>' + html[block_end:]
        
        # Replace the throughput table - look for exact table structure
        throughput_table_start = '<div class="bee-block bee-block-3 bee-table">\n<table style="table-layout:fixed;'
        throughput_section_start = html.find('<span class="tinyMce-placeholder">Throughput Comparison</span>')
        if throughput_section_start > 0:
            # Find the table section after the heading
            table_start = html.find(throughput_table_start, throughput_section_start)
            if table_start > 0:
                # Find the end of the table div
                table_div_end = html.find('</div>', table_start) + 6
                if table_div_end > 6:  # +6 for the length of "</div>"
                    html = html[:table_start] + '<div class="bee-block bee-block-3 bee-table">\n' + generate_html_table(throughput["table"]) + '\n</div>' + html[table_div_end:]
        
        # Replace the throughput summary
        throughput_summary = '<div class="bee-block bee-block-4 bee-paragraph">\n<p>This should be a sample summary of throughput</p>\n</div>'
        if throughput_section_start > 0:
            summary_start = html.find(throughput_summary, throughput_section_start)
            if summary_start > 0:
                summary_end = summary_start + len(throughput_summary)
                html = html[:summary_start] + f'<div class="bee-block bee-block-4 bee-paragraph">\n{throughput["summary"]}\n</div>' + html[summary_end:]
        
        # LATENCY SECTION
        # Create a proper HTML for latency charts with full width
        latency_charts_html = '<div style="margin: 15px 0;">'
        for i, chart in enumerate(latency["charts"]):
            latency_charts_html += f'<div style="width:100%; margin-bottom:25px;"><img src="{chart}" style="width:100%; display:block; margin:0 auto;"/></div>'
        latency_charts_html += '</div>'
        
        # Replace the latency image block
        latency_image_block = '<div class="bee-block bee-block-2 bee-image">\n<div></div>\n</div>'
        latency_section_start = html.find('<span class="tinyMce-placeholder">Latency Comparison</span>')
        if latency_section_start > 0:
            # Look for this specific image block after the latency heading
            block_start = html.find(latency_image_block, latency_section_start)
            if block_start > 0:
                block_end = block_start + len(latency_image_block)
                html = html[:block_start] + f'<div class="bee-block bee-block-2 bee-image">\n{latency_charts_html}\n</div>' + html[block_end:]
        
        # Replace the latency table
        latency_table_start = '<div class="bee-block bee-block-3 bee-table">\n<table style="table-layout:fixed;'
        if latency_section_start > 0:
            # Find the table after the latency heading
            table_start = html.find(latency_table_start, latency_section_start)
            if table_start > 0:
                # Find the end of the table div
                table_div_end = html.find('</div>', table_start) + 6
                if table_div_end > 6:
                    html = html[:table_start] + '<div class="bee-block bee-block-3 bee-table">\n' + generate_html_table(latency["table"]) + '\n</div>' + html[table_div_end:]
        
        # Replace the latency summary
        latency_summary = '<div class="bee-block bee-block-4 bee-paragraph">\n<p>This should be a sample summary of latency</p>\n</div>'
        if latency_section_start > 0:
            summary_start = html.find(latency_summary, latency_section_start)
            if summary_start > 0:
                summary_end = summary_start + len(latency_summary)
                html = html[:summary_start] + f'<div class="bee-block bee-block-4 bee-paragraph">\n{latency["summary"]}\n</div>' + html[summary_end:]
        
        # MEMORY SECTION
        # Create a proper HTML for memory charts with full width
        memory_charts_html = '<div style="margin: 15px 0;">'
        for i, chart in enumerate(memory["charts"]):
            memory_charts_html += f'<div style="width:100%; margin-bottom:25px;"><img src="{chart}" style="width:100%; display:block; margin:0 auto;"/></div>'
        memory_charts_html += '</div>'
        
        # Replace the memory image block
        memory_image_block = '<div class="bee-block bee-block-2 bee-image">\n<div></div>\n</div>'
        memory_section_start = html.find('<span class="tinyMce-placeholder">Memory Usage Comparison</span>')
        if memory_section_start > 0:
            # Look for the image block after the memory heading
            block_start = html.find(memory_image_block, memory_section_start)
            if block_start > 0:
                block_end = block_start + len(memory_image_block)
                html = html[:block_start] + f'<div class="bee-block bee-block-2 bee-image">\n{memory_charts_html}\n</div>' + html[block_end:]
        
        # Replace the memory table
        memory_table_start = '<div class="bee-block bee-block-3 bee-table">\n<table style="table-layout:fixed;'
        if memory_section_start > 0:
            # Find the table after the memory heading
            table_start = html.find(memory_table_start, memory_section_start)
            if table_start > 0:
                # Find the end of the table div
                table_div_end = html.find('</div>', table_start) + 6
                if table_div_end > 6:
                    html = html[:table_start] + '<div class="bee-block bee-block-3 bee-table">\n' + generate_html_table(memory["table"]) + '\n</div>' + html[table_div_end:]
        
        # Fix and replace the memory summary (which incorrectly has "latency" in the text)
        memory_summary_incorrect = '<div class="bee-block bee-block-4 bee-paragraph">\n<p>This should be a sample summary of latency</p>\n</div>'
        if memory_section_start > 0:
            # Look for this paragraph after the memory section
            summary_start = html.find(memory_summary_incorrect, memory_section_start)
            if summary_start > 0:
                summary_end = summary_start + len(memory_summary_incorrect)
                html = html[:summary_start] + f'<div class="bee-block bee-block-4 bee-paragraph">\n{memory["summary"]}\n</div>' + html[summary_end:]
        
        # Update logo and footer
        html = html.replace(
            '<img alt="" class="bee-autowidth" src="https://0c26875212.imgdist.com/pub/bfra/bpovlfhu/mx6/2cf/k54/logoipsum-345.svg" style="max-width:168px;" />', 
            '<img alt="" class="bee-autowidth" src="https://0c26875212.imgdist.com/pub/bfra/bpovlfhu/mx6/2cf/k54/logoipsum-345.svg" style="max-width:168px;" />'
        )
        
        # Add timestamp to footer
        import datetime
        html = html.replace(
            '<span class="tinyMce-placeholder">Report generated by EffiLLM</span>', 
            f'Report generated by EffiLLM on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
        )
        
        # Remove Beefree logo section completely
        beefree_row_start = html.find('<div class="bee-row bee-row-7">')
        if beefree_row_start > 0:
            beefree_row_end = html.find('</div>', html.find('</div>', html.find('</div>', beefree_row_start))) + 6
            if beefree_row_end > 6:
                html = html[:beefree_row_start] + html[beefree_row_end:]
        
        return html
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"""
        <html>
        <body>
            <h1>Error Generating Report</h1>
            <p>There was an error generating the benchmark report:</p>
            <pre>{str(e)}</pre>
            <pre>{traceback.format_exc()}</pre>
        </body>
        </html>
        """

def main():
    """Read benchmark results and generate HTML report"""
    try:
        # Create directory for reports
        os.makedirs("reports", exist_ok=True)
        
        # Load result files from results directory
        results_dir = "results"
        if not os.path.exists(results_dir):
            logger.error(f"Results directory '{results_dir}' not found")
            print(f"Error: Results directory '{results_dir}' not found. Create it and add benchmark JSON files.")
            return
        
        result_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]
        
        if not result_files:
            print("No result files found in the 'results' directory")
            return
        
        # Load the results
        results_data = []
        for file in result_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                results_data.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not results_data:
            print("No valid result files could be loaded")
            return
        
        print(f"Generating report from {len(results_data)} benchmark results")
        
        # Generate the HTML report
        html = generate_html_report(results_data, "EffiLLM Benchmark Results")
        
        # Save the report
        report_path = "reports/benchmark_report.html"
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"Report generated at {report_path}")
        
        # Try to display the report in a notebook environment
        try:
            from IPython.display import HTML, display
            print("Displaying report in notebook:")
            display(HTML(html))
        except ImportError:
            pass
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error generating report: {e}")

if __name__ == "__main__":
    main()