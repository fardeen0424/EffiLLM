# EffiLLM/effillm/reporting/export.py
import json
import csv
import os
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ResultExporter:
    """Handles exporting benchmark results in different formats."""
    
    @staticmethod
    def to_json(results: Dict[str, Any], filepath: Optional[str] = None, pretty: bool = True) -> str:
        """Export results to JSON format."""
        # Handle non-serializable objects like numpy arrays
        def json_serializable(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return str(obj)
                
        indent = 2 if pretty else None
        result_str = json.dumps(results, indent=indent, default=json_serializable)
        
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(result_str)
            logger.info(f"Results saved to {filepath}")
            
        return result_str
    
    @staticmethod
    def to_csv(results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export results to CSV format."""
        # Convert nested results to a flattened dataframe
        rows = []
        
        model_id = results.get("model_id", "unknown")
        device = results.get("device", "unknown")
        
        # Process inference results
        if "inference" in results:
            for config, config_data in results["inference"].items():
                row = {
                    "model_id": model_id,
                    "device": device,
                    "config": config,
                }
                
                # Extract batch size and sequence length from config key
                if "bs" in config and "seq" in config:
                    try:
                        # Parse config like "bs1_seq128"
                        parts = config.split("_")
                        batch_size = int(parts[0].replace("bs", ""))
                        seq_len = int(parts[1].replace("seq", ""))
                        row["batch_size"] = batch_size
                        row["sequence_length"] = seq_len
                    except (ValueError, IndexError):
                        pass
                
                # Throughput metrics
                if "throughput" in config_data:
                    row["tokens_per_second"] = config_data["throughput"].get("tokens_per_second", 0)
                    row["tokens_per_second_per_instance"] = config_data["throughput"].get("tokens_per_second_per_instance", 0)
                
                # Latency metrics
                if "time_to_first_token" in config_data:
                    row["time_to_first_token_ms"] = config_data["time_to_first_token"].get("mean", 0) * 1000
                    row["time_to_first_token_std_ms"] = config_data["time_to_first_token"].get("std", 0) * 1000
                
                # Memory metrics
                if "memory" in config_data and "impact" in config_data["memory"]:
                    memory_impact = config_data["memory"]["impact"]
                    row["ram_used_gb"] = memory_impact.get("ram_used_gb", 0)
                    row["vram_used_gb"] = memory_impact.get("vram_used_gb", 0)
                
                # Generate average if available
                if "generation_time" in config_data:
                    row["generation_time_sec"] = config_data["generation_time"].get("mean", 0)
                    row["generation_time_std_sec"] = config_data["generation_time"].get("std", 0)
                
                rows.append(row)
        
        # Handle empty results
        if not rows:
            rows.append({
                "model_id": model_id,
                "device": device,
                "error": "No inference results found"
            })
        
        # Convert to dataframe
        df = pd.DataFrame(rows)
        
        # Export
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"CSV results saved to {filepath}")
            
        # Return as string
        return df.to_csv(index=False)
    
    @staticmethod
    def to_markdown(results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export summary results to Markdown format for documentation."""
        model_id = results.get("model_id", "unknown")
        device = results.get("device", "unknown")
        
        md_lines = [
            f"# EffiLLM Benchmark Results: {model_id}",
            "",
            f"- **Model:** {model_id}",
            f"- **Device:** {device}",
            ""
        ]
        
        # Add loading time if available
        if "loading" in results:
            loading_time = results["loading"].get("time_seconds", 0)
            md_lines.extend([
                "## Loading Statistics",
                "",
                f"- **Loading Time:** {loading_time:.2f} seconds",
                ""
            ])
            
            # Add memory impact if available
            if "memory_impact" in results["loading"]:
                mem_impact = results["loading"]["memory_impact"]
                ram_used = mem_impact.get("ram_used_gb", 0)
                vram_used = mem_impact.get("vram_used_gb", 0)
                
                md_lines.extend([
                    f"- **RAM Impact:** {ram_used:.2f} GB",
                    f"- **VRAM Impact:** {vram_used:.2f} GB" if vram_used else "",
                    ""
                ])
        
        # Add inference results
        if "inference" in results and results["inference"]:
            md_lines.extend([
                "## Inference Performance",
                "",
                "| Configuration | Throughput (tokens/sec) | Latency (ms) | Memory Usage (GB) |",
                "|---------------|-------------------------|--------------|------------------|"
            ])
            
            for config, data in results["inference"].items():
                throughput = data["throughput"]["tokens_per_second"] if "throughput" in data else "N/A"
                latency = data["time_to_first_token"]["mean"] * 1000 if "time_to_first_token" in data else "N/A"
                
                mem_usage = "N/A"
                if "memory" in data and "impact" in data["memory"]:
                    mem_impact = data["memory"]["impact"]
                    vram_used = mem_impact.get("vram_used_gb", 0)
                    ram_used = mem_impact.get("ram_used_gb", 0)
                    mem_usage = vram_used if vram_used else ram_used
                
                md_lines.append(f"| {config} | {throughput:.2f} | {latency:.2f} | {mem_usage:.2f} |")
            
            md_lines.append("")
        
        md_content = "\n".join(md_lines)
        
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(md_content)
            logger.info(f"Markdown results saved to {filepath}")
        
        return md_content
    
    @staticmethod
    def to_excel(results: Dict[str, Any], filepath: str) -> None:
        """Export results to Excel format with multiple sheets."""
        if not filepath.endswith(('.xlsx', '.xls')):
            filepath += '.xlsx'
            
        # Create a Pandas Excel writer
        writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
        
        # Overall summary
        summary_data = {
            "Model ID": [results.get("model_id", "unknown")],
            "Device": [results.get("device", "unknown")]
        }
        
        if "loading" in results:
            summary_data["Loading Time (s)"] = [results["loading"].get("time_seconds", 0)]
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Inference results
        if "inference" in results and results["inference"]:
            rows = []
            for config, data in results["inference"].items():
                row = {"Configuration": config}
                
                # Extract metrics
                if "throughput" in data:
                    row["Throughput (tokens/sec)"] = data["throughput"].get("tokens_per_second", 0)
                    row["Throughput per Instance"] = data["throughput"].get("tokens_per_second_per_instance", 0)
                
                if "time_to_first_token" in data:
                    row["Time to First Token (ms)"] = data["time_to_first_token"].get("mean", 0) * 1000
                    row["TTFT Std Dev (ms)"] = data["time_to_first_token"].get("std", 0) * 1000
                
                if "generation_time" in data:
                    row["Generation Time (s)"] = data["generation_time"].get("mean", 0)
                    row["Generation Time Std Dev (s)"] = data["generation_time"].get("std", 0)
                
                if "memory" in data and "impact" in data["memory"]:
                    memory_impact = data["memory"]["impact"]
                    row["RAM Used (GB)"] = memory_impact.get("ram_used_gb", 0)
                    row["VRAM Used (GB)"] = memory_impact.get("vram_used_gb", 0)
                
                rows.append(row)
            
            inference_df = pd.DataFrame(rows)
            inference_df.to_excel(writer, sheet_name='Inference Results', index=False)
        
        # Save the Excel file
        writer.close()
        logger.info(f"Excel results saved to {filepath}")
    
    @staticmethod
    def export(results: Dict[str, Any], format: str = "json", filepath: Optional[str] = None) -> Union[str, None]:
        """Export results to the specified format."""
        if format.lower() == "json":
            return ResultExporter.to_json(results, filepath)
        elif format.lower() == "csv":
            return ResultExporter.to_csv(results, filepath)
        elif format.lower() == "markdown" or format.lower() == "md":
            return ResultExporter.to_markdown(results, filepath)
        elif format.lower() == "xlsx" or format.lower() == "excel":
            if not filepath:
                raise ValueError("Filepath must be provided for Excel export")
            ResultExporter.to_excel(results, filepath)
            return None
        else:
            raise ValueError(f"Unsupported export format: {format}")