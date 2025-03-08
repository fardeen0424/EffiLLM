# EffiLLM/tests/full_benchmark_and_report.py
import os
import sys
import time
import logging
import json
from pathlib import Path
import torch
import argparse
import shutil

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from effillm.core.benchmark import LLMBenchmark
from effillm.core.utils import get_system_info
from effillm.reporting import report_generator
import effillm.reporting.report_generator as report_gen

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configurations
RESULTS_DIR = os.path.join(project_root, "benchmark_results")
REPORTS_DIR = os.path.join(project_root, "benchmark_reports")

def run_benchmark(model_id, batch_sizes, seq_lengths, device="cuda", quantization=None):
    """Run benchmark for a specific model with given parameters."""
    # Create descriptive name for logging
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    quant_desc = f" with {quantization['bits']}-bit quantization" if quantization else " (FP16)"
    
    logger.info(f"Benchmarking {model_name}{quant_desc}")
    
    # Initialize benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device=device,
        batch_sizes=batch_sizes,
        sequence_lengths=seq_lengths,
        warmup_runs=2,
        num_runs=3,  # Use 3 runs for faster testing
        quantization_config=quantization
    )
    
    # Add system info
    benchmark.results["system_info"] = get_system_info()
    
    # Run benchmark
    start_time = time.time()
    results = benchmark.run_benchmark()
    benchmark_time = time.time() - start_time
    
    # Add quantization info
    if quantization:
        results["quantization"] = quantization
    
    # Generate descriptive filename
    quant_suffix = f"_{quantization['bits']}bit" if quantization else "_fp16"
    result_file = os.path.join(RESULTS_DIR, f"{model_name}{quant_suffix}.json")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # Export results
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark completed in {benchmark_time:.1f} seconds")
    logger.info(f"Results saved to: {result_file}")
    
    return result_file

def generate_reports(result_files):
    """Generate HTML reports from benchmark results."""
    if not result_files:
        logger.error("No result files to generate reports!")
        return
    
    # Create directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load result data
    results_data = []
    for file in result_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results_data.append(data)
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not results_data:
        logger.error("No valid benchmark data found!")
        return
    
    # Generate comprehensive report
    logger.info(f"Generating report from {len(results_data)} benchmark results")
    
    # Generate HTML report
    html_content = report_gen.generate_html_report(results_data, "EffiLLM Benchmark Results")
    
    # Save the report
    report_path = os.path.join(REPORTS_DIR, "benchmark_report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Benchmark report generated at: {report_path}")
    
    # Generate model-specific reports if multiple models were tested
    models = {}
    for result in results_data:
        model_id = result.get("model_id", "unknown")
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        if model_name not in models:
            models[model_name] = []
        
        models[model_name].append(result)
    
    # Generate per-model reports if we have quantization variations
    for model_name, model_results in models.items():
        if len(model_results) > 1:
            logger.info(f"Generating quantization comparison report for {model_name}...")
            
            html_content = report_gen.generate_html_report(
                model_results, 
                f"EffiLLM Quantization Impact: {model_name}"
            )
            
            report_path = os.path.join(REPORTS_DIR, f"{model_name}_quantization_report.html")
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Model-specific report generated at: {report_path}")
    
    return REPORTS_DIR

def main():
    """Run full benchmark tests and generate reports."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run EffiLLM benchmarks and generate reports")
    parser.add_argument("--models", nargs="+", default=["facebook/opt-125m", "facebook/opt-350m"], 
                      help="Models to benchmark")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4], 
                      help="Batch sizes to test")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256], 
                      help="Sequence lengths to test")
    parser.add_argument("--quantization", nargs="+", default=["fp16", "int8"], 
                      choices=["fp16", "int8", "int4"],
                      help="Quantization methods to test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run on (cuda or cpu)")
    parser.add_argument("--skip-benchmarks", action="store_true",
                      help="Skip benchmarks, only generate reports from existing results")
    parser.add_argument("--clear-results", action="store_true",
                      help="Clear previous benchmark results")
    
    args = parser.parse_args()
    
    # Check if template exists
    template_path = os.path.join(project_root, "effillm", "reporting", "template.html")
    if not os.path.exists(template_path):
        logger.error(f"Template file not found at {template_path}")
        return 1
    
    # Show hardware info
    if torch.cuda.is_available() and args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Running on GPU: {gpu_name} with {vram_gb:.1f}GB VRAM")
    else:
        logger.info("Running on CPU (this will be slower)")
    
    # Create/clear results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.clear_results:
        logger.info("Clearing previous benchmark results...")
        for file in os.listdir(RESULTS_DIR):
            if file.endswith(".json"):
                os.remove(os.path.join(RESULTS_DIR, file))
    
    # Run benchmarks
    result_files = []
    
    if not args.skip_benchmarks:
        for model_id in args.models:
            for quant_method in args.quantization:
                try:
                    # Configure quantization
                    quant_config = None
                    if quant_method == "int8":
                        quant_config = {"bits": 8, "method": "bitsandbytes"}
                    elif quant_method == "int4":
                        quant_config = {"bits": 4, "method": "bitsandbytes"}
                    # fp16 is None (default)
                    
                    # Skip if model is too large for device
                    if "7b" in model_id.lower() and args.device == "cpu":
                        logger.warning(f"Skipping {model_id} on CPU (too large)")
                        continue
                    
                    # Run benchmark
                    result_file = run_benchmark(
                        model_id, 
                        args.batch_sizes, 
                        args.seq_lengths, 
                        args.device, 
                        quant_config
                    )
                    
                    result_files.append(result_file)
                    
                    # Clean GPU cache between runs
                    if torch.cuda.is_available() and args.device == "cuda":
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error benchmarking {model_id} with {quant_method}: {e}")
    else:
        # Find existing result files
        for file in os.listdir(RESULTS_DIR):
            if file.endswith(".json"):
                result_files.append(os.path.join(RESULTS_DIR, file))
        
        if not result_files:
            logger.error("No benchmark results found in results directory.")
            return 1
    
    # Generate reports
    report_dir = generate_reports(result_files)
    
    # Try to display report in IPython if in notebook
    try:
        from IPython.display import IFrame, display
        report_path = os.path.join(report_dir, "benchmark_report.html")
        if os.path.exists(report_path):
            logger.info("Displaying report in notebook...")
            display(IFrame(src=report_path, width=1000, height=800))
    except ImportError:
        pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())