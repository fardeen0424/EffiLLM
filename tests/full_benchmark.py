# EffiLLM/examples/colab_benchmark_to_report.py
import os
import sys
import time
import logging
import json
from pathlib import Path
import torch
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from effillm.core.benchmark import LLMBenchmark
from effillm.core.utils import get_system_info, is_configuration_feasible
from effillm.reporting.report_generator import BenchmarkReportGenerator, save_html_template
from effillm.reporting.export import ResultExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======== Configuration ========
# Use smaller models for Colab
BASE_MODELS = [
    "facebook/opt-125m",    # Tiny model
    "facebook/opt-350m",    # Small model
]

# Try one reference test with a larger model if resources allow
OPTIONAL_MODELS = [
    "facebook/opt-1.3b",    # Medium model
]

# Quantization configs to test (only try INT8 for Colab)
QUANTIZATION_CONFIGS = [
    None,                                    # Baseline (no quantization)
    {"bits": 8, "method": "bitsandbytes"},   # INT8
]

# Smaller test matrix for Colab
BATCH_SIZES = [1, 2, 4]  
SEQUENCE_LENGTHS = [128, 256]

# Directories
RESULTS_DIR = "colab_benchmark_results"
REPORT_DIR = "colab_benchmark_reports"

# ======== Helper Functions ========

def is_model_feasible(model_id, quant_config=None):
    """Check if model is feasible to run given system resources."""
    # For larger models, check VRAM
    if "1.3b" in model_id.lower() or "7b" in model_id.lower():
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Skip large models if VRAM is limited
            if "7b" in model_id.lower() and vram_gb < 16:
                logger.warning(f"Skipping {model_id} (requires >16GB VRAM)")
                return False
            if "1.3b" in model_id.lower() and vram_gb < 8:
                logger.warning(f"Skipping {model_id} (requires >8GB VRAM)")
                return False
    return True

def get_result_filename(model_id, quant_config=None):
    """Generate a consistent filename for results."""
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    if quant_config:
        return f"{model_name}_{quant_config['bits']}bit.json"
    else:
        return f"{model_name}_fp16.json"

def run_benchmark(model_id, quant_config=None):
    """Run benchmark for a specific model with optional quantization."""
    # Skip if model is too large for our system
    if not is_model_feasible(model_id, quant_config):
        return None
    
    # Create a descriptive name for logging
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    quant_desc = f" with {quant_config['bits']}-bit quantization" if quant_config else " (FP16)"
    
    logger.info(f"Benchmarking {model_name}{quant_desc}")
    
    # Determine appropriate batch sizes and sequence lengths for this model
    if "1.3b" in model_id.lower():
        # For larger models, use smaller configs
        batch_sizes = [1, 2]
        sequence_lengths = [128]
    else:
        # For smaller models, use default configs
        batch_sizes = BATCH_SIZES
        sequence_lengths = SEQUENCE_LENGTHS
    
    # Initialize benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        warmup_runs=1,  # Reduce for Colab
        num_runs=3,     # Reduce for Colab
        quantization_config=quant_config
    )
    
    # Add system info to results
    benchmark.results["system_info"] = get_system_info()
    
    # Run benchmark
    start_time = time.time()
    results = benchmark.run_benchmark()
    benchmark_time = time.time() - start_time
    
    # Add quantization info to results
    if quant_config:
        results["quantization"] = quant_config
    
    # Add timing info
    results["benchmark_duration_seconds"] = benchmark_time
    
    # Export results
    result_filename = get_result_filename(model_id, quant_config)
    result_path = os.path.join(RESULTS_DIR, result_filename)
    ResultExporter.export(results, format="json", filepath=result_path)
    
    logger.info(f"Benchmark for {model_name}{quant_desc} completed in {benchmark_time:.1f} seconds")
    logger.info(f"Results saved to: {result_path}")
    
    return result_path

def generate_reports():
    """Generate various reports from benchmark results."""
    # Ensure report directory exists
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Ensure template file exists
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                "effillm", "reporting", "report_template.html")
    template_dir = os.path.dirname(template_path)
    os.makedirs(template_dir, exist_ok=True)
    
    if not os.path.exists(template_path):
        save_html_template(template_path)
    
    # Get all result files
    result_files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) 
                  if f.endswith(".json")]
    
    if not result_files:
        logger.error("No result files found for report generation!")
        return
    
    # Generate comprehensive report
    logger.info(f"Generating comprehensive benchmark report...")
    generator = BenchmarkReportGenerator(result_files, output_dir=REPORT_DIR)
    report_path = generator.generate_html_report(
        title="EffiLLM Benchmark Report (Colab Test)",
        filename="colab_report.html"
    )
    logger.info(f"Report generated: {report_path}")
    
    # Try to display the report in Colab
    try:
        from IPython.display import HTML, display
        with open(report_path, 'r') as f:
            html_content = f.read()
        
        # Create a simplified version for display
        logger.info("Displaying report in notebook...")
        display(HTML(html_content))
    except ImportError:
        logger.info(f"Report saved to {report_path}")
    
    return report_path

# ======== Main Testing Script ========

def main():
    """Main testing function that runs all benchmarks and generates reports."""
    logger.info("Starting EffiLLM benchmark and reporting test on Colab")
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Running on GPU: {device_name} with {vram_gb:.1f}GB VRAM")
    else:
        logger.warning("CUDA not available, running on CPU (this will be slow)")
    
    # Create/clean results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Track all successful benchmark results
    result_files = []
    
    # Run benchmarks for basic models
    for model_id in BASE_MODELS:
        for quant_config in QUANTIZATION_CONFIGS:
            try:
                result_file = run_benchmark(model_id, quant_config)
                if result_file:
                    result_files.append(result_file)
                    
                # Clean up CUDA cache between runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error benchmarking {model_id}: {e}")
                continue
    
    # Try optional larger model if we have resources
    for model_id in OPTIONAL_MODELS:
        try:
            # Only try with INT8 quantization for larger models on Colab
            quant_config = {"bits": 8, "method": "bitsandbytes"}
            result_file = run_benchmark(model_id, quant_config)
            if result_file:
                result_files.append(result_file)
                
            # Clean up CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Error benchmarking optional model {model_id}: {e}")
            continue
    
    # Generate reports
    if result_files:
        logger.info(f"Benchmarks completed. Generating reports from {len(result_files)} result files")
        final_report = generate_reports()
        logger.info(f"All reports generated successfully.")
    else:
        logger.error("No successful benchmarks completed. Cannot generate reports.")
    
    logger.info("Colab test completed")
    
    # Return success status
    return len(result_files) > 0

if __name__ == "__main__":
    success = main()
    # Set exit code for CI/CD pipelines
    sys.exit(0 if success else 1)