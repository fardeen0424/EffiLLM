# EffiLLM/examples/enhanced_colab_test.py
import os
import sys
import time
import logging
import json
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from effillm.core.benchmark import LLMBenchmark
from effillm.core.utils import get_system_info
from effillm.reporting.report_generator import BenchmarkReportGenerator, save_html_template
from effillm.reporting.export import ResultExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
RESULTS_DIR = "enhanced_test_results"
REPORT_DIR = "enhanced_test_reports"

def run_benchmark(model_id, batch_sizes, seq_lengths, quantization=None):
    """Run benchmark for a specific model with given configuration."""
    logger.info(f"Benchmarking {model_id}")
    
    # Initialize benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_sizes=batch_sizes,
        sequence_lengths=seq_lengths,
        warmup_runs=1,
        num_runs=3,
        quantization_config=quantization
    )
    
    # Add system info to results
    benchmark.results["system_info"] = get_system_info()
    
    # Run benchmark
    start_time = time.time()
    results = benchmark.run_benchmark()
    benchmark_time = time.time() - start_time
    
    # Add quantization info to results
    if quantization:
        results["quantization"] = quantization
    
    # Generate descriptive filename
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    quant_suffix = f"_{quantization['bits']}bit" if quantization else "_fp16"
    
    # Export results
    result_file = os.path.join(RESULTS_DIR, f"{model_name}{quant_suffix}.json")
    ResultExporter.export(results, format="json", filepath=result_file)
    
    logger.info(f"Benchmark completed in {benchmark_time:.1f} seconds")
    logger.info(f"Results saved to: {result_file}")
    
    return result_file

def prepare_template():
    """Prepare the report template."""
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              "effillm", "reporting", "report_template.html")
    template_dir = os.path.dirname(template_path)
    os.makedirs(template_dir, exist_ok=True)
    
    if not os.path.exists(template_path):
        save_html_template(template_path)
        logger.info(f"Created template at {template_path}")
    else:
        logger.info(f"Using existing template at {template_path}")

def main():
    """Run benchmark tests and generate report with multiple visualizations."""
    logger.info("Starting enhanced benchmark test")
    
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Prepare report template
    prepare_template()
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Using GPU: {gpu_name} with {vram_gb:.2f} GB VRAM")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Test configurations
    # Smaller configurations for quicker testing
    basic_batch_sizes = [1, 2, 4]
    basic_seq_lengths = [128, 256]
    
    # Run benchmarks for a few models to generate comparison data
    result_files = []
    
    # Test OPT-125M with different configurations
    try:
        # FP16 version
        result_file = run_benchmark("facebook/opt-125m", basic_batch_sizes, basic_seq_lengths)
        result_files.append(result_file)
        
        # INT8 version
        quant_config = {"bits": 8, "method": "bitsandbytes"}
        result_file = run_benchmark("facebook/opt-125m", basic_batch_sizes, basic_seq_lengths, quant_config)
        result_files.append(result_file)
        
        # Clean GPU memory between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error benchmarking opt-125m: {e}")
    
    # Test another model if time permits
    try:
        # Try OPT-350M if resources allow
        result_file = run_benchmark("facebook/opt-350m", [1, 2], [128])
        result_files.append(result_file)
        
        # Clean GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error benchmarking opt-350m: {e}")
    
    # Generate enhanced report
    if result_files:
        logger.info(f"Generating enhanced report from {len(result_files)} result files")
        generator = BenchmarkReportGenerator(result_files, output_dir=REPORT_DIR)
        report_path = generator.generate_html_report(
            title="EffiLLM Enhanced Visualization Report",
            filename="enhanced_report.html"
        )
        logger.info(f"Enhanced report generated at: {report_path}")
        
        # Try to display the report in Colab
        try:
            from IPython.display import HTML, display
            with open(report_path, 'r') as f:
                html_content = f.read()
            logger.info("Displaying report in notebook...")
            display(HTML(html_content))
        except ImportError:
            logger.info(f"Report saved to {report_path}")
    else:
        logger.error("No benchmark results to generate report")
    
    logger.info("Enhanced test completed")
    
    return result_files

if __name__ == "__main__":
    main()