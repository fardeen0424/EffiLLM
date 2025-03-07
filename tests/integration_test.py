# EffiLLM/tests/integration_test.py
import sys
import os
from pathlib import Path
import logging
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from effillm.core.benchmark import LLMBenchmark
from effillm.reporting.export import ResultExporter
from effillm.reporting.visualize import generate_summary_report
from effillm.core.metrics import MetricsCollector
from effillm.core.utils import get_system_info, is_configuration_feasible

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_benchmark_pipeline():
    """Test the complete benchmark pipeline with a small model."""
    
    # Use a tiny model for quick testing
    model_id = "facebook/opt-125m"  # ~125M parameters
    
    print(f"\n{'='*50}")
    print(f"Running EffiLLM integration test with {model_id}")
    print(f"{'='*50}")
    
    # Get system info
    sys_info = get_system_info()
    print("\nSystem information:")
    print(json.dumps(sys_info, indent=2))
    
    # Check configuration feasibility
    feasible, message = is_configuration_feasible(2, 512, model_id)
    print(f"\nConfiguration feasibility: {feasible}")
    print(f"Message: {message}")
    
    # Initialize the benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device="auto",
        batch_sizes=[1, 2],
        sequence_lengths=[128],
        warmup_runs=1,
        num_runs=2  # Use small number for testing
    )
    
    # Run the benchmark
    print("\nRunning benchmark...")
    results = benchmark.run_benchmark()
    
    # Export results in different formats
    os.makedirs("test_results", exist_ok=True)
    
    print("\nExporting results...")
    json_output = ResultExporter.to_json(results, filepath="test_results/results.json")
    csv_output = ResultExporter.to_csv(results, filepath="test_results/results.csv")
    md_output = ResultExporter.to_markdown(results, filepath="test_results/results.md")
    
    # Generate visualization
    print("\nGenerating visualization...")
    report_path = generate_summary_report(results, output_dir="test_results/report")
    
    print(f"\nIntegration test completed successfully!")
    print(f"Results saved to: test_results/")
    print(f"HTML report: {report_path}")

if __name__ == "__main__":
    test_full_benchmark_pipeline()