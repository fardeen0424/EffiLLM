# EffiLLM/examples/model_comparison.py
import os
import logging
from effillm.core.benchmark import LLMBenchmark
from effillm.reporting.compare import generate_comparison_report

logging.basicConfig(level=logging.INFO)

def run_benchmark(model_id, result_dir="results"):
    """Run benchmark on a specific model and save results."""
    os.makedirs(result_dir, exist_ok=True)
    
    # Get model name for filename
    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
    result_file = os.path.join(result_dir, f"{model_name}.json")
    
    # Skip if results already exist
    if os.path.exists(result_file):
        print(f"Results for {model_id} already exist at {result_file}")
        return result_file
    
    print(f"Running benchmark on {model_id}...")
    
    # Initialize the benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device="auto",
        batch_sizes=[1, 2, 4],
        sequence_lengths=[128, 256],
        warmup_runs=2,
        num_runs=5
    )
    
    # Run the benchmark
    benchmark.run_benchmark()
    
    # Save results
    benchmark.export_results(format="json", filepath=result_file)
    
    print(f"Benchmark for {model_id} complete! Results saved to {result_file}")
    return result_file

def main():
    """Run benchmarks on multiple models and generate comparison report."""
    # List of models to benchmark
    models = [
        "facebook/opt-125m",      # Small model
        "facebook/opt-350m",      # Medium model
        "EleutherAI/pythia-410m"  # Different architecture
    ]
    
    # Run benchmarks for all models
    result_files = []
    for model in models:
        result_file = run_benchmark(model)
        result_files.append(result_file)
    
    # Generate comparison report
    report_path = generate_comparison_report(result_files, output_dir="results/comparison")
    
    print(f"Comparison complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()