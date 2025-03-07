# EffiLLM/tests/test_basic_benchmark.py
import os
import sys
import logging
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from effillm.core.benchmark import LLMBenchmark

logging.basicConfig(level=logging.INFO)

def test_basic_benchmark():
    """Test basic benchmarking functionality with a small model."""
    
    # Use a small model for testing
    model_id = "facebook/opt-125m"  # Small model for quick testing
    
    print(f"\n\n{'-'*50}")
    print(f"Testing EffiLLM benchmark with model: {model_id}")
    print(f"{'-'*50}\n")
    
    # Initialize benchmark with minimal configurations for testing
    benchmark = LLMBenchmark(
        model_id=model_id,
        device="auto",
        batch_sizes=[1, 2],
        sequence_lengths=[128],
        warmup_runs=1,
        num_runs=2  # Use small number for testing
    )
    
    # Run the benchmark
    results = benchmark.run_benchmark()
    
    # Export and print results
    output = benchmark.export_results(format="json")
    print("\nBenchmark Results Summary:")
    print(json.dumps(results, indent=2))
    
    # Verify results structure
    assert "model_id" in results
    assert "device" in results
    assert "loading" in results
    assert "inference" in results
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_basic_benchmark()