# EffiLLM/effillm/cli/commands.py
import click
import json
import logging
import time
import os
from effillm.core.benchmark import LLMBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def main():
    """EffiLLM: Efficient LLM benchmarking tool."""
    pass

@main.command()
@click.argument('model_id')
@click.option('--device', default='auto', help='Device to run on (cuda, cpu, or auto)')
@click.option('--batch-sizes', default='1,4,16', help='Comma-separated batch sizes to test')
@click.option('--sequence-lengths', default='128,512,1024', help='Comma-separated sequence lengths to test')
@click.option('--num-runs', default=10, help='Number of benchmark runs for each configuration')
@click.option('--quantization', default=None, help='Quantization method (e.g., int8, int4, or bitsandbytes)')
@click.option('--output-file', default=None, help='Path to save benchmark results')
@click.option('--output-format', default='json', help='Format to save results (json or csv)')
def benchmark(
    model_id, 
    device, 
    batch_sizes, 
    sequence_lengths, 
    num_runs, 
    quantization, 
    output_file, 
    output_format
):
    """Run benchmark on a language model."""
    
    # Parse batch sizes and sequence lengths
    batch_sizes = [int(bs) for bs in batch_sizes.split(',')]
    sequence_lengths = [int(sl) for sl in sequence_lengths.split(',')]
    
    # Parse quantization config if provided
    quant_config = None
    if quantization:
        if quantization == "int8":
            quant_config = {"bits": 8, "method": "bitsandbytes"}
        elif quantization == "int4":
            quant_config = {"bits": 4, "method": "bitsandbytes"}
        elif ":" in quantization:
            method, bits = quantization.split(":")
            quant_config = {"bits": int(bits), "method": method}
    
    logger.info(f"Starting benchmark for model: {model_id}")
    logger.info(f"Configuration: device={device}, batch_sizes={batch_sizes}, "
                f"sequence_lengths={sequence_lengths}, quantization={quant_config}")
    
    # Setup output file
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
        output_file = f"effillm_results_{model_short_name}_{timestamp}.{output_format}"
    
    # Run benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device=device,
        batch_sizes=batch_sizes,
        sequence_lengths=sequence_lengths,
        num_runs=num_runs,
        quantization_config=quant_config,
    )
    
    results = benchmark.run_benchmark()
    output = benchmark.export_results(format=output_format, filepath=output_file)
    
    logger.info(f"Benchmark completed. Results saved to: {output_file}")
    
    # Print summary
    if "inference" in results:
        click.echo("\nBenchmark Summary:")
        for config, data in results["inference"].items():
            throughput = data["throughput"]["tokens_per_second"]
            latency = data["time_to_first_token"]["mean"] * 1000  # Convert to ms
            click.echo(f"  {config}: {throughput:.2f} tokens/sec, {latency:.2f}ms latency")

@main.command()
def list_devices():
    """List available devices for benchmarking."""
    import torch
    
    click.echo("Available devices:")
    click.echo("  CPU: Available")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        click.echo(f"  CUDA: Available ({device_count} devices)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            click.echo(f"    [{i}] {device_name} ({total_memory:.2f} GB)")
    else:
        click.echo("  CUDA: Not available")

if __name__ == "__main__":
    main()