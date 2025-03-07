# EffiLLM/effillm/cli/commands.py
import click
import json
import logging
import time
import os
import sys
import torch
from typing import List
from pathlib import Path

from effillm.core.benchmark import LLMBenchmark
from effillm.core.utils import get_system_info, get_optimal_configurations
from effillm.reporting.export import ResultExporter
from effillm.reporting.visualize import generate_summary_report
from effillm.core.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
def main():
    """EffiLLM: Efficient LLM benchmarking tool."""
    pass

@main.command()
@click.argument('model_id')
@click.option('--device', default='auto', help='Device to run on (cuda, cpu, or auto)')
@click.option('--batch-sizes', default=None, help='Comma-separated batch sizes to test')
@click.option('--sequence-lengths', default=None, help='Comma-separated sequence lengths to test')
@click.option('--num-runs', default=10, help='Number of benchmark runs for each configuration')
@click.option('--quantization', default=None, help='Quantization method (e.g., int8, int4)')
@click.option('--output-file', default=None, help='Path to save benchmark results')
@click.option('--output-format', default='json', help='Format to save results (json, csv, markdown, excel)')
@click.option('--report-dir', default=None, help='Directory to save visual report')
@click.option('--auto-config', is_flag=True, help='Automatically determine optimal configurations')
def benchmark(
    model_id, 
    device, 
    batch_sizes, 
    sequence_lengths, 
    num_runs, 
    quantization, 
    output_file, 
    output_format,
    report_dir,
    auto_config
):
    """Run benchmark on a language model."""
    
    # Parse batch sizes and sequence lengths
    if auto_config:
        logger.info("Using automatic configuration detection...")
        configs = get_optimal_configurations(model_id, device)
        
        # Use recommended configurations
        batch_sizes_list = [c["batch_size"] for c in configs["recommended"]]
        sequence_lengths_list = [c["seq_length"] for c in configs["recommended"]]
        
        # Remove duplicates while preserving order
        batch_sizes_list = list(dict.fromkeys(batch_sizes_list))
        sequence_lengths_list = list(dict.fromkeys(sequence_lengths_list))
        
        logger.info(f"Auto-detected batch sizes: {batch_sizes_list}")
        logger.info(f"Auto-detected sequence lengths: {sequence_lengths_list}")
    else:
        # Parse manual configurations
        batch_sizes_list = [int(bs) for bs in batch_sizes.split(',')] if batch_sizes else [1, 4, 16]
        sequence_lengths_list = [int(sl) for sl in sequence_lengths.split(',')] if sequence_lengths else [128, 512, 1024]
    
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
    logger.info(f"Configuration: device={device}, quantization={quant_config}")
    
    # Setup output file
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
        quant_suffix = f"_{quantization}" if quantization else ""
        output_file = f"effillm_results_{model_short_name}{quant_suffix}_{timestamp}.{output_format}"
    
    # Run benchmark
    benchmark = LLMBenchmark(
        model_id=model_id,
        device=device,
        batch_sizes=batch_sizes_list,
        sequence_lengths=sequence_lengths_list,
        num_runs=num_runs,
        quantization_config=quant_config,
    )
    
    # Add system info to results
    sys_info = get_system_info()
    benchmark.results["system_info"] = sys_info
    
    results = benchmark.run_benchmark()
    
    # Export results
    ResultExporter.export(results, format=output_format, filepath=output_file)
    
    # Generate visual report if requested
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        report_path = generate_summary_report(results, output_dir=report_dir)
        logger.info(f"Visual report generated at: {report_path}")
    
    # Print summary
    if "inference" in results:
        click.echo("\nBenchmark Summary:")
        for config, data in results["inference"].items():
            throughput = data["throughput"]["tokens_per_second"]
            latency = data["time_to_first_token"]["mean"] * 1000  # Convert to ms
            click.echo(f"  {config}: {throughput:.2f} tokens/sec, {latency:.2f}ms latency")

@main.command()
@click.argument('model_id')
@click.option('--device', default='auto', help='Device to run on (cuda, cpu, or auto)')
@click.option('--quantization', default=None, help='Quantization method to test (int8, int4)')
@click.option('--output-file', default=None, help='Path to save evaluation results')
def evaluate(model_id, device, quantization, output_file):
    """Evaluate model quality vs efficiency tradeoffs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
    logger.info(f"Evaluating model quality and efficiency tradeoffs for {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load original model (FP16 on GPU)
    logger.info("Loading original (FP16) model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else torch.float32,
        device_map=device
    )
    
    # Load quantized model if requested
    quantized_model = None
    if quantization:
        logger.info(f"Loading {quantization} quantized model...")
        from effillm.quantization.quantizers import apply_quantization
        
        quant_config = None
        if quantization == "int8":
            quant_config = {"bits": 8, "method": "bitsandbytes"}
        elif quantization == "int4":
            quant_config = {"bits": 4, "method": "bitsandbytes"}
        
        quantized_model = apply_quantization(model_id, quant_config, device)
    
    # Load evaluation data
    logger.info("Loading evaluation dataset...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        eval_texts = dataset["text"][:50]  # Use a subset for efficiency
    except Exception as e:
        logger.warning(f"Failed to load wikitext dataset: {e}")
        logger.info("Using fallback evaluation text...")
        eval_texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. " * 10,
            "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. " * 8
        ]
    
    # Initialize evaluator
    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    evaluator = ModelEvaluator(tokenizer, device=actual_device)
    
    # Evaluate original model
    logger.info("Evaluating original model...")
    original_results = evaluator.evaluate_perplexity(original_model, eval_texts)
    
    # Setup results dictionary
    results = {
        "model_id": model_id,
        "device": actual_device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_model": {
            "dtype": str(next(original_model.parameters()).dtype),
            "perplexity": original_results["perplexity"],
            "loss": original_results["loss"]
        }
    }
    
    # Evaluate quantized model if available
    if quantized_model:
        logger.info("Evaluating quantized model...")
        quantized_results = evaluator.evaluate_perplexity(quantized_model, eval_texts)
        
        # Compare models
        logger.info("Comparing models...")
        comparison = evaluator.evaluate_quality_vs_speed(original_model, quantized_model, eval_texts[:10])
        
        # Add to results
        results["quantized_model"] = {
            "quantization": quantization,
            "dtype": str(next(quantized_model.parameters()).dtype),
            "perplexity": quantized_results["perplexity"],
            "loss": quantized_results["loss"]
        }
        
        results["comparison"] = comparison["comparison"]
    
    # Export results
    if output_file is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_short_name = model_id.split('/')[-1] if '/' in model_id else model_id
        output_file = f"effillm_eval_{model_short_name}_{timestamp}.json"
    
    ResultExporter.to_json(results, filepath=output_file)
    logger.info(f"Evaluation results saved to {output_file}")
    
    # Print summary
    click.echo("\nEvaluation Summary:")
    click.echo(f"  Original model perplexity: {original_results['perplexity']:.2f}")
    
    if quantized_model:
        efficiency_score = comparison["comparison"]["efficiency_score"]
        click.echo(f"  Quantized model perplexity: {quantized_results['perplexity']:.2f}")
        click.echo(f"  Speed improvement: {comparison['comparison']['speed_ratio']:.2f}x")
        click.echo(f"  Memory reduction: {1/comparison['comparison']['memory_ratio']:.2f}x")
        click.echo(f"  Quality ratio: {comparison['comparison']['quality_ratio']:.2f}x (higher means worse)")
        click.echo(f"  Overall efficiency score: {efficiency_score:.2f}")

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

@main.command()
@click.argument('model_id', required=False)
@click.option('--device', default='auto', help='Target device (cuda, cpu, or auto)')
def recommend(model_id, device):
    """Recommend optimal benchmark configurations."""
    if not model_id:
        click.echo("Please specify a model ID to get recommendations for.")
        click.echo("Example models:")
        click.echo("  - Small: facebook/opt-125m, EleutherAI/pythia-160m")
        click.echo("  - Medium: facebook/opt-1.3b, EleutherAI/pythia-1.4b")
        click.echo("  - Large: facebook/opt-6.7b, meta-llama/Llama-2-7b")
        click.echo("  - XL: meta-llama/Llama-2-13b, mistralai/Mistral-7B-v0.1")
        return
    
    # Get system info
    sys_info = get_system_info()
    
    click.echo(f"\nSystem Information:")
    click.echo(f"  Platform: {sys_info['platform']}")
    click.echo(f"  CPU: {sys_info['processor']} ({sys_info['cpu_count']} physical cores)")
    
    if 'gpu_devices' in sys_info:
        click.echo(f"  GPUs: {', '.join(sys_info['gpu_devices'])}")
    
    # Get recommendations
    click.echo(f"\nRecommendations for {model_id}:")
    
    configs = get_optimal_configurations(model_id, device)
    
    click.echo("\nRecommended Configurations:")
    for i, config in enumerate(configs["recommended"], 1):
        click.echo(f"  {i}. Batch Size: {config['batch_size']}, Sequence Length: {config['seq_length']}")
    
    if configs["aggressive"]:
        click.echo("\nAggressive Configurations (may be memory-intensive):")
        for i, config in enumerate(configs["aggressive"], 1):
            click.echo(f"  {i}. Batch Size: {config['batch_size']}, Sequence Length: {config['seq_length']}")
    
    # Generate command example
    batch_sizes = ",".join(str(c["batch_size"]) for c in configs["recommended"][:3])
    seq_lengths = ",".join(str(c["seq_length"]) for c in configs["recommended"][:3])
    
    click.echo("\nSuggested Command:")
    click.echo(f"  effillm benchmark {model_id} --batch-sizes {batch_sizes} --sequence-lengths {seq_lengths} --device {device}")
    
    # Quantization recommendations
    click.echo("\nQuantization Recommendations:")
    if torch.cuda.is_available() and device != "cpu":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if "13b" in model_id.lower() or "20b" in model_id.lower() or "65b" in model_id.lower() or "70b" in model_id.lower():
            click.echo("  This is a large model. Consider using quantization:")
            click.echo(f"  effillm benchmark {model_id} --quantization int8 --device {device}")
            click.echo(f"  effillm benchmark {model_id} --quantization int4 --device {device}")
        elif vram_gb < 16:
            click.echo("  Your GPU has limited VRAM. Consider using quantization:")
            click.echo(f"  effillm benchmark {model_id} --quantization int8 --device {device}")
        else:
            click.echo("  Your system should handle this model in FP16 precision.")
            click.echo("  For comparison, you can also test with quantization:")
            click.echo(f"  effillm benchmark {model_id} --quantization int8 --device {device}")

if __name__ == "__main__":
    main()