# EffiLLM: Efficient LLM Benchmarking Framework

EffiLLM is a tool for benchmarking LLM inference speed, memory usage, and efficiency in resource-constrained environments.

## Features

- **Performance Metrics**: Measure latency, throughput, and time-to-first-token
- **Memory Analysis**: Track RAM and VRAM usage during inference
- **Quantization Support**: Test models with different quantization methods (INT8, INT4)
- **Detailed Reports**: Generate visualizations and comparison reports

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA toolkit (for GPU benchmarking)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/EffiLLM.git
cd EffiLLM