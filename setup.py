# EffiLLM/setup.py
from setuptools import setup, find_packages

setup(
    name="effillm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.25.0",
        "datasets>=2.8.0",
        "numpy>=1.23.0",
        "psutil>=5.9.0",
        "matplotlib>=3.6.0",
        "click>=8.1.0",
        "tqdm>=4.64.0",
        "pynvml>=11.4.1",  # For NVIDIA GPU monitoring
    ],
    entry_points={
        "console_scripts": [
            "effillm=effillm.cli.commands:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Efficient LLM benchmarking tool for resource-constrained environments",
    keywords="llm, benchmark, efficiency, quantization",
    url="https://github.com/yourusername/EffiLLM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)