# EffiLLM/effillm/core/workloads.py
from typing import List, Dict, Any
import torch

class BenchmarkWorkload:
    """Defines a standard workload for benchmarking LLMs."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def get_inputs(self, tokenizer, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        """Return tokenized inputs for this workload."""
        raise NotImplementedError
        
    def get_generation_params(self) -> Dict[str, Any]:
        """Return parameters for generation."""
        raise NotImplementedError

class ChatWorkload(BenchmarkWorkload):
    """Benchmark workload simulating chat interactions."""
    
    def __init__(self):
        super().__init__(
            name="chat",
            description="Simulates chat interactions with varying complexity"
        )
        self.prompts = [
            "Hello, how are you today?",
            "Can you explain quantum computing in simple terms?",
            "Write a short story about a robot learning to paint.",
            "What are the main differences between Python and JavaScript?"
        ]
        
    def get_inputs(self, tokenizer, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        # Select prompts based on batch size
        selected_prompts = self.prompts[:batch_size]
        if len(selected_prompts) < batch_size:
            # Repeat prompts to fill the batch
            selected_prompts = selected_prompts * (batch_size // len(selected_prompts) + 1)
            selected_prompts = selected_prompts[:batch_size]
            
        # Tokenize
        inputs = tokenizer(selected_prompts, padding=True, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
        
    def get_generation_params(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9
        }

class CodeGenerationWorkload(BenchmarkWorkload):
    """Benchmark workload for code generation tasks."""
    
    def __init__(self):
        super().__init__(
            name="code",
            description="Code generation tasks of varying complexity"
        )
        self.prompts = [
            "Write a Python function to calculate the Fibonacci sequence.",
            "Create a JavaScript function that sorts an array of objects by a specified property.",
            "Write a SQL query to find the top 5 customers who placed the most orders.",
            "Create a simple React component that displays a list of items."
        ]
        
    def get_inputs(self, tokenizer, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        # Similar implementation to ChatWorkload
        selected_prompts = self.prompts[:batch_size]
        if len(selected_prompts) < batch_size:
            selected_prompts = selected_prompts * (batch_size // len(selected_prompts) + 1)
            selected_prompts = selected_prompts[:batch_size]
            
        inputs = tokenizer(selected_prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
        
    def get_generation_params(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95
        }

class SummaryWorkload(BenchmarkWorkload):
    """Benchmark workload for text summarization tasks."""
    
    def __init__(self):
        super().__init__(
            name="summary",
            description="Text summarization of long contexts"
        )
        self.texts = [
            "Long article about climate change...", # Replace with actual longer texts
            "Scientific paper on neural networks...",
            "News article about global politics..."
        ]
        
    def get_inputs(self, tokenizer, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        # Similar pattern to other workloads
        selected_texts = self.texts[:batch_size]
        if len(selected_texts) < batch_size:
            selected_texts = selected_texts * (batch_size // len(selected_texts) + 1)
            selected_texts = selected_texts[:batch_size]
            
        # Add summarization instruction
        prompts = [f"Summarize the following text:\n\n{text}\n\nSummary:" for text in selected_texts]
        
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
        
    def get_generation_params(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9
        }

def get_available_workloads() -> Dict[str, BenchmarkWorkload]:
    """Get all available benchmark workloads."""
    return {
        "chat": ChatWorkload(),
        "code": CodeGenerationWorkload(),
        "summary": SummaryWorkload()
    }