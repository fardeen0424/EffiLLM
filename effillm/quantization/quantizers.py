# EffiLLM/effillm/quantization/quantizers.py
from typing import Dict, Any
import torch
import logging

logger = logging.getLogger(__name__)

def apply_quantization(model_id: str, quantization_config: Dict[str, Any], device: str):
    """Apply the specified quantization method to the model."""
    bits = quantization_config.get("bits", 8)
    method = quantization_config.get("method", "bitsandbytes")
    
    logger.info(f"Applying {bits}-bit quantization using {method}")
    
    if method == "bitsandbytes":
        try:
            import bitsandbytes as bnb
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            
            if bits == 8:
                quantization = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            elif bits == 4:
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                raise ValueError(f"Unsupported bit depth for bitsandbytes: {bits}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization,
                device_map=device
            )
            return model
            
        except ImportError:
            logger.error("bitsandbytes not installed. Install with: pip install bitsandbytes")
            raise
            
    elif method == "gptq":
        try:
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            return model
            
        except Exception as e:
            logger.error(f"Error loading GPTQ model: {e}")
            raise
            
    else:
        raise ValueError(f"Unsupported quantization method: {method}")