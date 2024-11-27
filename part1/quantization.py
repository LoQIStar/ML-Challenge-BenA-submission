# part1/quantization.py
import torch
from timm import create_model
import copy
import numpy as np

class ModelQuantizer:
    def __init__(self, model_name="vit_base_patch16_224_dino", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.original_model = create_model(
            model_name,
            pretrained=True
        ).to(self.device)
        self.original_model.eval()
    
    def quantize_dynamic(self):
        """Simple 8-bit quantization"""
        model_cpu = copy.deepcopy(self.original_model).cpu()
        model_cpu.eval()
        
        def quantize_weights(tensor, num_bits=8):
            # Compute the scaling factor
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / (2 ** (num_bits - 1) - 1)
            
            # Quantize
            quantized = torch.round(tensor / scale)
            quantized = torch.clamp(quantized, -2**(num_bits-1), 2**(num_bits-1)-1)
            
            # Dequantize
            dequantized = quantized * scale
            return dequantized
        
        # Apply quantization to all parameters
        with torch.no_grad():
            for param in model_cpu.parameters():
                param.data = quantize_weights(param.data)
        
        return model_cpu