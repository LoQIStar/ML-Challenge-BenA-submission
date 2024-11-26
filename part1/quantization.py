# part1/quantization.py
import torch
import torch.quantization
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

class ModelQuantizer:
    def __init__(self, model_name="facebook/dino-vitb16"):
        self.original_model = torch.hub.load(
            'facebookresearch/dino:main', 
            model_name
        )
        self.original_model.eval()
        
    def quantize_dynamic(self):
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.original_model,
            {torch.nn.Linear},  # Quantize only linear layers
            dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_static(self, calibration_loader):
        # Static quantization
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}
        
        model_to_quantize = self.original_model
        
        # Prepare
        model_prepared = prepare_fx(model_to_quantize, qconfig_dict)
        
        # Calibrate
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)
                
        # Convert
        quantized_model = convert_fx(model_prepared)
        
        return quantized_model