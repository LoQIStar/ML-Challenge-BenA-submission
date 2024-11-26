# part3/onnx_export.py
import torch
import onnx
import onnxruntime
import numpy as np
from typing import Tuple, Dict

class ONNXExporter:
    def __init__(self, model_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        self.model_path = model_path
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_pytorch_model(self):
        """Load the PyTorch model from Part 1"""
        model = torch.load(self.model_path, map_location=self.device)
        model.eval()
        return model
    
    def export_to_onnx(self, output_path: str) -> str:
        """Export PyTorch model to ONNX format"""
        model = self.load_pytorch_model()
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        return output_path
    
    def verify_onnx_output(self, onnx_path: str) -> Dict[str, np.ndarray]:
        """Compare PyTorch and ONNX model outputs"""
        # PyTorch inference
        model = self.load_pytorch_model()
        dummy_input = torch.randn(self.input_shape, device=self.device)
        with torch.no_grad():
            torch_output = model(dummy_input).cpu().numpy()
        
        # ONNX Runtime inference
        ort_session = onnxruntime.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        ort_inputs = {
            'input': dummy_input.cpu().numpy()
        }
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        np.testing.assert_allclose(
            torch_output, 
            ort_output, 
            rtol=1e-03, 
            atol=1e-05
        )
        
        return {
            'pytorch_output': torch_output,
            'onnx_output': ort_output
        }