import argparse
import torch
from pathlib import Path
import logging
import time
from utils.model_utils import DummyViT

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on optimized models')
    parser.add_argument('--model', type=str, choices=['original', 'quantized', 'tensorrt'],
                      default='quantized', help='Model version to use')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--input_size', type=int, default=224,
                      help='Input image size')
    return parser.parse_args()

def load_model(model_type):
    """Load the specified model version"""
    # Define model paths based on project structure
    if model_type == 'original':
        model_path = 'part1/models/vit_base_original.pth'
    elif model_type == 'quantized':
        model_path = 'part1/models/vit_base_quantized.pth'
    elif model_type == 'tensorrt':
        model_path = 'part3/models/vit_base_tensorrt.pth'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please ensure the model file exists in the correct location:\n"
            f"- Original model: part1/models/vit_base_original.pth\n"
            f"- Quantized model: part1/models/vit_base_quantized.pth\n"
            f"- TensorRT model: part3/models/vit_base_tensorrt.pth"
        )
    
    # Load model with appropriate method based on type
    if model_type == 'tensorrt':
        import tensorrt as trt
        return load_tensorrt_model(model_path)
    else:
        # Create model instance and load state dict
        model = DummyViT()
        model.load_state_dict(torch.load(model_path))
        return model

def load_tensorrt_model(model_path):
    """Load TensorRT model"""
    try:
        import tensorrt as trt
        with open(model_path, 'rb') as f:
            engine_str = f.read()
        
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_str)
        
        return engine
    except ImportError:
        raise ImportError("TensorRT not installed. Please install TensorRT to use TensorRT models.")

def run_inference(model, batch_size=32, input_size=224):
    """Run inference with the specified model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dummy input for testing
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times) * 1000  # Convert to ms
    return avg_time

def main():
    args = parse_args()
    
    try:
        # Check if models exist, if not create dummy ones
        if not all(Path(p).exists() for p in [
            'part1/models/vit_base_original.pth',
            'part1/models/vit_base_quantized.pth',
            'part3/models/vit_base_tensorrt.pth'
        ]):
            logging.info("Models not found. Creating dummy models for testing...")
            from utils.model_utils import setup_models
            setup_models(download=False)
        
        # Load model
        logging.info(f"Loading {args.model} model...")
        model = load_model(args.model)
        
        # Run inference
        logging.info(f"Running inference with batch size {args.batch_size}...")
        avg_time = run_inference(model, args.batch_size, args.input_size)
        
        logging.info(f"Average inference time: {avg_time:.2f}ms")
        logging.info(f"Throughput: {(args.batch_size * 1000 / avg_time):.2f} images/sec")
        
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 