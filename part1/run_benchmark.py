# part1/run_benchmark.py
# Add the parent directory to Python path if needed
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.dataset import TinyImageNetDataset
from quantization import ModelQuantizer
from benchmark import ModelBenchmark
import torch

def main():
    # Setup device
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU will be significantly slower.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
    
    # Setup dataset
    data_dir = Path(__file__).parent.parent / 'data' / 'tiny-imagenet-200'
    print(f"\nChecking dataset directory: {data_dir}")
    print(f"Directory exists: {data_dir.exists()}")
    
    # Use validation set for benchmarking
    dataset = TinyImageNetDataset(root_dir=str(data_dir), split='val')
    test_loader = dataset.get_loader(batch_size=32, shuffle=False, num_workers=2)
    
    # Debug first batch
    inputs, labels = next(iter(test_loader))
    print(f"\nInput tensor shape: {inputs.shape}")
    print(f"Labels tensor shape: {labels.shape}")
    print(f"Label values: {labels}")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
    
    # Initialize models
    print("\nInitializing models...")
    quantizer = ModelQuantizer(device=device)
    original_model = quantizer.original_model
    
    print("Converting model to quantized version...")
    quantized_model = quantizer.quantize_dynamic()
    
    # Setup benchmarker
    benchmarker = ModelBenchmark(device=device)
    
    # Run benchmarks
    print(f"\nBenchmarking original model ({device.upper()})...")
    original_metrics = benchmarker.measure_inference_time(original_model, test_loader)
    
    print("\nBenchmarking quantized model (CPU)...")
    quantized_metrics = benchmarker.measure_inference_time(quantized_model, test_loader)
    
    # Print results
    print("\nResults:")
    print(f"Original Model ({device.upper()}):")
    print(f"Mean inference time: {original_metrics['mean_inference_time']*1000:.2f}ms")
    print(f"Std inference time: {original_metrics['std_inference_time']*1000:.2f}ms")
    print(f"Accuracy: {original_metrics['accuracy']:.2f}%")
    
    print("\nQuantized Model (CPU):")
    print(f"Mean inference time: {quantized_metrics['mean_inference_time']*1000:.2f}ms")
    print(f"Std inference time: {quantized_metrics['std_inference_time']*1000:.2f}ms")
    print(f"Accuracy: {quantized_metrics['accuracy']:.2f}%")
    
    # Calculate relative performance
    speedup = original_metrics['mean_inference_time'] / quantized_metrics['mean_inference_time']
    print(f"\nQuantized model is {abs(1-speedup)*100:.1f}% {'faster' if speedup > 1 else 'slower'}")

if __name__ == "__main__":
    main()