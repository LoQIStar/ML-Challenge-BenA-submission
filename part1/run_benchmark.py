# part1/run_benchmark.py
from utils.dataset import TinyImageNetDataset
from quantization import ModelQuantizer
from benchmark import ModelBenchmark

def main():
    # Setup dataset
    dataset = TinyImageNetDataset(root_dir='data/tiny-imagenet-200')
    test_loader = dataset.get_loader(batch_size=32, shuffle=False)
    
    # Initialize quantizer
    quantizer = ModelQuantizer()
    
    # Get models
    original_model = quantizer.original_model
    quantized_model = quantizer.quantize_dynamic()
    
    # Setup benchmarker
    benchmarker = ModelBenchmark()
    
    # Run benchmarks
    print("Benchmarking original model...")
    original_metrics = benchmarker.measure_inference_time(
        original_model, 
        test_loader
    )
    
    print("Benchmarking quantized model...")
    quantized_metrics = benchmarker.measure_inference_time(
        quantized_model, 
        test_loader
    )
    
    # Print results
    print("\nResults:")
    print("Original Model:")
    print(f"Mean inference time: {original_metrics['mean_inference_time']:.4f}s")
    print(f"Mean accuracy: {original_metrics['mean_accuracy']:.2f}%")
    
    print("\nQuantized Model:")
    print(f"Mean inference time: {quantized_metrics['mean_inference_time']:.4f}s")
    print(f"Mean accuracy: {quantized_metrics['mean_accuracy']:.2f}%")

if __name__ == "__main__":
    main()