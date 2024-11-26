# part3/run_conversion.py
import os
from onnx_export import ONNXExporter
from tensorrt_conversion import TensorRTConverter
from benchmark import InferenceBenchmark

def main():
    # Paths
    model_path = '../part1/best_model.pth'
    onnx_path = 'model.onnx'
    engine_path = 'model.trt'
    
    # Export to ONNX
    print("Exporting to ONNX...")
    exporter = ONNXExporter(model_path)
    onnx_path = exporter.export_to_onnx(onnx_path)
    
    # Verify ONNX export
    print("Verifying ONNX export...")
    exporter.verify_onnx_output(onnx_path)
    
    # Convert to TensorRT
    print("Converting to TensorRT...")
    converter = TensorRTConverter(onnx_path)
    engine = converter.build_engine(fp16_mode=True)
    converter.save_engine(engine_path)
    
    # Benchmark
    print("Running benchmarks...")
    benchmark = InferenceBenchmark()
    
    # PyTorch benchmark
    pytorch_model = exporter.load_pytorch_model()
    pytorch_results = benchmark.benchmark_pytorch(
        pytorch_model,
        (1, 3, 224, 224)
    )
    
    # TensorRT benchmark
    engine = converter.load_engine(engine_path)
    tensorrt_results = benchmark.benchmark_tensorrt(
        engine,
        (1, 3, 224, 224)
    )
    
    # Print results
    print("\nBenchmark Results:")
    print("\nPyTorch Model:")
    print(f"Mean inference time: {pytorch_results['mean']*1000:.2f} ms")
    print(f"Std deviation: {pytorch_results['std']*1000:.2f} ms")
    
    print("\nTensorRT Engine:")
    print(f"Mean inference time: {tensorrt_results['mean']*1000:.2f} ms")
    print(f"Std deviation: {tensorrt_results['std']*1000:.2f} ms")
    
    speedup = pytorch_results['mean'] / tensorrt_results['mean']
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()