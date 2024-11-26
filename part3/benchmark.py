# part3/benchmark.py
import time
import numpy as np
import torch
import tensorrt as trt
from typing import Dict, List

class InferenceBenchmark:
    def __init__(self):
        self.results = {}
        
    def benchmark_pytorch(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """Benchmark PyTorch model inference"""
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        
        self.results['pytorch'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'median': np.median(times)
        }
        return self.results['pytorch']
    
    def benchmark_tensorrt(
        self,
        engine: trt.ICudaEngine,
        input_shape: tuple,
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Dict[str, float]:
        """Benchmark TensorRT engine inference"""
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = self.allocate_buffers(engine)
        
        # Prepare input data
        input_data = np.random.random(input_shape).astype(np.float32)
        np.copyto(inputs[0]['host'], input_data.ravel())
        
        # Warmup
        for _ in range(warmup):
            self._infer_tensorrt(context, bindings, inputs, outputs, stream)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self._infer_tensorrt(context, bindings, inputs, outputs, stream)
            times.append(time.perf_counter() - start)
        
        self.results['tensorrt'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'median': np.median(times)
        }
        return self.results['tensorrt']
    
    @staticmethod
    def _infer_tensorrt(context, bindings, inputs, outputs, stream):
        """Helper method for TensorRT inference"""
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()