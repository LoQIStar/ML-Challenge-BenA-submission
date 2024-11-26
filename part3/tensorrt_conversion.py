# part3/tensorrt_conversion.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from typing import Union, Dict

class TensorRTConverter:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        
    def build_engine(
        self,
        fp16_mode: bool = True,
        int8_mode: bool = False,
        max_workspace_size: int = 1 << 30,
        max_batch_size: int = 32
    ) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX model"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            
        if int8_mode and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # Would need calibration data for INT8
        
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",                          # Input tensor name
            (1, 3, 224, 224),                # Min shape
            (max_batch_size, 3, 224, 224),   # Optimal shape
            (max_batch_size * 2, 3, 224, 224) # Max shape
        )
        config.add_optimization_profile(profile)
        
        # Build engine
        self.engine = builder.build_engine(network, config)
        return self.engine
    
    def save_engine(self, engine_path: str):
        """Save TensorRT engine to file"""
        if self.engine is None:
            raise RuntimeError("Engine not built yet!")
            
        with open(engine_path, 'wb') as f:
            f.write(self.engine.serialize())
            
    def load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(self.logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        return self.engine
    
    def allocate_buffers(self, engine: trt.ICudaEngine, batch_size: int = 1):
        """Allocate device buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                
        return inputs, outputs, bindings, stream