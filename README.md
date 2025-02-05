# 🚀 Machine Learning Engineer Coding Challenge - BenA Submission
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6+-green.svg)](https://developer.nvidia.com/tensorrt)

> Vision Transformer implementation with quantization, hyperparameter optimization, and TensorRT acceleration.
> from the repo: ML-Engineer-Challenge

## 📋 Table of Contents
- [Overview](#overview)
- [Challenge Parts](#challenge-parts)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Performance Results](#performance-results)
- [Technical Implementation](#technical-implementation)
- [Deployment Guide](#deployment-guide)

## 🎯 Overview
Note:
There are a multitude of different ways we can implement this and improve it. For now I’ve submitted a basic version, we can discuss how this can be improved.
I provisioned a NVDIA GPU to run training on the dataset (runpod.io)
The Quantization script can be run a macos machine (however I do not recommend as the performance is significantly slower!)

This is nn implementation of Vision Transformer optimization, featuring:
- Model quantization with minimal accuracy loss
- Automated hyperparameter optimization
- TensorRT acceleration for production deployment

## 🎯 Challenge Parts
### Part 1: Model Quantization ([Report](part1/part1-submission-quantization-report.md))
- Implementation of dynamic quantization for ViT model
- Optimization of memory usage and inference speed
- Preservation of model accuracy within 0.3%
- 📓 [Deployment Implementation](part1/deployment-notebook.ipynb)

### Part 2: Hyperparameter Optimization ([Report](part2/part2-submissions-hyperparameter-report.md))
- Automated architecture search implementation
- Systematic hyperparameter tuning
- Cross-validation and performance analysis
- 📓 [Deployment Implementation](part2/deploy-hyperparameter.ipynb)

### Part 3: TensorRT Acceleration ([Report](part3/part3-submission.report.md))
- Model conversion to TensorRT format
- Implementation of FP16 precision optimization
- Dynamic batching and custom layer optimization
- 📓 [Deployment Implementation](part3/deploy-model-conversion.ipynb)

## ✨ Key Features
- **36.6%** faster inference time
- **63.2%** reduction in memory usage
- **74.4%** smaller model size
- Maintained accuracy within 0.3%

## 🛠️ Prerequisites
- NVIDIA GPU (Compute Capability 7.0+)
- 16GB+ GPU Memory
- 32GB System RAM
- CUDA 11.8+

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/LoQIStar/ML-Challenge-BenA-submission.git

# Install dependencies
pip install -r requirements.txt

# Create dummy models for testing (or download real models if available)
python utils/model_utils.py

# Run inference
python run_inference.py --model quantized --batch_size 32
```

## 📊 Model Availability
The repository supports three model variants:
- Original ViT model (`part1/models/vit_base_original.pth`)
- Quantized model (`part1/models/vit_base_quantized.pth`)
- TensorRT optimized model (`part3/models/vit_base_tensorrt.pth`)

For testing purposes, dummy models will be automatically created if the real models are not found.

## 📊 Performance Results

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Inference Time | 45.3ms | 28.7ms | 36.6% ⬇️ |
| Memory Usage | 342MB | 126MB | 63.2% ⬇️ |
| Model Size | 86.4MB | 22.1MB | 74.4% ⬇️ |

## 🔧 Technical Implementation
Three main optimization phases:

### 1. Model Quantization
- Dynamic quantization with INT8 precision
- Preserved floating-point activations
- Custom quantization for attention layers

### 2. Hyperparameter Optimization
- Automated architecture search
- Optuna-based optimization
- Cross-validation for robustness

### 3. TensorRT Acceleration
- FP16 precision optimization
- Dynamic batching support
- Custom layer optimization

## 📦 Deployment Guide
Recommended production setup:
- NVIDIA T4/L40 GPU instances
- Containerized deployment with Docker
- Regular model recalibration
- Comprehensive monitoring setup

## 🔄 Future Improvements
- Multi-GPU scaling support
- Custom CUDA kernel optimization
- Advanced quantization techniques
- Distributed training capabilities

## 📈 Benchmarking Results
![Performance Comparison: Original vs Optimized Model](docs/assets/performance_chart.png)


