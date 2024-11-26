# part1/benchmark.py
import time
import torch
import numpy as np
from tqdm import tqdm

class ModelBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        
    def measure_inference_time(self, model, test_loader, num_runs=100):
        model = model.to(self.device)
        times = []
        accuracies = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_runs)):
                batch_times = []
                correct = 0
                total = 0
                
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    start_time = time.time()
                    outputs = model(inputs)
                    batch_time = time.time() - start_time
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    batch_times.append(batch_time)
                
                times.append(np.mean(batch_times))
                accuracies.append(100 * correct / total)
        
        return {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }