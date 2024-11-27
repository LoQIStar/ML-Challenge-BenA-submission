# part1/benchmark.py
import torch
import time
from tqdm import tqdm

class ModelBenchmark:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def measure_inference_time(self, model, dataloader, num_iterations=100):
        model.eval()
        device = next(model.parameters()).device
        total_time = 0
        times = []
        correct = 0
        total = 0
        
        print(f"\nRunning {num_iterations} iterations on {device.type.upper()}")
        
        with torch.no_grad():
            for i in tqdm(range(num_iterations)):
                # Get a batch of data
                try:
                    inputs, labels = next(iter(dataloader))
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    inputs, labels = next(dataloader_iter)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Measure inference time
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Record time
                inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
                total_time += inference_time
                times.append(inference_time)
        
        mean_time = total_time / num_iterations
        std_time = torch.tensor(times).std().item()
        accuracy = 100 * correct / total
        
        return {
            'mean_inference_time': mean_time,
            'std_inference_time': std_time,
            'accuracy': accuracy
        }