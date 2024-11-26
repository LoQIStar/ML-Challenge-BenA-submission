# part2/training.py
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train(self, config):
        """
        config = {
            'learning_rate': float,
            'num_epochs': int,
            'weight_decay': float,
            'patience': int
        }
        """
        optimizer = Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            patience=config['patience'],
            factor=0.1,
            verbose=True
        )
        
        criterion = CrossEntropyLoss()
        best_accuracy = 0
        
        for epoch in range(config['num_epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            val_accuracy = self.validate()
            
            # Learning rate scheduling
            scheduler.step(val_accuracy)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            print(f'Epoch {epoch+1}: Train Acc = {100.*train_correct/train_total:.2f}%, Val Acc = {val_accuracy:.2f}%')
            
        return best_accuracy
    
    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total