# part2/cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigurableCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        config = {
            'num_conv_layers': int,
            'conv_channels': list[int],
            'conv_kernel_sizes': list[int],
            'dropout_rate': float,
            'fc_layers': list[int]
        }
        """
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB input
        
        # Configure convolutional layers
        for i in range(config['num_conv_layers']):
            out_channels = config['conv_channels'][i]
            kernel_size = config['conv_kernel_sizes'][i]
            
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ))
            in_channels = out_channels
        
        # Calculate final conv output size
        self.flatten_size = config['conv_channels'][-1] * (224 // (2**config['num_conv_layers']))**2
        
        # Configure fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in = self.flatten_size
        
        for fc_size in config['fc_layers']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_in, fc_size),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate'])
            ))
            fc_in = fc_size
            
        # Final classification layer
        self.classifier = nn.Linear(fc_in, 200)  # TinyImageNet has 200 classes
    
    def forward(self, x):
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            
        # Classification
        return self.classifier(x)