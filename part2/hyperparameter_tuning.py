# part2/hyperparameter_tuning.py
import optuna
from utils.dataset import TinyImageNetDataset
from cnn_model import ConfigurableCNN
from training import ModelTrainer

class HyperparameterTuner:
    def __init__(self, train_loader, val_loader, device='cuda'):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def objective(self, trial):
        # Define hyperparameter search space
        config = {
            # Model architecture
            'num_conv_layers': trial.suggest_int('num_conv_layers', 3, 5),
            'conv_channels': [
                trial.suggest_int(f'conv_channels_{i}', 32, 256, step=32)
                for i in range(5)  # Max possible conv layers
            ],
            'conv_kernel_sizes': [
                trial.suggest_int(f'kernel_size_{i}', 3, 7, step=2)
                for i in range(5)
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'fc_layers': [
                trial.suggest_int(f'fc_size_{i}', 128, 1024, step=128)
                for i in range(2)  # 2 FC layers
            ],
            
            # Training parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'num_epochs': 30,  # Fixed for consistency
            'patience': trial.suggest_int('patience', 3, 7)
        }
        
        # Create model with trial hyperparameters
        model = ConfigurableCNN(config)
        
        # Train model
        trainer = ModelTrainer(model, self.train_loader, self.val_loader, self.device)
        best_accuracy = trainer.train(config)
        
        return best_accuracy
    
    def run_study(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        return study.best_params, study.best_value