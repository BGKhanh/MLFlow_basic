import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Any, Tuple, Optional, Callable
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy

from src.utils.metrics import compute_metrics, AverageMeter, calculate_confusion_matrix


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any]):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.training_config = config['training']
        self.mlflow_config = config['mlflow']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Create checkpoint directory
        os.makedirs(self.training_config['checkpoint_dir'], exist_ok=True)
        
        # MLflow setup
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        
        # Initialize best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        # Lưu thời gian bắt đầu huấn luyện
        self.start_time = datetime.now()
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        lr = self.training_config['learning_rate']
        weight_decay = self.training_config['weight_decay']
        
        # Only AdamW is supported
        return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on configuration."""
        scheduler_name = self.training_config['scheduler'].lower()
        num_epochs = self.training_config['num_epochs']
        warmup_epochs = self.training_config.get('warmup_epochs', 0)
        
        if warmup_epochs > 0:
            # Warmup scheduler
            warmup_scheduler = LinearLR(self.optimizer, 
                                        start_factor=0.1, 
                                        end_factor=1.0, 
                                        total_iters=warmup_epochs)
            
            if scheduler_name == 'cosine':
                main_scheduler = CosineAnnealingLR(self.optimizer, 
                                                   T_max=num_epochs - warmup_epochs)
                return SequentialLR(self.optimizer,
                                    schedulers=[warmup_scheduler, main_scheduler],
                                    milestones=[warmup_epochs])
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        else:
            # Default to constant LR if not specified or not implemented
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)
    
    def _save_final_model(self, metrics: Dict[str, float]) -> str:
        """
        Save final model checkpoint with timestamp.
        
        Args:
            metrics: Validation metrics
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.training_config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Tạo tên file với tên mô hình và thời gian huấn luyện
        model_name = self.config['model']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{model_name}_{timestamp}.pth"
            
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def _log_to_mlflow(self, metrics: Dict[str, float], step: int, phase: str) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (epoch)
            phase: Training phase ('train', 'val', or 'test')
        """
        for name, value in metrics.items():
            mlflow.log_metric(f"{phase}_{name}", value, step=step)
    
    def _log_confusion_matrix(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Log confusion matrix to MLflow chỉ một lần sau khi hoàn thành test.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
        """
        cm = calculate_confusion_matrix(outputs, targets)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Final Confusion Matrix - Test Set')
        plt.colorbar()
        plt.tight_layout()
        
        # Save plot to temporary file
        model_name = self.config['model']['name']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cm_filename = f'confusion_matrix_{model_name}_{timestamp}.png'
        plt.savefig(cm_filename)
        plt.close()
        
        # Log artifact theo loại tracking URI
        tracking_uri = self.mlflow_config['tracking_uri']
        try:
            if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                # Sử dụng HTTP server 
                mlflow.log_artifact(cm_filename)
            else:
                # Sử dụng filesystem trực tiếp qua mlflow API
                # Thay vì tạo thủ công, để mlflow tự quản lý cấu trúc thư mục
                mlflow.log_artifact(cm_filename)
                
                # Nếu muốn thêm thông tin về việc đã log artifact
                if os.path.exists(cm_filename):
                    mlflow.log_metrics({"has_final_confusion_matrix": 1})
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")
            # Trong trường hợp tracking_uri là filesystem, vẫn ghi log thay vì dừng training
            if not tracking_uri.startswith("http://") and not tracking_uri.startswith("https://"):
                print(f"Consider running MLflow UI with: mlflow ui --backend-store-uri {tracking_uri}")
        
        # Remove temporary file
        os.remove(cm_filename)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch.
        
        Args:
            epoch: Current epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        
        # Store all outputs and targets for metric computation
        all_outputs = []
        all_targets = []
        
        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.training_config['num_epochs']} [Train]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            # Store outputs and targets for overall metrics computation
            all_outputs.append(outputs.detach())
            all_targets.append(targets)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss_meter.avg})
        
        # Compute overall metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_outputs, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics
    
    def validate(self, epoch: int, phase: str = 'val') -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch
            phase: Validation phase ('val' or 'test')
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        
        # Store all outputs and targets for metric computation
        all_outputs = []
        all_targets = []
        
        loader = self.val_loader if phase == 'val' else self.test_loader
        
        # Use tqdm for progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.training_config['num_epochs']} [{phase.capitalize()}]")
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                loss_meter.update(loss.item(), batch_size)
                
                # Store outputs and targets for overall metrics computation
                all_outputs.append(outputs)
                all_targets.append(targets)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss_meter.avg})
        
        # Compute overall metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = compute_metrics(all_outputs, all_targets)
        metrics['loss'] = loss_meter.avg
        
        return metrics, all_outputs, all_targets
    
    def train(self) -> nn.Module:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Trained model
        """
        num_epochs = self.training_config['num_epochs']
        
        # Start MLflow experiment
        with mlflow.start_run(run_name=f"{self.mlflow_config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "model_name": self.config['model']['name'],
                "batch_size": self.training_config['batch_size'],
                "num_epochs": num_epochs,
                "learning_rate": self.training_config['learning_rate'],
                "weight_decay": self.training_config['weight_decay'],
                "optimizer": self.training_config['optimizer'],
                "scheduler": self.training_config['scheduler'],
                "dataset": self.config['dataset']['name'],
                "img_size": self.config['dataset']['img_size'],
                "transfer_mode": self.config['model']['transfer_mode']
            })
            
            # Log the entire config as a YAML file
            import yaml
            with open('run_config.yaml', 'w') as f:
                yaml.dump(self.config, f)
            try:
                tracking_uri = self.mlflow_config['tracking_uri']
                if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                    mlflow.log_artifact('run_config.yaml')
                else:
                    # Sử dụng filesystem trực tiếp qua mlflow API
                    mlflow.log_artifact('run_config.yaml')
            except Exception as e:
                print(f"Warning: Could not log config file: {e}")
                if not tracking_uri.startswith("http://") and not tracking_uri.startswith("https://"):
                    print(f"Consider running MLflow UI with: mlflow ui --backend-store-uri {tracking_uri}")
            os.remove('run_config.yaml')
            
            # Print training information
            print(f"Starting training on {self.device}")
            print(f"Model: {self.config['model']['name']}")
            print(f"Dataset: {self.config['dataset']['name']}")
            print(f"Batch size: {self.training_config['batch_size']}")
            print(f"Learning rate: {self.training_config['learning_rate']}")
            
            early_stopping_patience = self.training_config.get('early_stopping_patience', float('inf'))
            early_stopping_counter = 0
            
            # Training loop
            for epoch in range(num_epochs):
                # Train one epoch
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics, val_outputs, val_targets = self.validate(epoch, 'val')
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                    mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=epoch)
                
                # Log metrics to MLflow
                self._log_to_mlflow(train_metrics, epoch, 'train')
                self._log_to_mlflow(val_metrics, epoch, 'val')
                
                # Print metrics
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                
                # Check if this is the best model so far
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                    early_stopping_counter = 0
                    
                    # Lưu trạng thái model tốt nhất trong bộ nhớ
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    print(f"New best model with validation accuracy: {self.best_val_acc:.4f}")
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Evaluate on the test set using the best model
            print("Evaluating best model on test set...")
            
            # Load the best model weights
            if hasattr(self, 'best_model_state'):
                self.model.load_state_dict(self.best_model_state)
            
            # Run test evaluation
            test_metrics, test_outputs, test_targets = self.validate(num_epochs, 'test')
            
            # Log test metrics
            self._log_to_mlflow(test_metrics, num_epochs, 'test')
            
            # Tính và lưu confusion matrix chỉ một lần duy nhất ở đây
            self._log_confusion_matrix(test_outputs, test_targets)
            
            # Print test metrics
            print(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
            
            # Lưu model cuối cùng với định dạng tên mới
            final_model_path = self._save_final_model(test_metrics)
            print(f"Final model saved at: {final_model_path}")
            
            # Log model checkpoint cuối cùng vào MLflow
            try:
                tracking_uri = self.mlflow_config['tracking_uri']
                if tracking_uri.startswith("http://") or tracking_uri.startswith("https://"):
                    mlflow.log_artifact(final_model_path)
                else:
                    mlflow.log_metric("final_model_saved", 1)
            except Exception as e:
                print(f"Warning: Could not log final model checkpoint: {e}")
                
            # Log the best model
            if self.mlflow_config.get('log_model', True):
                mlflow.pytorch.log_model(self.model, "model")
            
            # Register the model in MLflow Model Registry if configured
            if self.mlflow_config.get('register_model', False):
                mlflow.pytorch.log_model(
                    self.model, 
                    "model", 
                    registered_model_name=self.mlflow_config['experiment_name']
                )
        
        return self.model 