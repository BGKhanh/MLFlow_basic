import os
import torch
import ray
from typing import Dict, Any, List, Optional
from ray.train import trainer as RayTrainer
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchConfig
import mlflow

from src.utils.config_utils import Config


def train_func(config: Dict[str, Any]) -> None:
    """
    Training function for distributed training with Ray.
    
    Args:
        config: Training configuration
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import ray.train.torch
    from torch.utils.data import Subset, DistributedSampler
    
    from ..data import create_dataloaders, CIFAR10Dataset, create_transforms
    from ..models import create_model
    from ..utils.metrics import compute_metrics, AverageMeter
    
    # Get worker information
    rank = ray.train.torch.get_rank()
    world_size = ray.train.torch.get_world_size()
    
    # Set the device
    device = ray.train.torch.get_device()
    
    # Prepare model for distributed training
    model = create_model(config)
    model = model.to(device)
    model = ray.train.torch.prepare_model(model)
    
    # Create transforms
    train_transform, val_transform = create_transforms(config)
    
    # Create datasets
    data_dir = config['dataset']['data_dir']
    train_dataset = CIFAR10Dataset(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=rank == 0  # Only download on the first worker
    )
    
    val_dataset = CIFAR10Dataset(
        root=data_dir,
        train=False,
        transform=val_transform,
        download=rank == 0  # Only download on the first worker
    )
    
    # Create data samplers for distributed training
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    optimizer_name = config['training']['optimizer'].lower()
    # Only AdamW is supported
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    num_epochs = config['training']['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Train the model
    for epoch in range(num_epochs):
        # Set epoch for samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training loop
        model.train()
        train_loss = AverageMeter()
        all_outputs = []
        all_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_size = inputs.size(0)
            train_loss.update(loss.item(), batch_size)
            
            # Store for metrics computation
            all_outputs.append(outputs.detach())
            all_targets.append(targets)
        
        # Compute training metrics
        if all_outputs and all_targets:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            train_metrics = compute_metrics(all_outputs, all_targets)
            train_metrics['loss'] = train_loss.avg
        else:
            train_metrics = {'loss': train_loss.avg, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Validation loop
        model.eval()
        val_loss = AverageMeter()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                batch_size = inputs.size(0)
                val_loss.update(loss.item(), batch_size)
                
                # Store for metrics computation
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Compute validation metrics
        if all_outputs and all_targets:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            val_metrics = compute_metrics(all_outputs, all_targets)
            val_metrics['loss'] = val_loss.avg
        else:
            val_metrics = {'loss': val_loss.avg, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        metrics = {
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch
        }
        
        # Report metrics for Ray
        ray.train.report(metrics)
        
        # Print progress on rank 0
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
    
    # Save model on rank 0
    if rank == 0:
        # Save the model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config
        }
        checkpoint_path = os.path.join(config['training']['checkpoint_dir'], "distributed_model.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)


def run_distributed_training(config_path_or_obj=None, tune_first: bool = False) -> None:
    """
    Run distributed training using Ray.
    
    Args:
        config_path_or_obj: Path to the configuration file or Config object
        tune_first: If True, run hyperparameter tuning before distributed training
    """
    # Load configuration
    if isinstance(config_path_or_obj, str):
        config = Config(config_path_or_obj)
    elif config_path_or_obj is None:
        config = Config('config/config.yaml')
    else:
        config = config_path_or_obj
        
    cfg = config.get_config()
    
    # Run hyperparameter tuning if requested
    if tune_first:
        from .tune import run_hyperparameter_tuning
        best_config = run_hyperparameter_tuning(config_path_or_obj)
        
        # Update configuration with best parameters
        for key, value in best_config.items():
            if key.startswith("training."):
                section, param = key.split(".", 1)
                cfg[section][param] = value
            elif key.startswith("model."):
                section, param = key.split(".", 1)
                cfg[section][param] = value
    
    # Get distributed training configuration
    distributed_config = cfg['distributed']
    num_workers = distributed_config.get('num_workers', 2)
    use_gpu = distributed_config.get('use_gpu', False)
    
    # Set up MLflow callback
    mlflow_config = cfg['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    experiment_name = f"{mlflow_config['experiment_name']}_distributed"
    mlflow.set_experiment(experiment_name)
    
    mlflow_callback = MLflowLoggerCallback(
        experiment_name=experiment_name,
        tracking_uri=mlflow_config['tracking_uri'],
        save_artifact=True
    )
    
    # Configure resources
    resources_per_worker = {"CPU": 1}
    if use_gpu:
        resources_per_worker["GPU"] = 1
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create a ray trainer
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker
    )
    
    # Configure failure handling
    run_config = RunConfig(
        callbacks=[mlflow_callback],
        failure_config=None  # Default failure handling
    )
    
    # Create PyTorch trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=TorchConfig(backend="gloo" if not use_gpu else "nccl")
    )
    
    # Run distributed training
    print(f"Starting distributed training with {num_workers} workers...")
    result = trainer.fit()
    
    # Print final results
    metrics = result.metrics
    print("Distributed training completed!")
    print(f"Final validation accuracy: {metrics.get('val_accuracy', 0):.4f}")
    
    # Load and convert best model
    checkpoint_path = os.path.join(cfg['training']['checkpoint_dir'], "distributed_model.pth")
    if os.path.exists(checkpoint_path):
        from ..models import create_model
        
        # Create model
        model = create_model(cfg)
        
        # Load state dict
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Log model to MLflow
        with mlflow.start_run(run_name=f"distributed_final_model"):
            mlflow.pytorch.log_model(model, "model")
            
            # Log parameters
            mlflow.log_params({
                "model_name": cfg['model']['name'],
                "batch_size": cfg['training']['batch_size'],
                "num_epochs": cfg['training']['num_epochs'],
                "learning_rate": cfg['training']['learning_rate'],
                "weight_decay": cfg['training']['weight_decay'],
                "optimizer": cfg['training']['optimizer'],
                "num_workers": num_workers,
                "use_gpu": use_gpu
            })
            
            # Log final metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
    
    # Return the final model
    return model 