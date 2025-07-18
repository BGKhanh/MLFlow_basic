import os
import torch
import numpy as np
from typing import Dict, Any, Callable
import ray
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import mlflow
from datetime import datetime

from src.utils.config_utils import Config
from src.data import create_dataloaders
from src.models import initialize_model
from src.training.trainer import Trainer


def train_with_config(config: Dict[str, Any], base_config: Config, checkpoint_dir: str = None) -> None:
    """
    Training function for Ray Tune. Focuses only on the training process.
    
    Args:
        config: Configuration dictionary from Ray Tune with hyperparameters to try
        base_config: Base configuration object
        checkpoint_dir: Directory to load checkpoints from
    """
    # Get the base configuration
    cfg = base_config.get_config()
    
    # Update base configuration with tuned parameters
    for key, value in config.items():
        if key.startswith("training."):
            section, param = key.split(".", 1)
            cfg[section][param] = value
        elif key.startswith("model."):
            section, param = key.split(".", 1)
            cfg[section][param] = value
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(cfg)

    # Create model
    model_name = cfg['model']['name']
    transfer_mode = cfg['model']['transfer_mode']
    model = initialize_model(model_name, num_classes, transfer_mode)
    
    # Initialize from checkpoint if available
    start_epoch = 0
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg
    )
    
    # Get number of epochs from the trial config
    # This is now explicitly passed from the tuning configuration
    num_epochs = config.get("trial_epochs", 1)
    
    # Train for specified number of epochs
    for epoch in range(start_epoch, num_epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics, _, _ = trainer.validate(epoch, 'val')
        
        # Report metrics to Ray Tune
        tune.report({
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'training_iteration': epoch + 1
        })
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict()
        }
        trial_dir = train.get_context().get_trial_dir()
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(trial_dir, "checkpoint.pth"))


def run_hyperparameter_tuning(config_path_or_obj=None) -> Dict[str, Any]:
    """
    Run hyperparameter tuning using Ray Tune. Handles all the setup, 
    configuration, and decision-making for the tuning process.
    
    Args:
        config_path_or_obj: Path to the base configuration file or Config object
    
    Returns:
        Best configuration found
    """
    # Load base configuration
    if isinstance(config_path_or_obj, str):
        base_config = Config(config_path_or_obj)
    elif config_path_or_obj is None:
        base_config = Config('config/config.yaml')
    else:
        base_config = config_path_or_obj
    cfg = base_config.get_config()
    tune_config = cfg['ray_tune']
    training_config = cfg['training']
    
    # Set up MLflow
    mlflow_config = cfg['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    experiment_name = f"{mlflow_config['experiment_name']}_tuning"
    mlflow.set_experiment(experiment_name)
    
    # Create parameter space
    search_space = {}
    
    # Add trial configuration parameters
    # Decide how many epochs each trial should run for
    epochs_per_trial = training_config.get('num_epochs', 5)  # Default to 3 epochs per trial if not specified
    search_space["trial_epochs"] = epochs_per_trial
    
    # Add learning rate
    if 'learning_rate' in tune_config['parameters']:
        lr_config = tune_config['parameters']['learning_rate']
        if 'min' in lr_config and 'max' in lr_config:
            search_space["training.learning_rate"] = tune.loguniform(
                lr_config['min'], lr_config['max']
            )
    
    # Add optimizer choice
    if 'optimizer' in tune_config['parameters']:
        opt_config = tune_config['parameters']['optimizer']
        if 'values' in opt_config:
            search_space["training.optimizer"] = tune.choice(opt_config['values'])
    
    # Add weight decay
    if 'weight_decay' in tune_config['parameters']:
        wd_config = tune_config['parameters']['weight_decay']
        if 'min' in wd_config and 'max' in wd_config:
            search_space["training.weight_decay"] = tune.loguniform(
                wd_config['min'], wd_config['max']
            )
    
    # Add batch size
    if 'batch_size' in tune_config['parameters']:
        bs_config = tune_config['parameters']['batch_size']
        if 'values' in bs_config:
            search_space["training.batch_size"] = tune.choice(bs_config['values'])
    
    # Add dropout rate
    if 'dropout_rate' in tune_config['parameters']:
        dr_config = tune_config['parameters']['dropout_rate']
        if 'min' in dr_config and 'max' in dr_config:
            search_space["model.dropout_rate"] = tune.uniform(
                dr_config['min'], dr_config['max']
            )
        elif 'values' in dr_config:
            search_space["model.dropout_rate"] = tune.choice(dr_config['values'])
    
    # Set up ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=epochs_per_trial,  # Maximum epochs per trial
        grace_period=4,  # Minimum epochs before pruning
        reduction_factor=2
    )
    
    # Set up Optuna search algorithm
    search_alg = OptunaSearch(
        metric="val_accuracy", 
        mode="max"
    )
    
    # Configure resources per trial
    resources_per_trial = tune_config.get('resources_per_trial', {})
    num_cpus = resources_per_trial.get('cpu', 1)
    num_gpus = resources_per_trial.get('gpu', 0)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Determine number of trials to run
    num_samples = tune_config.get('num_samples', 5)
    
    # Run hyperparameter optimization
    analysis = tune.run(
        tune.with_parameters(train_with_config, base_config=base_config),
        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=os.path.join(os.getcwd(), "ray_results"),
        name=experiment_name,
        verbose=1
    )
    
    # Get best trial
    best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max")
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["val_accuracy"]
    
    # Save best model to standardized location
    try:
        # Create standardized checkpoint directory
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints', 'models')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get best trial checkpoint path
        best_checkpoint_path = os.path.join(best_trial.checkpoint.path, "checkpoint.pth")
        
        if os.path.exists(best_checkpoint_path):
            # Load best checkpoint
            best_checkpoint = torch.load(best_checkpoint_path)
            
            # Add config and metrics to checkpoint  
            best_checkpoint['config'] = cfg
            best_checkpoint['best_metrics'] = best_trial.last_result
            
            # Standardized naming: {model_name}_tuned_{timestamp}.pth
            model_name = cfg['model']['name']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_name}_tuned_{timestamp}.pth"
            
            # Save to standardized location
            standardized_path = os.path.join(checkpoint_dir, filename)
            torch.save(best_checkpoint, standardized_path)
            print(f"Best tuned model saved to: {standardized_path}")
        else:
            print(f"Warning: Best checkpoint not found at {best_checkpoint_path}")
            
    except Exception as e:
        print(f"Warning: Could not save best tuned model: {e}")

    # Log best configuration to MLflow
    with mlflow.start_run(run_name=f"best_tuning_result"):
        # Log best parameters
        for param, value in best_config.items():
            if param != "trial_epochs":  # Don't log the internal trial_epochs parameter
                mlflow.log_param(param, value)
        
        # Log best metrics
        for metric, value in best_trial.last_result.items():
            mlflow.log_metric(metric, value)
        
        # Log a summary of the tuning results
        mlflow.log_text(
            analysis.best_result_df.to_string(), 
            "best_results_summary.txt"
        )
        
        # Generate a plot of the tuning results
        try:
            import matplotlib.pyplot as plt
            
            # Plot validation accuracy vs. learning rate
            plt.figure(figsize=(10, 6))
            results_df = analysis.results_df
            
            if "training.learning_rate" in results_df.columns:
                plt.scatter(
                    results_df["training.learning_rate"], 
                    results_df["val_accuracy"],
                    alpha=0.7
                )
                plt.xscale('log')
                plt.xlabel('Learning Rate')
                plt.ylabel('Validation Accuracy')
                plt.title('Validation Accuracy vs Learning Rate')
                plt.colorbar(label='Validation Accuracy')
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = "tuning_results_lr.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                # Log plot to MLflow
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)
        except Exception as e:
            print(f"Error generating tuning visualization: {e}")
    
    print(f"Best hyperparameters found: {best_config}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Update the base configuration with the best parameters (excluding trial_epochs)
    for key, value in best_config.items():
        if key != "trial_epochs" and (key.startswith("training.") or key.startswith("model.")):
            section, param = key.split(".", 1)
            base_config.update_config(section, param, value)
    
    # Save the updated configuration
    tuned_config_path = 'config/tuned_config.yaml'
    base_config.save_config(tuned_config_path)
    print(f"Tuned configuration saved to {tuned_config_path}")
    
    return best_config