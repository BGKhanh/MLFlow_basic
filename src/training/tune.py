import os
import torch
import numpy as np
from typing import Dict, Any, Callable
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import mlflow

from src.utils.config_utils import Config
from src.data import create_dataloaders
from src.models import initialize_model
from src.training.trainer import Trainer


def train_with_config(config: Dict[str, Any], base_config: Config, checkpoint_dir: str = None) -> None:
    """
    Training function for Ray Tune.
    
    Args:
        config: Configuration dictionary from Ray Tune
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


    model_name = cfg['model']['name']
    transfer_mode = cfg['model']['transfer_mode']
    # Create model
    model = initialize_model(model_name, num_classes, transfer_mode)
    
    # Initialize from checkpoint if available
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg
    )
    
    # Train for a single epoch
    train_metrics = trainer.train_epoch(0)
    
    # Validate
    val_metrics, _, _ = trainer.validate(0, 'val')
    
    metrics_to_report = {
        'train_loss': train_metrics['loss'],
        'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy']
    }
    # Report metrics to Ray Tune
    tune.report(metrics_to_report)
    
    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    checkpoint_dir = train.get_context().get_trial_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(checkpoint_dir, "checkpoint.pth"))


def run_hyperparameter_tuning(config_path_or_obj=None) -> Dict[str, Any]:
    """
    Run hyperparameter tuning using Ray Tune.
    
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
    
    # Set up MLflow
    mlflow_config = cfg['mlflow']
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    experiment_name = f"{mlflow_config['experiment_name']}_tuning"
    mlflow.set_experiment(experiment_name)
    
    # Create parameter space
    search_space = {}
    
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
            search_space["training.optimizer"] = tune.choice(["adamw"])
    
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
        if 'values' in dr_config:
            search_space["model.dropout_rate"] = tune.choice(dr_config['values'])
    
    # Set up ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=5,  # Maximum epochs per trial
        grace_period=1,  # Minimum epochs before pruning
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
    
    # Run hyperparameter optimization
    analysis = tune.run(
        tune.with_parameters(train_with_config, base_config=base_config),
        resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
        config=search_space,
        num_samples=tune_config.get('num_samples', 10),
        scheduler=scheduler,
        search_alg=search_alg,
        storage_path=os.path.join(os.getcwd(), "ray_results"),
        name=experiment_name,
        verbose=1
    )
    
    # Get best trial
    best_trial = analysis.best_trial
    best_config = best_trial.config
    best_accuracy = best_trial.last_result["val_accuracy"]
    
    # Log best configuration to MLflow
    with mlflow.start_run(run_name=f"best_tuning_result"):
        # Log best parameters
        for param, value in best_config.items():
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
    
    # Update the base configuration with the best parameters
    for key, value in best_config.items():
        if key.startswith("training."):
            section, param = key.split(".", 1)
            base_config.update_config(section, param, value)
        elif key.startswith("model."):
            section, param = key.split(".", 1)
            base_config.update_config(section, param, value)
    
    # Save the updated configuration
    tuned_config_path = 'config/tuned_config.yaml'
    base_config.save_config(tuned_config_path)
    print(f"Tuned configuration saved to {tuned_config_path}")
    
    return best_config 