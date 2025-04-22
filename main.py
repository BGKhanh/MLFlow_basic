import os
import argparse
from typing import Dict, Any

from src.utils import load_config, Config
from src.data import create_dataloaders
from src.models import initialize_model
from src.training import Trainer, run_hyperparameter_tuning, run_distributed_training


def train(config_path_or_obj: str or Config) -> None:
    """
    Run the standard training pipeline.
    
    Args:
        config_path_or_obj: Path to the configuration file or Config object
    """
    # Load configuration
    if isinstance(config_path_or_obj, str):
        config = load_config(config_path_or_obj)
    else:
        config = config_path_or_obj
    
    cfg = config.get_config()
    
    # Create data loaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(cfg)
    
    
    # Lấy các tham số model
    model_name = cfg['model']['name']
    transfer_mode = cfg['model']['transfer_mode']
    
    # Create model
    model = initialize_model(model_name, num_classes, transfer_mode)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg
    )
    
    # Train model
    trained_model = trainer.train()
    
    print("Training completed successfully!")
    return trained_model


def main():
    """Main entry point for the training pipeline."""
    parser = argparse.ArgumentParser(description='CIFAR-10 Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--distributed', action='store_true',
                        help='Run distributed training')
    
    args = parser.parse_args()
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Run the selected pipeline
    if args.tune and args.distributed:
        print("Running hyperparameter tuning followed by distributed training...")
        best_config = run_hyperparameter_tuning(config)
        print(f"Best hyperparameters: {best_config}")
        print("Now starting distributed training with tuned parameters...")
        run_distributed_training(config, tune_first=True)
    elif args.tune:
        print("Running hyperparameter tuning...")
        best_config = run_hyperparameter_tuning(config)
        print(f"Best hyperparameters: {best_config}")
    elif args.distributed:
        print("Running distributed training...")
        run_distributed_training(config, tune_first=False)
    else:
        print("Running standard training pipeline...")
        train(config)


if __name__ == "__main__":
    main() 