#!/usr/bin/env python3
"""
CIFAR-10 Inference Pipeline
Entry point for model inference and evaluation.
"""

import os
import argparse
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from src.utils import load_config
from src.data import create_dataloaders
from src.utils.model_loader import load_model, get_latest_model, list_models
from src.models import initialize_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_single_image(model: torch.nn.Module, 
                        image_tensor: torch.Tensor, 
                        device: str,
                        class_names: List[str]) -> Dict[str, Any]:
    """
    Predict single image.
    
    Args:
        model: Trained model
        image_tensor: Input image tensor (C, H, W)
        device: Device to run inference on
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Add batch dimension and move to device
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': probabilities[0].cpu().numpy()
    }


def evaluate_model(model: torch.nn.Module, 
                  test_loader: torch.utils.data.DataLoader, 
                  device: str,
                  class_names: List[str]) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    all_predictions = []
    all_labels = []
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == targets[i]:
                    class_correct[label] += 1
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    overall_accuracy = correct_predictions / total_samples
    
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[class_name] = class_correct[i] / class_total[i]
        else:
            class_accuracies[class_name] = 0.0
    
    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'class_accuracies': class_accuracies,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    logger.info(f"Evaluation completed. Overall accuracy: {overall_accuracy:.4f}")
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (if not provided, uses latest)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'list-models'], 
                        default='evaluate',
                        help='Inference mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    cfg = config.get_config()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Handle different modes
    if args.mode == 'list-models':
        models_info = list_models()
        if not models_info:
            logger.info("No trained models found.")
            return
        
        logger.info("Available trained models:")
        for model_info in models_info:
            logger.info(f"- {model_info['filename']} (Size: {model_info['file_size_mb']} MB)")
            if 'metrics' in model_info and model_info['metrics']:
                metrics = model_info['metrics']
                if 'test_accuracy' in metrics:
                    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        return
    
    # Load model checkpoint
    if args.checkpoint is None:
        checkpoint_path = get_latest_model()
        if checkpoint_path is None:
            logger.error("No trained model found. Please train a model first.")
            return
        logger.info(f"Using latest model: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
    
    # Load model
    try:
        model = load_model(checkpoint_path, device)
        logger.info(f"Model loaded successfully from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader, num_classes = create_dataloaders(cfg)
        logger.info(f"Data loaders created successfully")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # Define class names
    if cfg['dataset']['name'].lower() == 'cifar10':
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    else:  # CIFAR-100
        class_names = [f'class_{i}' for i in range(100)]  # Simplified for CIFAR-100
    
    # Run evaluation
    if args.mode == 'evaluate':
        results = evaluate_model(model, test_loader, device, class_names)
        
        # Display results
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
        logger.info(f"Total Samples: {results['total_samples']}")
        logger.info(f"Correct Predictions: {results['correct_predictions']}")
        
        logger.info("\nPer-class Accuracies:")
        for class_name, accuracy in results['class_accuracies'].items():
            logger.info(f"  {class_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()