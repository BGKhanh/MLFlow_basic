"""
Inference utilities for the CIFAR-10 training pipeline.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int], 
                         class_names: List[str],
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_top_predictions(probabilities: np.ndarray,
                        class_names: List[str],
                        top_k: int = 5,
                        save_path: Optional[str] = None) -> None:
    """
    Plot top-k predictions with probabilities.
    
    Args:
        probabilities: Array of class probabilities
        class_names: List of class names
        top_k: Number of top predictions to show
        save_path: Path to save the plot (optional)
    """
    # Get top-k indices
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probs = probabilities[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    plt.figure(figsize=(8, 6))
    bars = plt.barh(range(top_k), top_probs)
    plt.yticks(range(top_k), top_classes)
    plt.xlabel('Probability')
    plt.title(f'Top-{top_k} Predictions')
    plt.gca().invert_yaxis()
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Top predictions plot saved to {save_path}")
    
    plt.show()


def get_classification_report(y_true: List[int], 
                            y_pred: List[int], 
                            class_names: List[str]) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=class_names)


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size and parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'size_mb': size_mb,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }


def benchmark_inference_speed(model: torch.nn.Module,
                            input_shape: Tuple[int, ...],
                            device: str,
                            num_runs: int = 100) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (C, H, W)
        device: Device to run benchmark on
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'total_time_s': total_time,
        'num_runs': num_runs
    }