import torch
import numpy as np
import torch.nn as nn
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        outputs: Model predictions
        targets: Ground truth labels
    
    Returns:
        Dictionary of metrics
    """
    # Convert outputs to predicted class
    _, preds = torch.max(outputs, 1)
    
    # Convert tensors to numpy arrays
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Compute metrics
    accuracy = accuracy_score(targets_np, preds_np)
    precision = precision_score(targets_np, preds_np, average='macro', zero_division=0)
    recall = recall_score(targets_np, preds_np, average='macro', zero_division=0)
    f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        outputs: Model predictions
        targets: Ground truth labels
    
    Returns:
        Confusion matrix as numpy array
    """
    _, preds = torch.max(outputs, 1)
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return confusion_matrix(targets_np, preds_np)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 