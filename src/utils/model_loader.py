import os
import torch
import glob
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from src.models.model import initialize_model
import logging

logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str, device: str = 'auto') -> torch.nn.Module:
    """
    Load model from checkpoint path.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on ('auto', 'cpu', 'cuda')
        
    Returns:
        Loaded PyTorch model ready for inference
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    # Handle old checkpoint format without config
    if 'config' not in checkpoint:
        logger.warning(f"No config found in checkpoint. Using default config for inference.")
        
        # Use params.json if available in same directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        params_file = os.path.join(checkpoint_dir, 'params.json')
        
        if os.path.exists(params_file):
            import json
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            config_dict = {
                'model': {
                    'name': params.get('model.name', 'mobilenet'),
                    'transfer_mode': params.get('transfer_mode', 'classifier_only')
                },
                'dataset': {
                    'num_classes': params.get('dataset.num_classes', 10)
                }
            }
            logger.info(f"Loaded config from params.json: {params_file}")
        else:
            # Fallback to default config
            config_dict = {
                'model': {
                    'name': 'mobilenet',
                    'transfer_mode': 'classifier_only'
                },
                'dataset': {
                    'num_classes': 10
                }
            }
            logger.warning("Using fallback default config")
    else:
        config_dict = checkpoint['config']
        
    config = config_dict
    model_name = config['model']['name']
    num_classes = config['dataset']['num_classes']
    transfer_mode = config['model'].get('transfer_mode', 'classifier_only')
    
    # Initialize model
    model = initialize_model(model_name, num_classes, transfer_mode)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def get_latest_model(training_type: Optional[str] = None, 
                    checkpoints_dir: str = 'checkpoints/models') -> Optional[str]:
    """
    Get path to the latest model checkpoint.
    
    Args:
        training_type: Filter by training type ('trainer', 'distributed', 'tuned')
        checkpoints_dir: Directory containing model checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Build search pattern
    if training_type:
        pattern = os.path.join(checkpoints_dir, f"*_{training_type}_*.pth")
    else:
        pattern = os.path.join(checkpoints_dir, "*.pth")
    
    # Find all matching files
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (latest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    return checkpoint_files[0]


def list_models(checkpoints_dir: str = 'checkpoints/models') -> List[Dict[str, Any]]:
    """
    List all available model checkpoints with metadata.
    
    Args:
        checkpoints_dir: Directory containing model checkpoints
        
    Returns:
        List of dictionaries with model information
    """
    if not os.path.exists(checkpoints_dir):
        return []
    
    pattern = os.path.join(checkpoints_dir, "*.pth")
    checkpoint_files = glob.glob(pattern)
    
    models_info = []
    for checkpoint_path in checkpoint_files:
        try:
            info = get_model_info(checkpoint_path)
            models_info.append(info)
        except Exception as e:
            # Skip corrupted checkpoints
            continue
    
    # Sort by timestamp (newest first)
    models_info.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return models_info


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get metadata information from a model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing model metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Extract info from filename
    filename = os.path.basename(checkpoint_path)
    parts = filename.replace('.pth', '').split('_')
    
    # Handle standardized format: {model_name}_{training_type}_{timestamp}.pth
    if len(parts) >= 3:
        model_name = parts[0]
        training_type = parts[1] 
        timestamp_str = '_'.join(parts[2:])
    # Handle legacy/non-standard format (e.g., checkpoint.pth)
    else:
        model_name = "unknown"
        training_type = "legacy"
        timestamp_str = "unknown"
    
    # Try to load checkpoint for detailed info
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint.get('config', {})
        metrics = checkpoint.get('metrics', {})
        
        # Try to extract model info from config if filename parsing failed
        if model_name == "unknown" and config:
            model_name = config.get('model', {}).get('name', 'unknown')
        
        return {
            'path': checkpoint_path,
            'filename': filename,
            'model_name': model_name,
            'training_type': training_type,
            'timestamp': timestamp_str,
            'config': config,
            'metrics': metrics,
            'file_size_mb': round(os.path.getsize(checkpoint_path) / (1024*1024), 2)
        }
    except Exception as e:
        # Return basic info if checkpoint can't be loaded
        return {
            'path': checkpoint_path,
            'filename': filename,
            'model_name': model_name,
            'training_type': training_type,
            'timestamp': timestamp_str,
            'config': {},
            'metrics': {},
            'file_size_mb': round(os.path.getsize(checkpoint_path) / (1024*1024), 2),
            'error': str(e)
        }