import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets import CIFAR10, CIFAR100
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CIFARDataset(Dataset):
    """Dataset class for CIFAR-10 and CIFAR-100 datasets."""
    
    def __init__(self, 
                 root: str, 
                 dataset_name: str = "cifar10",
                 train: bool = True, 
                 transform: Optional[A.Compose] = None):
        """
        Initialize CIFAR dataset.
        
        Args:
            root: Root directory for datasets
            dataset_name: Name of the dataset ('cifar10' or 'cifar100')
            train: If True, loads the training dataset, else loads the test dataset
            transform: Albumentations transform pipeline
        """
        self.transform = transform
        self.train = train
        
        # Load the appropriate dataset
        if dataset_name.lower() == "cifar10":
            self.dataset = CIFAR10(root=root, train=train, download=True)
            self.num_classes = 10
            # CIFAR-10 mean and std values
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2470, 0.2435, 0.2616]
        elif dataset_name.lower() == "cifar100":
            self.dataset = CIFAR100(root=root, train=train, download=True)
            self.num_classes = 100
            # CIFAR-100 mean and std values
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        else:
            raise ValueError(f"Dataset {dataset_name} not supported. Use 'cifar10' or 'cifar100'.")
        
        logger.info(f"Loaded {dataset_name} dataset with {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image, label)
        """
        image, label = self.dataset[idx]
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def update_transform(self, transform: A.Compose) -> None:
        """
        Update the transform pipeline.
        
        Args:
            transform: New Albumentations transform pipeline
        """
        self.transform = transform


def get_model_img_size(model_name: str) -> int:
    """
    Get the recommended image size for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended image size
    """
    model_sizes = {
        "efficientnet": 480,
        "mobilenet": 232,
        "densenet": 256,
        "regnet": 384
    }
    
    # Check for partial matches
    for key in model_sizes:
        if key in model_name.lower():
            return model_sizes[key]
    
    # Default size if model not found
    logger.warning(f"Model {model_name} not found in size map, using default size 224")
    return 224


def create_transforms(config: Dict[str, Any], dataset_name: str = "cifar10", is_training: bool = True) -> A.Compose:
    """
    Create transforms pipeline for the dataset.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset ('cifar10' or 'cifar100')
        is_training: Whether to create transforms for training or validation/testing
    
    Returns:
        Albumentations transforms pipeline
    """
    aug_config = config['augmentation']
    
    # Get image size based on model if specified
    if 'model' in config and 'name' in config['model']:
        img_size = get_model_img_size(config['model']['name'])
    else:
        img_size = config['dataset']['img_size']
    
    # Set dataset specific normalization values
    if dataset_name.lower() == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset_name.lower() == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Use 'cifar10' or 'cifar100'.")
    
    transforms = []
    
    if is_training and aug_config.get('use_augmentation', True):
        # Spatial transformations
        if aug_config.get('random_resized_crop', False):
            transforms.append(A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)))
        elif aug_config.get('random_crop', False):
            # Add padding before random crop
            transforms.append(A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, value=0))
            transforms.append(A.RandomCrop(height=32, width=32))
            transforms.append(A.Resize(height=img_size, width=img_size))
        else:
            transforms.append(A.Resize(height=img_size, width=img_size))
        
        # Flips and rotations
        if aug_config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if aug_config.get('square_symmetry_rotations', False):
            transforms.append(A.RandomRotate90(p=0.5))
        elif aug_config.get('rotation_degrees', 0) > 0:
            max_degrees = aug_config.get('rotation_degrees', 15)
            transforms.append(A.Rotate(limit=max_degrees, p=0.5))
        
        # Color transformations
        if aug_config.get('color_jitter', False):
            transforms.append(A.ColorJitter(
                brightness=aug_config.get('brightness', 0.1),
                contrast=aug_config.get('contrast', 0.1),
                saturation=aug_config.get('saturation', 0.1),
                hue=aug_config.get('hue', 0.1),
                p=0.5
            ))
            
            # Add random gamma as well
            transforms.append(A.RandomGamma(gamma_limit=(80, 120), p=0.3))
        
        # Dropout transformations
        if aug_config.get('coarse_dropout', False):
            transforms.append(A.CoarseDropout(
                max_holes=aug_config.get('coarse_dropout_max_holes', 8),
                max_height=aug_config.get('coarse_dropout_max_height', 8),
                max_width=aug_config.get('coarse_dropout_max_width', 8),
                min_holes=aug_config.get('coarse_dropout_min_holes', 1),
                fill_value=tuple([int(x * 255) for x in mean]),
                p=0.5
            ))
        
        if aug_config.get('channel_dropout', False):
            transforms.append(A.ChannelDropout(p=0.5))
            transforms.append(A.ChannelShuffle(p=0.3))
        
        # Geometric transformations
        if aug_config.get('affine', False):
            transforms.append(A.Affine(
                scale=aug_config.get('affine_scale', [0.8, 1.2]),
                rotate=aug_config.get('affine_rotate', [-15, 15]),
                shear=[-5, 5],
                p=0.5
            ))
        
        # Robustness methods
        if aug_config.get('noise', False):
            transforms.extend([
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0)
                ], p=0.5)
            ])
        
        if aug_config.get('blur', False):
            transforms.extend([
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0)
                ], p=0.5)
            ])
            
        # Add grid distortion and elastic transform
        transforms.append(A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.2))
            
    else:
        # Simple resize for validation/testing
        transforms.append(A.Resize(height=img_size, width=img_size))
    
    # Always normalize and convert to tensor
    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return A.Compose(transforms)


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract configuration
    dataset_config = config['dataset']
    training_config = config['training']
    
    data_dir = dataset_config['data_dir']
    dataset_name = dataset_config.get('name', 'cifar10')
    batch_size = training_config.get('batch_size', 64)
    num_workers = training_config.get('num_workers', 4)
    seed = training_config.get('seed', 42)
    
    # Create transforms
    train_transform = create_transforms(config, dataset_name, is_training=True)
    val_transform = create_transforms(config, dataset_name, is_training=False)
    
    # Create datasets with appropriate transforms
    train_dataset = CIFARDataset(
        root=data_dir,
        dataset_name=dataset_name,
        train=True,
        transform=train_transform
    )
    
    val_dataset = CIFARDataset(
        root=data_dir,
        dataset_name=dataset_name,
        train=True,
        transform=val_transform
    )
    
    test_dataset = CIFARDataset(
        root=data_dir,
        dataset_name=dataset_name,
        train=False,
        transform=val_transform
    )
    
    num_classes = train_dataset.num_classes
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Split training data into training and validation sets (10% for validation)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    # Create indices for train/val split
    indices = torch.randperm(len(train_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    logger.info(f"Created datasets: Training({len(train_subset)}), Validation({len(val_subset)}), Test({len(test_dataset)})")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes