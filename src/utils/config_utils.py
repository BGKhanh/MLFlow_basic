import os
import yaml
import copy
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path: str):
        """
        Initialize the Config object.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config_dict = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from a YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return copy.deepcopy(self.config_dict)
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get the dataset configuration."""
        return copy.deepcopy(self.config_dict.get('dataset', {}))
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return copy.deepcopy(self.config_dict.get('model', {}))
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get the training configuration."""
        return copy.deepcopy(self.config_dict.get('training', {}))
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """Get the augmentation configuration."""
        return copy.deepcopy(self.config_dict.get('augmentation', {}))
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get the MLflow configuration."""
        return copy.deepcopy(self.config_dict.get('mlflow', {}))
    
    def get_tune_config(self) -> Dict[str, Any]:
        """Get the Ray Tune configuration."""
        return copy.deepcopy(self.config_dict.get('tune', {}))
    
    def get_distributed_config(self) -> Dict[str, Any]:
        """Get the distributed training configuration."""
        return copy.deepcopy(self.config_dict.get('distributed', {}))
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific configuration value.
        
        Args:
            section: Configuration section (e.g., 'model', 'training')
            key: Configuration key to update
            value: New value
        """
        if section not in self.config_dict:
            self.config_dict[section] = {}
        
        self.config_dict[section][key] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the config. If None, overwrites the original.
        """
        save_path = output_path if output_path else self.config_path
        
        with open(save_path, 'w') as file:
            yaml.dump(self.config_dict, file, default_flow_style=False)


def load_config(config_path: str = 'config/config.yaml') -> Config:
    """
    Helper function to load configuration.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Config object
    """
    return Config(config_path) 