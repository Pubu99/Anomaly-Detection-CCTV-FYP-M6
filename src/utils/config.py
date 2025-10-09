"""
Configuration management for the anomaly detection system.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Get the root directory of the project
        root_dir = Path(__file__).parent.parent.parent
        config_path = root_dir / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration parameters for a specific section or entire config.
    
    Args:
        section: Configuration section name (e.g., 'model', 'training', 'dataset')
                If None, returns entire config.
        
    Returns:
        Configuration dictionary for the specified section or entire config
    """
    config = load_config()
    
    if section is None:
        return config
    
    if section not in config:
        raise KeyError(f"Configuration section '{section}' not found")
    
    return config[section]


def get_model_config() -> Dict[str, Any]:
    """Get model configuration parameters."""
    return get_config("model")


def get_training_config() -> Dict[str, Any]:
    """Get training configuration parameters."""
    return get_config("training")


def get_dataset_config() -> Dict[str, Any]:
    """Get dataset configuration parameters."""
    return get_config("dataset")


def get_data_paths() -> Dict[str, str]:
    """Get data paths configuration."""
    return get_config("data_paths")


def get_system_config() -> Dict[str, Any]:
    """Get system configuration parameters."""
    return get_config("system")


def update_config(section: str, key: str, value: Any) -> None:
    """
    Update configuration parameter (in memory only).
    
    Args:
        section: Configuration section name
        key: Parameter key
        value: New parameter value
    """
    config = load_config()
    
    if section not in config:
        config[section] = {}
    
    config[section][key] = value


class Config:
    """Configuration class for easy access to configuration parameters."""
    
    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None):
        """
        Get configuration parameter.
        
        Args:
            section: Configuration section name
            key: Parameter key (optional)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if section not in self._config:
            return default
            
        section_config = self._config[section]
        
        if key is None:
            return section_config
            
        return section_config.get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration parameter.
        
        Args:
            section: Configuration section name
            key: Parameter key
            value: Parameter value
        """
        if section not in self._config:
            self._config[section] = {}
            
        self._config[section][key] = value
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get("model", default={})
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get("training", default={})
    
    @property
    def dataset(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.get("dataset", default={})
    
    @property
    def data_paths(self) -> Dict[str, str]:
        """Get data paths configuration."""
        return self.get("data_paths", default={})
    
    @property
    def system(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self.get("system", default={})


# Global config instance
_global_config = None


def get_global_config() -> Config:
    """Get global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_global_config(config: Config) -> None:
    """Set global configuration instance."""
    global _global_config
    _global_config = config