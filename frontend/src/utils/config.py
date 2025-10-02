"""
Configuration Management Module
=============================

Professional configuration management for the multi-camera anomaly detection system.
Handles YAML config loading, environment variables, and dynamic configuration updates.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class SystemConfig:
    """System-level configuration"""
    name: str
    version: str
    environment: str
    debug: bool


@dataclass  
class DatasetConfig:
    """Dataset configuration"""
    name: str
    classes: list
    paths: Dict[str, str]
    image_size: list
    channels: int
    batch_size: int
    num_workers: int


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str
    yolo: Dict[str, Any]
    anomaly_classifier: Dict[str, Any]
    fusion: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    epochs: int
    learning_rate: float
    optimizer: str
    scheduler: str
    weight_decay: float
    loss: Dict[str, Any]
    augmentation: Dict[str, Any]
    class_balance: Dict[str, Any]
    validation: Dict[str, Any]
    early_stopping: Dict[str, Any]
    # Optional knobs
    use_mixup: bool = True
    use_cutmix: bool = True
    warmup_epochs: int = 0
    ema_decay: float = 0.0
    progressive_resizing: Dict[str, Any] = field(default_factory=dict)
    logit_adjustment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int
    max_fps: int
    frame_sampling_rate: int
    optimization: Dict[str, Any]
    alerts: Dict[str, Any]


class ConfigManager:
    """Professional configuration manager with validation and hot-reload capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file. If None, looks for config.yaml
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._validate_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        possible_paths = [
            "config/config.yaml",
            "config.yaml",
            "../config/config.yaml",
            "../../config/config.yaml"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("Could not find config.yaml in standard locations")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Substitute environment variables
            config = self._substitute_env_vars(config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            default_value = None
            
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            return os.getenv(env_var, default_value)
        else:
            return config
    
    def _validate_config(self):
        """Validate configuration structure and required fields"""
        required_sections = ['system', 'dataset', 'model', 'training', 'inference']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate specific constraints
        if self.config['dataset']['batch_size'] <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.config['training']['epochs'] <= 0:
            raise ValueError("Number of epochs must be positive")
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'model.yolo.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration as dataclass"""
        return SystemConfig(**self.config['system'])
    
    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration as dataclass"""
        return DatasetConfig(**self.config['dataset'])
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as dataclass"""
        return ModelConfig(**self.config['model'])
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration as dataclass"""
        return TrainingConfig(**self.config['training'])
    
    def get_inference_config(self) -> InferenceConfig:
        """Get inference configuration as dataclass"""
        return InferenceConfig(**self.config['inference'])
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = path or self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded")
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        self._validate_config()
        logger.info("Configuration updated from dictionary")


# Global configuration instance
config = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global config
    if config is None:
        config = ConfigManager()
    return config

def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration"""
    global config
    config = ConfigManager(config_path)
    return config