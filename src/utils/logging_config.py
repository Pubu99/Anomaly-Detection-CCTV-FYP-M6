"""
Logging configuration for the anomaly detection system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Log format string (optional)
        console_output: Whether to output logs to console
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_app_logger(name: str = "anomaly_detection") -> logging.Logger:
    """
    Get application logger with standardized configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set up default logging
    if not logger.handlers:
        setup_logging()
    
    return logger


def get_training_logger() -> logging.Logger:
    """Get logger specifically for training operations."""
    return get_app_logger("training")


def get_model_logger() -> logging.Logger:
    """Get logger specifically for model operations."""
    return get_app_logger("model")


def get_inference_logger() -> logging.Logger:
    """Get logger specifically for inference operations."""
    return get_app_logger("inference")


def get_data_logger() -> logging.Logger:
    """Get logger specifically for data operations."""
    return get_app_logger("data")


def create_experiment_logger(experiment_name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a logger for a specific experiment with file output.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        
    Returns:
        Configured logger with file output
    """
    # Create unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{experiment_name}_{timestamp}.log"
    
    # Setup logging with file output
    setup_logging(log_file=log_file)
    
    # Return logger for this experiment
    return get_app_logger(f"experiment.{experiment_name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_app_logger(self.__class__.__name__)


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance (optional, will use default if not provided)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                log = get_app_logger(func.__module__)
            else:
                log = logger
                
            log.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                log.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                log.error(f"{func.__name__} failed with error: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_performance(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function performance metrics.
    
    Args:
        logger: Logger instance (optional, will use default if not provided)
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                log = get_app_logger(func.__module__)
            else:
                log = logger
                
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {str(e)}")
                raise
                
        return wrapper
    return decorator


# Initialize default logging when module is imported
setup_logging(log_level="INFO")