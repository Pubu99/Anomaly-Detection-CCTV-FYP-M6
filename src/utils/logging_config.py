"""
Professional Logging Configuration
=================================

Structured logging setup for the multi-camera anomaly detection system.
Features JSON logging, performance metrics, and distributed logging support.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import structlog


class StructuredLogger:
    """Professional structured logger with performance tracking"""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize structured logger
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
        """
        self.log_level = log_level
        self.log_file = log_file
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup loguru logger with structured formatting"""
        # Remove default logger
        logger.remove()
        
        # JSON formatter for structured logging
        def json_formatter(record):
            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "thread": record["thread"].name,
                "process": record["process"].name,
            }
            
            # Add extra fields
            if record["extra"]:
                log_entry.update(record["extra"])
            
            return json.dumps(log_entry, ensure_ascii=False)
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler with JSON format
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                self.log_file,
                level=self.log_level,
                format=json_formatter,
                rotation="100 MB",
                retention="7 days",
                compression="gz",
                enqueue=True
            )
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        logger.info(
            f"Performance: {operation}",
            duration=duration,
            operation=operation,
            **kwargs
        )
    
    def log_detection(self, detection_data: Dict[str, Any]):
        """Log anomaly detection event"""
        logger.info(
            "Anomaly detection",
            event_type="detection",
            **detection_data
        )
    
    def log_alert(self, alert_data: Dict[str, Any]):
        """Log alert generation"""
        logger.warning(
            "Alert generated",
            event_type="alert",
            **alert_data
        )
    
    def log_model_metrics(self, metrics: Dict[str, float], epoch: int = None):
        """Log model training/validation metrics"""
        logger.info(
            "Model metrics",
            event_type="metrics",
            epoch=epoch,
            **metrics
        )
    
    def log_system_health(self, health_data: Dict[str, Any]):
        """Log system health metrics"""
        logger.info(
            "System health",
            event_type="health",
            **health_data
        )


class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, operation_name: str, logger_instance: StructuredLogger = None):
        self.operation_name = operation_name
        self.logger = logger_instance
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        if self.logger:
            self.logger.log_performance(self.operation_name, duration)
        else:
            logger.info(f"Performance: {self.operation_name} took {duration:.4f}s")


def setup_logging(config: Dict[str, Any]) -> StructuredLogger:
    """Setup logging from configuration"""
    log_config = config.get("monitoring", {}).get("logging", {})
    
    log_level = log_config.get("level", "INFO")
    log_file = log_config.get("file", "logs/app.log")
    
    return StructuredLogger(log_level, log_file)


def get_logger(name: str = None) -> logger:
    """Get logger instance"""
    if name:
        return logger.bind(module=name)
    return logger


# Decorators for automatic logging
def log_function_call(func):
    """Decorator to log function calls with parameters and timing"""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Log function entry
        logger.debug(f"Entering {func_name}", args=str(args), kwargs=kwargs)
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log successful completion
            logger.debug(
                f"Completed {func_name}",
                duration=duration,
                success=True
            )
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log error
            logger.error(
                f"Error in {func_name}",
                duration=duration,
                error=str(e),
                success=False
            )
            raise
    
    return wrapper


def log_model_prediction(func):
    """Decorator to log model predictions"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        
        # Extract prediction info
        prediction_info = {
            "inference_time": duration,
            "model_function": func.__name__,
        }
        
        # Add result info if available
        if hasattr(result, 'confidence'):
            prediction_info["confidence"] = result.confidence
        if hasattr(result, 'class_name'):
            prediction_info["predicted_class"] = result.class_name
        
        logger.info("Model prediction", **prediction_info)
        return result
    
    return wrapper


# Global logger instance
app_logger = None

def init_logging(config: Dict[str, Any]) -> StructuredLogger:
    """Initialize global logging"""
    global app_logger
    app_logger = setup_logging(config)
    return app_logger

def get_app_logger() -> StructuredLogger:
    """Get global application logger"""
    global app_logger
    if app_logger is None:
        app_logger = StructuredLogger()
    return app_logger