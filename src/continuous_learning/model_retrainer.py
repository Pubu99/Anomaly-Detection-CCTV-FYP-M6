"""
Model Retraining System
======================

Automated model retraining pipeline with continuous learning capabilities.
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
from PIL import Image
import yaml
from dataclasses import dataclass
import pickle
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Import our existing modules
from .feedback_collector import FeedbackCollector, FeedbackEntry, FeedbackType, ConfidenceLevel
from ..models.hybrid_model import HybridAnomalyDetector
from ..training.train import ModelTrainer
from ..utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrainingConfig:
    """Configuration for model retraining"""
    # Retraining triggers
    min_feedback_count: int = 100
    feedback_quality_threshold: float = 0.8
    accuracy_drop_threshold: float = 0.05
    retrain_interval_days: int = 7
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.0001
    max_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Data parameters
    min_confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    balance_classes: bool = True
    augment_data: bool = True
    
    # Model versioning
    model_registry_path: str = "models/registry/"
    backup_models: bool = True
    max_model_versions: int = 10
    
    # Performance tracking
    track_with_mlflow: bool = True
    mlflow_experiment_name: str = "continuous_learning"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'RetrainingConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('retraining', {}))

class FeedbackDataset(Dataset):
    """Dataset for training on user feedback data"""
    
    def __init__(self, feedback_entries: List[FeedbackEntry], 
                 transform=None, target_transform=None):
        self.feedback_entries = feedback_entries
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.feedback_entries)
    
    def __getitem__(self, idx):
        feedback = self.feedback_entries[idx]
        
        # Load image
        image = None
        if feedback.image_path and Path(feedback.image_path).exists():
            image = cv2.imread(feedback.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Create dummy image if no image available
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Prepare label based on feedback type
        label = self._extract_label(feedback)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return {
            'image': image,
            'label': label,
            'feedback_id': feedback.id,
            'confidence_level': feedback.confidence_level.value,
            'feedback_type': feedback.feedback_type.value
        }
    
    def _extract_label(self, feedback: FeedbackEntry) -> Dict[str, Any]:
        """Extract training label from feedback"""
        if feedback.feedback_type == FeedbackType.TRUE_POSITIVE:
            # Use original prediction as ground truth
            return feedback.original_prediction
        elif feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
            # Mark as background/normal
            return {'anomaly': False, 'confidence': 1.0, 'class': 'normal'}
        elif feedback.feedback_type == FeedbackType.CLASSIFICATION_CORRECTION:
            # Use user correction
            return feedback.user_correction
        elif feedback.feedback_type == FeedbackType.SEVERITY_CORRECTION:
            # Update severity while keeping other info
            corrected = feedback.original_prediction.copy()
            corrected.update(feedback.user_correction)
            return corrected
        else:
            # Default to original prediction
            return feedback.original_prediction

class ModelRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, config: RetrainingConfig, 
                 feedback_collector: FeedbackCollector):
        self.config = config
        self.feedback_collector = feedback_collector
        self.model_registry = Path(config.model_registry_path)
        self.model_registry.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow if enabled
        if config.track_with_mlflow:
            mlflow.set_experiment(config.mlflow_experiment_name)
        
        # Model performance tracking
        self.performance_history = []
        self.current_model_version = None
        self.baseline_metrics = None
        
        # Load current model and metrics
        self._load_current_model_info()
        
    def _load_current_model_info(self):
        """Load information about current model"""
        try:
            info_file = self.model_registry / "current_model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    self.current_model_version = info.get('version')
                    self.baseline_metrics = info.get('metrics')
                    logger.info(f"Loaded current model info: version {self.current_model_version}")
        except Exception as e:
            logger.error(f"Failed to load current model info: {e}")
    
    async def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model should be retrained
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        try:
            # Check feedback count
            feedback_summary = await self.feedback_collector.get_feedback_summary()
            total_feedback = feedback_summary.get('total_feedback', 0)
            
            if total_feedback < self.config.min_feedback_count:
                return False, f"Insufficient feedback: {total_feedback} < {self.config.min_feedback_count}"
            
            # Check feedback quality
            quality_score = self._calculate_feedback_quality(feedback_summary)
            if quality_score < self.config.feedback_quality_threshold:
                return False, f"Low feedback quality: {quality_score:.3f} < {self.config.feedback_quality_threshold}"
            
            # Check time since last retraining
            last_retrain = self._get_last_retrain_date()
            if last_retrain:
                days_since = (datetime.now() - last_retrain).days
                if days_since < self.config.retrain_interval_days:
                    return False, f"Too soon since last retrain: {days_since} < {self.config.retrain_interval_days} days"
            
            # Check performance degradation
            if self.baseline_metrics:
                current_performance = await self._evaluate_current_model()
                if current_performance:
                    accuracy_drop = self.baseline_metrics.get('accuracy', 0) - current_performance.get('accuracy', 0)
                    if accuracy_drop > self.config.accuracy_drop_threshold:
                        return True, f"Performance degradation detected: {accuracy_drop:.3f} drop in accuracy"
            
            # Default: retrain if we have enough quality feedback
            return True, f"Sufficient quality feedback available: {total_feedback} entries"
            
        except Exception as e:
            logger.error(f"Failed to check retraining criteria: {e}")
            return False, f"Error checking criteria: {str(e)}"
    
    def _calculate_feedback_quality(self, feedback_summary: Dict[str, Any]) -> float:
        """Calculate overall feedback quality score"""
        try:
            confidence_distribution = feedback_summary.get('feedback_by_confidence', {})
            total_feedback = sum(confidence_distribution.values())
            
            if total_feedback == 0:
                return 0.0
            
            # Weight feedback by confidence level
            quality_weights = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'expert': 1.0
            }
            
            weighted_sum = sum(
                confidence_distribution.get(level, 0) * weight 
                for level, weight in quality_weights.items()
            )
            
            return weighted_sum / total_feedback
            
        except Exception as e:
            logger.error(f"Failed to calculate feedback quality: {e}")
            return 0.0
    
    def _get_last_retrain_date(self) -> Optional[datetime]:
        """Get date of last retraining"""
        try:
            retrain_log = self.model_registry / "retrain_log.json"
            if retrain_log.exists():
                with open(retrain_log, 'r') as f:
                    log_data = json.load(f)
                    if log_data.get('retraining_history'):
                        last_entry = log_data['retraining_history'][-1]
                        return datetime.fromisoformat(last_entry['timestamp'])
            return None
        except Exception as e:
            logger.error(f"Failed to get last retrain date: {e}")
            return None
    
    async def _evaluate_current_model(self) -> Optional[Dict[str, float]]:
        """Evaluate current model performance"""
        try:
            # Get recent feedback for evaluation
            evaluation_feedback = await self.feedback_collector.get_training_data(
                feedback_types=[FeedbackType.TRUE_POSITIVE, FeedbackType.FALSE_POSITIVE],
                min_confidence=ConfidenceLevel.HIGH,
                limit=1000
            )
            
            if len(evaluation_feedback) < 50:
                logger.warning("Insufficient feedback for model evaluation")
                return None
            
            # Calculate metrics
            correct_predictions = 0
            total_predictions = len(evaluation_feedback)
            
            for feedback in evaluation_feedback:
                original_pred = feedback.original_prediction
                is_correct = (feedback.feedback_type == FeedbackType.TRUE_POSITIVE)
                if is_correct:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_predictions
            
            return {
                'accuracy': accuracy,
                'total_samples': total_predictions,
                'evaluation_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate current model: {e}")
            return None
    
    async def retrain_model(self) -> Dict[str, Any]:
        """
        Perform model retraining with feedback data
        
        Returns:
            Retraining results including metrics and model version
        """
        try:
            logger.info("Starting model retraining...")
            
            # Start MLflow run if enabled
            if self.config.track_with_mlflow:
                mlflow.start_run()
                mlflow.log_params({
                    'min_feedback_count': self.config.min_feedback_count,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                    'max_epochs': self.config.max_epochs
                })
            
            # Get training data from feedback
            training_feedback = await self.feedback_collector.get_training_data(
                min_confidence=self.config.min_confidence_level
            )
            
            if len(training_feedback) < self.config.min_feedback_count:
                raise ValueError(f"Insufficient training data: {len(training_feedback)} < {self.config.min_feedback_count}")
            
            logger.info(f"Retrieved {len(training_feedback)} feedback entries for training")
            
            # Prepare datasets
            train_dataset, val_dataset = self._prepare_datasets(training_feedback)
            
            # Load existing model
            model = self._load_model_for_retraining()
            
            # Create trainer
            trainer = ModelTrainer(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=self._get_training_config()
            )
            
            # Train model
            training_results = await trainer.train()
            
            # Evaluate retrained model
            evaluation_results = await trainer.evaluate()
            
            # Save new model version
            new_version = await self._save_model_version(model, training_results, evaluation_results)
            
            # Update performance tracking
            await self._update_performance_tracking(evaluation_results, new_version)
            
            # Mark feedback as processed
            feedback_ids = [f.id for f in training_feedback]
            await self.feedback_collector.mark_feedback_processed(feedback_ids)
            
            # Log to MLflow
            if self.config.track_with_mlflow:
                mlflow.log_metrics(evaluation_results)
                mlflow.pytorch.log_model(model, "retrained_model")
                mlflow.end_run()
            
            # Update retraining log
            await self._update_retrain_log(new_version, training_results, evaluation_results)
            
            results = {
                'success': True,
                'new_model_version': new_version,
                'training_samples': len(training_feedback),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'retrain_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Model retraining completed successfully. New version: {new_version}")
            return results
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            if self.config.track_with_mlflow and mlflow.active_run():
                mlflow.end_run(status='FAILED')
            
            return {
                'success': False,
                'error': str(e),
                'retrain_timestamp': datetime.now().isoformat()
            }
    
    def _prepare_datasets(self, feedback_entries: List[FeedbackEntry]) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        try:
            # Split feedback into train/val
            train_feedback, val_feedback = train_test_split(
                feedback_entries,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=[f.feedback_type.value for f in feedback_entries]
            )
            
            # Create datasets
            # Note: Transform functions would be defined based on model requirements
            train_dataset = FeedbackDataset(train_feedback, transform=None)
            val_dataset = FeedbackDataset(val_feedback, transform=None)
            
            logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} validation")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare datasets: {e}")
            raise
    
    def _load_model_for_retraining(self) -> nn.Module:
        """Load existing model for retraining"""
        try:
            if self.current_model_version:
                model_path = self.model_registry / f"model_v{self.current_model_version}.pth"
                if model_path.exists():
                    # Load existing model
                    model = HybridAnomalyDetector()
                    model.load_state_dict(torch.load(model_path))
                    logger.info(f"Loaded model version {self.current_model_version} for retraining")
                    return model
            
            # Create new model if no existing model
            config = Config()
            model = HybridAnomalyDetector(config)
            logger.info("Created new model for training")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model for retraining: {e}")
            raise
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for retraining"""
        return {
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'max_epochs': self.config.max_epochs,
            'early_stopping_patience': self.config.early_stopping_patience,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    async def _save_model_version(self, model: nn.Module, 
                                training_results: Dict[str, Any],
                                evaluation_results: Dict[str, Any]) -> str:
        """Save new model version"""
        try:
            # Generate new version number
            new_version = self._generate_version_number()
            
            # Save model state
            model_path = self.model_registry / f"model_v{new_version}.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save model metadata
            metadata = {
                'version': new_version,
                'creation_date': datetime.now().isoformat(),
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'config': self.config.__dict__
            }
            
            metadata_path = self.model_registry / f"model_v{new_version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update current model info
            current_info = {
                'version': new_version,
                'metrics': evaluation_results,
                'last_updated': datetime.now().isoformat()
            }
            
            info_path = self.model_registry / "current_model_info.json"
            with open(info_path, 'w') as f:
                json.dump(current_info, f, indent=2)
            
            # Backup old models if enabled
            if self.config.backup_models:
                await self._manage_model_versions()
            
            self.current_model_version = new_version
            logger.info(f"Saved model version {new_version}")
            return new_version
            
        except Exception as e:
            logger.error(f"Failed to save model version: {e}")
            raise
    
    def _generate_version_number(self) -> str:
        """Generate new version number"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}"
    
    async def _manage_model_versions(self):
        """Manage model versions, keeping only the latest N versions"""
        try:
            # Get all model files
            model_files = list(self.model_registry.glob("model_v*.pth"))
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old versions if we exceed the limit
            if len(model_files) > self.config.max_model_versions:
                for old_model in model_files[self.config.max_model_versions:]:
                    # Remove model file and metadata
                    old_model.unlink()
                    metadata_file = old_model.with_suffix('_metadata.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    logger.info(f"Removed old model: {old_model.name}")
            
        except Exception as e:
            logger.error(f"Failed to manage model versions: {e}")
    
    async def _update_performance_tracking(self, evaluation_results: Dict[str, Any], version: str):
        """Update performance tracking records"""
        try:
            performance_entry = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'metrics': evaluation_results
            }
            
            self.performance_history.append(performance_entry)
            
            # Save performance history
            history_path = self.model_registry / "performance_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update performance tracking: {e}")
    
    async def _update_retrain_log(self, version: str, 
                                training_results: Dict[str, Any],
                                evaluation_results: Dict[str, Any]):
        """Update retraining log"""
        try:
            log_entry = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
            
            log_path = self.model_registry / "retrain_log.json"
            
            if log_path.exists():
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'retraining_history': []}
            
            log_data['retraining_history'].append(log_entry)
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to update retrain log: {e}")
    
    async def get_retraining_status(self) -> Dict[str, Any]:
        """Get current retraining system status"""
        try:
            should_retrain, reason = await self.should_retrain()
            feedback_summary = await self.feedback_collector.get_feedback_summary()
            
            return {
                'should_retrain': should_retrain,
                'reason': reason,
                'current_model_version': self.current_model_version,
                'feedback_summary': feedback_summary,
                'last_retrain_date': self._get_last_retrain_date(),
                'performance_history': self.performance_history[-5:],  # Last 5 entries
                'model_registry_path': str(self.model_registry)
            }
            
        except Exception as e:
            logger.error(f"Failed to get retraining status: {e}")
            return {'error': str(e)}

# Export classes
__all__ = ['ModelRetrainer', 'RetrainingConfig', 'FeedbackDataset']