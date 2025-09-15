"""
Professional Training Pipeline for Multi-Camera Anomaly Detection
================================================================

Advanced training pipeline with focal loss, progressive learning, 
model optimization, and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from src.models.hybrid_model import create_model, HybridAnomalyModel
from src.data.data_loader import DataLoaderFactory, load_data_splits
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger, PerformanceTimer


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: Predictions [B, num_classes]
            targets: Ground truth labels [B]
            
        Returns:
            Focal loss value
        """
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with label smoothing"""
        log_probs = torch.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.fill_(self.smoothing / (self.num_classes - 1))
        targets_one_hot.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(targets_one_hot * log_probs, dim=1)
        return loss.mean()


class MetricsCalculator:
    """Professional metrics calculation for anomaly detection"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
        self.confidences = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, confidences: torch.Tensor):
        """Update metrics with batch results"""
        self.predictions.extend(preds.detach().cpu().numpy())
        self.targets.extend(targets.detach().cpu().numpy())
        self.confidences.extend(confidences.detach().cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute comprehensive metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        confidences = np.array(self.confidences)
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        f1_macro = f1_score(targets, predictions, average='macro')
        f1_weighted = f1_score(targets, predictions, average='weighted')
        
        # Per-class metrics
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Confidence-based metrics
        avg_confidence = np.mean(confidences)
        correct_confidence = np.mean(confidences[predictions == targets])
        incorrect_confidence = np.mean(confidences[predictions != targets]) if np.any(predictions != targets) else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_confidence': avg_confidence,
            'correct_confidence': correct_confidence,
            'incorrect_confidence': incorrect_confidence,
            'confidence_gap': correct_confidence - incorrect_confidence
        }
        
        # Add per-class F1 scores
        for class_name in self.class_names:
            if class_name in report:
                metrics[f'f1_{class_name}'] = report[class_name]['f1-score']
        
        return metrics


class AnomalyTrainer:
    """Professional trainer for anomaly detection model"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = get_config() if config_path is None else get_config(config_path)
        self.logger = get_app_logger()
        
        # Setup device with comprehensive CUDA information
        self._setup_device()
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_monitoring()
    
    def _setup_device(self):
        """Setup device and display comprehensive CUDA information"""
        self.logger.info("=" * 60)
        self.logger.info("DEVICE AND CUDA SETUP")
        self.logger.info("=" * 60)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            # CUDA device information
            current_device = torch.cuda.current_device()
            device_count = torch.cuda.device_count()
            
            self.logger.info(f"CUDA Device Count: {device_count}")
            self.logger.info(f"Current CUDA Device: {current_device}")
            
            # Device properties
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                self.logger.info(f"Device {i}: {props.name}")
                self.logger.info(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                self.logger.info(f"  - Multi-processor Count: {props.multi_processor_count}")
                self.logger.info(f"  - CUDA Capability: {props.major}.{props.minor}")
            
            # Memory information
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"CUDA Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            # CUDA version information
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
            self.logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
            
            self.device = torch.device(f'cuda:{current_device}')
            self.logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(current_device)}")
            
        else:
            self.logger.warning("❌ CUDA not available! Training will use CPU")
            self.logger.info("Reasons CUDA might not be available:")
            self.logger.info("  - PyTorch not installed with CUDA support")
            self.logger.info("  - No NVIDIA GPU detected")
            self.logger.info("  - CUDA drivers not properly installed")
            self.logger.info("  - CUDA version mismatch")
            
            self.device = torch.device('cpu')
        
        # PyTorch and system information
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"Device Selected: {self.device}")
        
        # Performance recommendations
        if cuda_available:
            self.logger.info("🚀 GPU Training Recommendations:")
            self.logger.info("  - Increase batch size if memory allows")
            self.logger.info("  - Use mixed precision training for speed")
            self.logger.info("  - Monitor GPU utilization during training")
        else:
            self.logger.info("💡 CPU Training Recommendations:")
            self.logger.info("  - Reduce batch size to prevent memory issues")
            self.logger.info("  - Consider using smaller model variants")
            self.logger.info("  - Expect significantly longer training times")
        
        self.logger.info("=" * 60)
    
    def _monitor_cuda_status(self, epoch: int = None):
        """Monitor CUDA status and memory usage"""
        if not torch.cuda.is_available():
            return
        
        try:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Memory statistics
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
            
            # GPU utilization (if available)
            if epoch is not None:
                prefix = f"[Epoch {epoch}] "
            else:
                prefix = ""
            
            self.logger.info(f"{prefix}CUDA Status:")
            self.logger.info(f"  Device: {device_name}")
            self.logger.info(f"  Memory Allocated: {allocated:.2f} GB")
            self.logger.info(f"  Memory Reserved: {reserved:.2f} GB")
            self.logger.info(f"  Max Memory Allocated: {max_allocated:.2f} GB")
            self.logger.info(f"  Max Memory Reserved: {max_reserved:.2f} GB")
            
            # Check for memory issues
            total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            usage_percent = (allocated / total_memory) * 100
            
            if usage_percent > 90:
                self.logger.warning(f"⚠️  High GPU memory usage: {usage_percent:.1f}%")
            elif usage_percent > 75:
                self.logger.info(f"📊 GPU memory usage: {usage_percent:.1f}%")
            else:
                self.logger.info(f"✅ GPU memory usage: {usage_percent:.1f}%")
                
        except Exception as e:
            self.logger.warning(f"Failed to get CUDA status: {e}")
    
    def _check_cuda_health(self):
        """Perform a quick CUDA health check"""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Test tensor operations
            test_tensor = torch.randn(100, 100).to(self.device)
            result = torch.mm(test_tensor, test_tensor.T)
            del test_tensor, result
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            self.logger.error(f"CUDA health check failed: {e}")
            return False
    
    def _setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        # Load data splits
        splits = load_data_splits("data/processed/data_splits.json")
        
        # Create data loaders
        self.data_loaders = DataLoaderFactory.create_data_loaders(
            train_paths=splits['train']['paths'],
            train_labels=splits['train']['labels'],
            val_paths=splits['val']['paths'],
            val_labels=splits['val']['labels'],
            class_names=self.config.get('dataset.classes'),
            config=self.config.config,
            test_paths=splits['test']['paths'],
            test_labels=splits['test']['labels']
        )
        
        self.logger.info(f"Data loaders created: Train={len(self.data_loaders['train'])}, "
                        f"Val={len(self.data_loaders['val'])}, Test={len(self.data_loaders['test'])}")
    
    def _setup_model(self):
        """Setup model and move to device"""
        self.logger.info("Setting up model...")
        
        self.model = create_model(self.config.config)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _setup_training(self):
        """Setup training components"""
        training_config = self.config.get_training_config()
        
        # Loss function
        loss_config = training_config.loss
        if loss_config['type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=loss_config['alpha'],
                gamma=loss_config['gamma']
            )
        elif loss_config['type'] == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(
                num_classes=len(self.config.get('dataset.classes')),
                smoothing=loss_config['label_smoothing']
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if training_config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        elif training_config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                momentum=0.9,
                weight_decay=training_config.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        
        # Scheduler
        if training_config.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.epochs
            )
        elif training_config.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Metrics
        self.train_metrics = MetricsCalculator(
            num_classes=len(self.config.get('dataset.classes')),
            class_names=self.config.get('dataset.classes')
        )
        self.val_metrics = MetricsCalculator(
            num_classes=len(self.config.get('dataset.classes')),
            class_names=self.config.get('dataset.classes')
        )
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.best_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rates': []
        }
    
    def _setup_monitoring(self):
        """Setup monitoring and logging"""
        # Create directories
        self.checkpoint_dir = Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path("logs/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if configured
        if self.config.get('monitoring.use_wandb', False):
            wandb.init(
                project="anomaly-detection",
                config=self.config.config,
                name=f"experiment_{int(time.time())}"
            )
            wandb.watch(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        progress_bar = tqdm(
            self.data_loaders['train'], 
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False
        )
        
        for batch_idx, (images, labels, metadata) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            results = self.model(images)
            logits = results['anomaly_logits']
            confidences = results['final_scores'].squeeze()
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(logits, dim=1)
            self.train_metrics.update(predictions, labels, confidences)
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.train_metrics.compute()
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.data_loaders['val'])
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.data_loaders['val'], desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                results = self.model(images)
                logits = results['anomaly_logits']
                confidences = results['final_scores'].squeeze()
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                running_loss += loss.item()
                
                # Update metrics
                predictions = torch.argmax(logits, dim=1)
                self.val_metrics.update(predictions, labels, confidences)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.val_metrics.compute()
        
        return {'loss': epoch_loss, **epoch_metrics}
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with F1: {metrics['f1_macro']:.4f}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score plot
        axes[1, 0].plot(self.training_history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.training_history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.training_history['learning_rates'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'training_history_epoch_{self.current_epoch}.png', dpi=300)
        plt.close()
    
    def train(self):
        """Main training loop"""
        training_config = self.config.get_training_config()
        num_epochs = training_config.epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Initial CUDA health check
        if torch.cuda.is_available():
            cuda_healthy = self._check_cuda_health()
            if not cuda_healthy:
                self.logger.error("❌ CUDA health check failed! Training may encounter issues.")
            else:
                self.logger.info("✅ CUDA health check passed!")
            
            # Initial CUDA status
            self._monitor_cuda_status()
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Periodic CUDA monitoring (every 5 epochs)
            if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                self._monitor_cuda_status(epoch + 1)
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1_macro'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Train F1: {train_metrics['f1_macro']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1_macro'])
            self.training_history['val_f1'].append(val_metrics['f1_macro'])
            self.training_history['learning_rates'].append(current_lr)
            
            # Check for best model
            is_best = val_metrics['f1_macro'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['f1_macro']
                self.best_accuracy = val_metrics['accuracy']
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Log to wandb
            if hasattr(wandb, 'log'):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1_macro'],
                    'val_f1': val_metrics['f1_macro'],
                    'learning_rate': current_lr
                })
            
            # Plot training history periodically
            if (epoch + 1) % 10 == 0:
                self.plot_training_history()
        
        # Final evaluation
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation F1: {self.best_f1:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_accuracy:.4f}")
        
        # Final CUDA status check
        if torch.cuda.is_available():
            self.logger.info("\n" + "=" * 50)
            self.logger.info("FINAL CUDA STATUS")
            self.logger.info("=" * 50)
            self._monitor_cuda_status()
            
            # Reset peak memory stats for final reporting
            torch.cuda.reset_peak_memory_stats()
        
        # Final plots
        self.plot_training_history()
        
        # Test evaluation
        self.evaluate_test_set()
    
    def evaluate_test_set(self):
        """Evaluate on test set"""
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / "best_model.pth")
        self.model.load_state_dict(best_checkpoint['state_dict'])
        
        test_metrics = MetricsCalculator(
            num_classes=len(self.config.get('dataset.classes')),
            class_names=self.config.get('dataset.classes')
        )
        
        self.model.eval()
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.data_loaders['test'], desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                results = self.model(images)
                logits = results['anomaly_logits']
                confidences = results['final_scores'].squeeze()
                
                predictions = torch.argmax(logits, dim=1)
                test_metrics.update(predictions, labels, confidences)
        
        final_metrics = test_metrics.compute()
        
        self.logger.info("Test Set Results:")
        self.logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
        self.logger.info(f"F1 Macro: {final_metrics['f1_macro']:.4f}")
        self.logger.info(f"F1 Weighted: {final_metrics['f1_weighted']:.4f}")
        
        # Save test results
        test_results = {
            'test_metrics': final_metrics,
            'best_validation_f1': self.best_f1,
            'best_validation_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }
        
        with open(self.checkpoint_dir / "test_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)


def main():
    """Main training function"""
    # Quick CUDA connectivity check before starting
    print("=" * 60)
    print("ANOMALY DETECTION TRAINING - CUDA CONNECTIVITY CHECK")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"🎯 Device count: {torch.cuda.device_count()}")
        print(f"🔧 Current device: {torch.cuda.current_device()}")
        print(f"💻 Device name: {torch.cuda.get_device_name()}")
        print(f"🚀 CUDA version: {torch.version.cuda}")
        print(f"⚡ Ready for GPU training!")
    else:
        print("❌ CUDA is NOT available!")
        print("💡 Training will proceed on CPU (slower)")
        print("🔍 Check your PyTorch installation and GPU drivers")
    
    print("=" * 60)
    print()
    
    trainer = AnomalyTrainer()
    trainer.train()


if __name__ == "__main__":
    main()