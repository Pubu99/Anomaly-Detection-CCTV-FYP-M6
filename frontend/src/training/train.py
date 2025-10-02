"""
Professional Training Pipeline for Multi-Camera Anomaly Detection
================================================================

Advanced training pipeline with focal loss, progressive learning, 
model optimization, and comprehensive evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ClassBalancedFocalLoss(nn.Module):
    """Enhanced Focal Loss with class balancing for extreme imbalance"""
    
    def __init__(self, class_freq: List[int], beta: float = 0.9999, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Class-Balanced Focal Loss
        
        Args:
            class_freq: List of class frequencies
            beta: Hyperparameter for re-weighting
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.FloatTensor(class_freq))
        weights = (1.0 - beta) / effective_num
        self.alpha = weights / weights.sum() * len(weights)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with class balancing"""
        self.alpha = self.alpha.to(inputs.device)
        
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Apply class-specific alpha weights
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BalancedCrossEntropyLoss(nn.Module):
    """Balanced Cross Entropy with automatic class weight calculation"""
    
    def __init__(self, class_freq: List[int], beta: float = 0.999):
        super().__init__()
        self.beta = beta
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(beta, torch.FloatTensor(class_freq))
        weights = (1.0 - beta) / effective_num
        self.weights = weights / weights.sum() * len(weights)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass with balanced weights"""
        self.weights = self.weights.to(inputs.device)
        return nn.CrossEntropyLoss(weight=self.weights)(inputs, targets)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for better feature separation"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss"""
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create label matrix
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Remove diagonal
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(features.device)
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute contrastive loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function for extreme imbalance"""
    
    def __init__(self, class_freq: List[int], focal_weight: float = 0.6, ce_weight: float = 0.3, contrastive_weight: float = 0.1):
        super().__init__()
        self.focal_loss = ClassBalancedFocalLoss(class_freq, beta=0.9999, gamma=2.5)
        self.ce_loss = BalancedCrossEntropyLoss(class_freq, beta=0.999)
        self.contrastive_loss = ContrastiveLoss(temperature=0.05)
        
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(self, logits: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined loss computation"""
        focal = self.focal_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        contrastive = self.contrastive_loss(features, targets)
        
        total_loss = (self.focal_weight * focal + 
                     self.ce_weight * ce + 
                     self.contrastive_weight * contrastive)
        
        return total_loss


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


class MixupAugmentation:
    """Mixup data augmentation for better generalization"""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """Apply mixup augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMixAugmentation:
    """CutMix data augmentation for better localization"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """Apply cutmix augmentation"""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        y_a, y_b = y, y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


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
        
        # AMP and EMA initialization (must be before model setup)
        self.use_amp = bool(self.config.get('hardware.mixed_precision', True)) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)  # Fixed deprecated warning
        self.ema_decay = float(self.config.get('training.ema_decay', 0.999))
        self.use_ema = self.ema_decay > 0
        self.ema_state = None

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

        # Initialize EMA with current weights
        if getattr(self, 'use_ema', False):
            self._init_ema()
    
    def _setup_training(self):
        """Setup training components with advanced techniques for high accuracy"""
        training_config = self.config.get_training_config()
        
        # Calculate class frequencies for balanced loss
        class_names = self.config.get('dataset.classes')
        class_frequencies = self._calculate_class_frequencies()
        
        # Advanced Combined Loss Function for extreme imbalance
        loss_config = training_config.loss
        if loss_config['type'] == 'focal':
            # Use combined loss for extreme imbalance scenarios
            self.criterion = CombinedLoss(
                class_freq=class_frequencies,
                focal_weight=0.5,
                ce_weight=0.3,
                contrastive_weight=0.2
            )
            self.use_contrastive = True
        elif loss_config['type'] == 'class_balanced_focal':
            # Use class-balanced focal loss
            self.criterion = ClassBalancedFocalLoss(
                class_freq=class_frequencies,
                beta=0.9999,  # For extreme imbalance
                gamma=loss_config.get('gamma', 2.5)
            )
            self.use_contrastive = False
        elif loss_config['type'] == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(
                num_classes=len(class_names),
                smoothing=loss_config['label_smoothing']
            )
            self.use_contrastive = False
        else:
            # Use weighted CrossEntropyLoss as fallback
            class_weights = self._calculate_class_weights(class_frequencies)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.use_contrastive = False
        
        # Advanced data augmentation
        self.mixup = MixupAugmentation(alpha=0.2)
        self.cutmix = CutMixAugmentation(alpha=1.0)
        self.use_mixup = getattr(training_config, 'use_mixup', True)
        self.use_cutmix = getattr(training_config, 'use_cutmix', True)

        # Logit adjustment for long-tail with enhanced parameters
        la_cfg = self.config.get('training.logit_adjustment', {'enabled': True, 'tau': 2.0})  # Enhanced defaults
        self.use_logit_adjust = bool(la_cfg.get('enabled', True))
        tau = float(la_cfg.get('tau', 2.0))
        if self.use_logit_adjust:
            freq = torch.tensor([max(f, 1) for f in class_frequencies], dtype=torch.float32)
            prior = freq / freq.sum()
            # Enhanced logit adjustment for extreme imbalance
            self.logit_bias = (-tau * torch.log(prior.clamp_min(1e-8))).to(self.device)
            self.logger.info(f"✅ Logit adjustment enabled with tau={tau}")
        else:
            self.logit_bias = None
        
        # Optimizer with improved settings
        if training_config.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif training_config.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                momentum=0.9,
                weight_decay=training_config.weight_decay,
                nesterov=True
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
        
        # Scheduler (with optional warmup)
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
        
        # Early stopping
        training_config = self.config.get_training_config()
        early_stopping_config = getattr(training_config, 'early_stopping', {})
        self.early_stopping_patience = early_stopping_config.get('patience', 10)
        self.early_stopping_min_delta = early_stopping_config.get('min_delta', 0.001)
        self.early_stopping_counter = 0
        self.best_metric_value = 0.0

        # Progressive resizing support
        pr_cfg = self.config.get('training.progressive_resizing', None)
        self.progressive_sizes = None
        if isinstance(pr_cfg, dict) and pr_cfg.get('enabled', False):
            start = tuple(pr_cfg.get('start_size', [128, 128]))
            mid = tuple(pr_cfg.get('mid_size', [192, 192]))
            end = tuple(self.config.get('dataset.image_size', [224, 224]))
            self.progressive_sizes = [start, mid, tuple(end)]
            self.pr_epochs = list(map(int, pr_cfg.get('milestones', [int(training_config.epochs*0.3), int(training_config.epochs*0.6)])))
        self.warmup_epochs = int(self.config.get('training.warmup_epochs', 0))
        self.base_lr = float(training_config.learning_rate)
    
    def _setup_monitoring(self):
        """Setup monitoring and logging"""
        # Create directories
        self.checkpoint_dir = Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path("logs/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if configured
        if self.config.get('monitoring.use_wandb', False):
            # Create a more descriptive experiment name
            experiment_name = f"anomaly_detection_{self.config.get('model.anomaly_classifier.backbone', 'unknown')}_{int(time.time())}"
            
            wandb.init(
                project="anomaly-detection-cctv",
                name=experiment_name,
                config={
                    "model_architecture": self.config.get('model.architecture'),
                    "backbone": self.config.get('model.anomaly_classifier.backbone'),
                    "num_classes": self.config.get('model.anomaly_classifier.num_classes'),
                    "batch_size": self.config.get('dataset.batch_size'),
                    "learning_rate": self.config.get('training.learning_rate'),
                    "epochs": self.config.get('training.epochs'),
                    "image_size": self.config.get('dataset.image_size'),
                    "optimizer": self.config.get('training.optimizer'),
                    "yolo_version": self.config.get('model.yolo.version'),
                },
                tags=["multi-camera", "anomaly-detection", "hybrid-model"]
            )
            
            # Watch model for gradient tracking
            wandb.watch(self.model, log="all", log_freq=100)
            
            self.logger.info(f"✅ Weights & Biases initialized: {experiment_name}")
            self.logger.info(f"🔗 Dashboard: {wandb.run.url}")
        else:
            self.logger.info("ℹ️  Weights & Biases logging disabled")

    def _init_ema(self):
        """Initialize EMA state dict from current model."""
        self.ema_state = {k: v.detach().clone() for k, v in self.model.state_dict().items() if v.dtype.is_floating_point}

    def _update_ema(self):
        """Update EMA with current model weights."""
        if not self.use_ema or self.ema_state is None:
            return
        with torch.no_grad():
            msd = self.model.state_dict()
            for k, v in self.ema_state.items():
                if k in msd:
                    v.mul_(self.ema_decay).add_(msd[k].detach(), alpha=1.0 - self.ema_decay)
    
    def _calculate_class_frequencies(self) -> List[int]:
        """Calculate class frequencies from dataset analysis"""
        try:
            # Load dataset analysis results
            import json
            with open('data/processed/dataset_analysis_report.json', 'r') as f:
                analysis = json.load(f)
            
            class_names = self.config.get('dataset.classes')
            class_freq = []
            
            for class_name in class_names:
                if class_name in analysis['class_distribution']:
                    class_freq.append(analysis['class_distribution'][class_name])
                else:
                    # Default frequency if not found
                    class_freq.append(1000)
            
            return class_freq
        except:
            # Fallback to default frequencies if analysis not available
            return [50000] * len(self.config.get('dataset.classes'))
    
    def _calculate_class_weights(self, class_frequencies: List[int]) -> torch.Tensor:
        """Calculate class weights for balanced training"""
        total_samples = sum(class_frequencies)
        num_classes = len(class_frequencies)
        
        # Calculate inverse frequency weights
        weights = []
        for freq in class_frequencies:
            weight = total_samples / (num_classes * freq)
            weights.append(weight)
        
        # Normalize weights
        weights = torch.FloatTensor(weights)
        weights = weights / weights.sum() * len(weights)
        
        return weights.to(self.device)
    
    def _mixup_criterion(self, pred, y_a, y_b, lam):
        """Mixed criterion for mixup/cutmix"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

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
            
            # Apply advanced augmentation randomly
            use_augmentation = np.random.rand() < 0.5  # 50% chance
            if use_augmentation and (self.use_mixup or self.use_cutmix):
                # Choose between mixup and cutmix
                if self.use_mixup and self.use_cutmix:
                    use_mixup = np.random.rand() < 0.5
                else:
                    use_mixup = self.use_mixup
                
                if use_mixup:
                    mixed_images, labels_a, labels_b, lam = self.mixup(images, labels)
                else:
                    mixed_images, labels_a, labels_b, lam = self.cutmix(images, labels)
                
                # Forward pass with mixed data (AMP)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    results = self.model(mixed_images)
                    logits = results['anomaly_logits']
                    if self.use_logit_adjust:
                        logits = logits + self.logit_bias.unsqueeze(0)
                    confidences = results['final_scores'].squeeze()
                    
                    # Enhanced mixup loss calculation
                    if self.use_contrastive:
                        fusion_features = results['fusion_features']
                        loss = self._mixup_criterion_contrastive(logits, fusion_features, labels_a, labels_b, lam)
                    else:
                        loss = self._mixup_criterion(logits, labels_a, labels_b, lam)
                
                # For metrics, use original data predictions
                with torch.no_grad():
                    original_results = self.model(images)
                    original_logits = original_results['anomaly_logits']
                    if self.use_logit_adjust:
                        original_logits = original_logits + self.logit_bias.unsqueeze(0)
                    original_confidences = original_results['final_scores'].squeeze()
                    predictions = torch.argmax(original_logits, dim=1)
                    confidences = original_confidences  # Use original confidences for metrics
                    
            else:
                # Standard forward pass with enhanced model outputs
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    results = self.model(images)
                    logits = results['anomaly_logits']
                    if self.use_logit_adjust:
                        logits = logits + self.logit_bias.unsqueeze(0)
                    confidences = results['final_scores'].squeeze()
                    
                    # Enhanced loss calculation for contrastive learning
                    if self.use_contrastive:
                        fusion_features = results['fusion_features']
                        loss = self.criterion(logits, fusion_features, labels)
                    else:
                        loss = self.criterion(logits, labels)
                        
                predictions = torch.argmax(logits, dim=1)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping for stable training
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA after optimizer step
            self._update_ema()
            
            # Update metrics (always use original predictions for metrics)
            self.train_metrics.update(predictions, labels, confidences)
            
            running_loss += loss.item()
            
            # Log to wandb every 100 batches for real-time monitoring
            if (self.config.get('monitoring.use_wandb', False) and 
                wandb.run is not None and 
                batch_idx % 100 == 0):
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_avg_loss': running_loss / (batch_idx + 1),
                    'epoch': self.current_epoch + 1,
                    'batch': batch_idx,
                    'global_step': self.current_epoch * num_batches + batch_idx
                })
            
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
                
                # Forward pass (AMP)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    results = self.model(images)
                    logits = results['anomaly_logits']
                    if self.use_logit_adjust:
                        logits = logits + self.logit_bias.unsqueeze(0)
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
            'ema_state_dict': self.ema_state,
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
    
    def _mixup_criterion_contrastive(self, logits: torch.Tensor, features: torch.Tensor, 
                                   labels_a: torch.Tensor, labels_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Mixup criterion for contrastive loss"""
        if self.use_contrastive and hasattr(self.criterion, 'forward'):
            # Use a weighted combination for mixup with contrastive loss
            loss_a = self.criterion(logits, features, labels_a)
            loss_b = self.criterion(logits, features, labels_b)
            return lam * loss_a + (1 - lam) * loss_b
        else:
            # Fallback to standard mixup
            return self._mixup_criterion(logits, labels_a, labels_b, lam)
    
    def _mixup_criterion(self, logits: torch.Tensor, labels_a: torch.Tensor, 
                        labels_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Standard mixup criterion"""
        if hasattr(self.criterion, 'forward') and 'CombinedLoss' in str(type(self.criterion)):
            # For combined loss, we need dummy features
            batch_size = logits.size(0)
            dummy_features = torch.zeros(batch_size, logits.size(1)).to(logits.device)
            loss_a = self.criterion(logits, dummy_features, labels_a)
            loss_b = self.criterion(logits, dummy_features, labels_b)
        else:
            # Standard loss functions
            if hasattr(self.criterion, '__call__'):
                loss_a = self.criterion(logits, labels_a)
                loss_b = self.criterion(logits, labels_b)
            else:
                # Fallback to CrossEntropyLoss
                loss_a = F.cross_entropy(logits, labels_a)
                loss_b = F.cross_entropy(logits, labels_b)
        
        return lam * loss_a + (1 - lam) * loss_b
    
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

            # Warmup learning rate
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_lr = self.base_lr * float(epoch + 1) / float(self.warmup_epochs)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            # Progressive resizing: update loaders at milestones
            if self.progressive_sizes is not None:
                milestones = self.pr_epochs
                new_size = None
                if epoch == 0:
                    new_size = self.progressive_sizes[0]
                elif epoch == milestones[0]:
                    new_size = self.progressive_sizes[1]
                elif epoch == milestones[1]:
                    new_size = self.progressive_sizes[2]
                if new_size is not None:
                    self.logger.info(f"🔁 Progressive resizing -> {new_size}")
                    # Update config and rebuild loaders
                    self.config.set('dataset.image_size', list(new_size))
                    splits = load_data_splits("data/processed/data_splits.json")
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
                    # Avoid stepping scheduler during warmup
                    if not (self.warmup_epochs > 0 and epoch < self.warmup_epochs):
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
            
            # Check for best model and early stopping
            current_metric = val_metrics['f1_macro']
            is_best = current_metric > self.best_f1
            
            if is_best:
                self.best_f1 = current_metric
                self.best_accuracy = val_metrics['accuracy']
                self.best_metric_value = current_metric
                self.early_stopping_counter = 0
                self.logger.info(f"New best model saved with F1: {self.best_f1:.4f}")
            else:
                # Check for early stopping
                if current_metric <= self.best_metric_value + self.early_stopping_min_delta:
                    self.early_stopping_counter += 1
                else:
                    self.early_stopping_counter = 0
                    self.best_metric_value = max(self.best_metric_value, current_metric)
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {self.early_stopping_patience} epochs of no improvement")
                break
            
            # Log to wandb if initialized
            if self.config.get('monitoring.use_wandb', False) and wandb.run is not None:
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
        """Evaluate on test set with Test-Time Augmentation for maximum accuracy"""
        self.logger.info("Evaluating on test set with Test-Time Augmentation...")
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / "best_model.pth", weights_only=False)
        # Prefer EMA weights if available
        if self.use_ema and best_checkpoint.get('ema_state_dict') is not None:
            self.logger.info("Using EMA weights for evaluation")
            self.model.load_state_dict(best_checkpoint['ema_state_dict'], strict=False)
        else:
            self.model.load_state_dict(best_checkpoint['state_dict'])
        
        test_metrics = MetricsCalculator(
            num_classes=len(self.config.get('dataset.classes')),
            class_names=self.config.get('dataset.classes')
        )
        
        self.model.eval()
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.data_loaders['test'], desc="Testing with TTA"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Test-Time Augmentation: multiple predictions and average
                tta_logits = []
                tta_confidences = []
                
                # Original images
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    results = self.model(images)
                tta_logits.append(results['anomaly_logits'])
                tta_confidences.append(results['final_scores'])
                
                # Horizontal flip
                flipped_images = torch.flip(images, dims=[3])
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    results = self.model(flipped_images)
                tta_logits.append(results['anomaly_logits'])
                tta_confidences.append(results['final_scores'])
                
                # Vertical flip
                vflipped_images = torch.flip(images, dims=[2])
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    results = self.model(vflipped_images)
                tta_logits.append(results['anomaly_logits'])
                tta_confidences.append(results['final_scores'])
                
                # Both flips
                both_flipped = torch.flip(torch.flip(images, dims=[2]), dims=[3])
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    results = self.model(both_flipped)
                tta_logits.append(results['anomaly_logits'])
                tta_confidences.append(results['final_scores'])
                
                # Average TTA predictions
                avg_logits = torch.stack(tta_logits).mean(dim=0)
                if self.use_logit_adjust:
                    avg_logits = avg_logits + self.logit_bias.unsqueeze(0)
                avg_confidences = torch.stack(tta_confidences).mean(dim=0).squeeze()
                
                predictions = torch.argmax(avg_logits, dim=1)
                test_metrics.update(predictions, labels, avg_confidences)
        
        final_metrics = test_metrics.compute()
        
        self.logger.info("Test Set Results:")
        self.logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
        self.logger.info(f"F1 Macro: {final_metrics['f1_macro']:.4f}")
        self.logger.info(f"F1 Weighted: {final_metrics['f1_weighted']:.4f}")
        
        # Log final test results to wandb
        if self.config.get('monitoring.use_wandb', False) and wandb.run is not None:
            wandb.log({
                'test_accuracy': final_metrics['accuracy'],
                'test_f1_macro': final_metrics['f1_macro'],
                'test_f1_weighted': final_metrics['f1_weighted'],
                'test_precision_macro': final_metrics.get('precision_macro', 0),
                'test_recall_macro': final_metrics.get('recall_macro', 0),
                'final_epoch': 'test_evaluation'
            })
            
            # Log a summary table of all results
            wandb.summary.update({
                "best_val_accuracy": self.best_accuracy,
                "best_val_f1": self.best_f1,
                "final_test_accuracy": final_metrics['accuracy'],
                "final_test_f1": final_metrics['f1_macro'],
                "total_epochs_trained": len(self.training_history)
            })
        
        # Save test results
        # Ensure JSON serializable types
        def to_json_serializable(obj):
            """Convert numpy/torch types to Python native types for JSON serialization"""
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().item() if obj.numel() == 1 else obj.cpu().detach().numpy().tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_json_serializable(item) for item in obj]
            else:
                return obj

        test_results = {
            'test_metrics': to_json_serializable(final_metrics),
            'best_validation_f1': float(self.best_f1),
            'best_validation_accuracy': float(self.best_accuracy),
            'training_history': {k: [float(x) if isinstance(x, (float, int)) else float(x.item()) if hasattr(x, 'item') else float(x) if isinstance(x, np.floating) else x for x in v] for k, v in self.training_history.items()}
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