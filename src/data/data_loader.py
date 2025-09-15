"""
Professional Data Loader for Multi-Camera Anomaly Detection
==========================================================

Advanced PyTorch data loader with professional augmentation strategies,
class balancing, and multi-camera data handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import random
from sklearn.utils.class_weight import compute_class_weight


class AnomalyDataset(Dataset):
    """Professional dataset class for anomaly detection with advanced augmentation"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        class_names: List[str],
        transform=None,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0
    ):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            class_names: List of class names
            transform: Albumentations transform pipeline
            image_size: Target image size (width, height)
            is_training: Whether this is training dataset
            mixup_alpha: Alpha parameter for mixup augmentation
            cutmix_alpha: Alpha parameter for cutmix augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        
        # Create label to class name mapping
        self.label_to_class = {i: name for i, name in enumerate(class_names)}
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get dataset item with advanced augmentation
        
        Returns:
            image: Augmented image tensor
            label: Class label
            metadata: Additional metadata
        """
        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image using OpenCV for better performance
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Handle corrupted images
            if image is None:
                # Return a black image with label
                image = np.zeros((64, 64, 3), dtype=np.uint8)
                
        except Exception as e:
            # Fallback for corrupted images
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            print(f"Warning: Failed to load {image_path}: {e}")
        
        # Resize to target size
        image = cv2.resize(image, self.image_size)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Create metadata
        metadata = {
            'image_path': str(image_path),
            'class_name': self.label_to_class[label],
            'original_size': image.shape[-2:] if hasattr(image, 'shape') else self.image_size
        }
        
        return image, label, metadata
    
    def apply_mixup(self, batch_images: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation to batch"""
        if not self.is_training or self.mixup_alpha <= 0:
            return batch_images, batch_labels, batch_labels, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Create random permutation
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        
        # Return mixed images and both sets of labels
        return mixed_images, batch_labels, batch_labels[index], lam
    
    def apply_cutmix(self, batch_images: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply cutmix augmentation to batch"""
        if not self.is_training or self.cutmix_alpha <= 0:
            return batch_images, batch_labels, batch_labels, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size)
        
        # Get image dimensions
        _, _, H, W = batch_images.shape
        
        # Sample bounding box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        batch_images[:, :, bby1:bby2, bbx1:bbx2] = batch_images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        return batch_images, batch_labels, batch_labels[index], lam


class AugmentationFactory:
    """Factory for creating professional augmentation pipelines"""
    
    @staticmethod
    def create_training_transforms(
        image_size: Tuple[int, int] = (224, 224),
        severity: str = 'medium'
    ) -> A.Compose:
        """
        Create training augmentation pipeline
        
        Args:
            image_size: Target image size
            severity: Augmentation severity ('light', 'medium', 'heavy')
        """
        if severity == 'light':
            transforms = [
                A.Resize(image_size[1], image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        elif severity == 'medium':
            transforms = [
                A.Resize(image_size[1], image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.3
                ),
                A.GaussNoise(noise_scale_factor=0.1, p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        else:  # heavy
            transforms = [
                A.Resize(image_size[1], image_size[0]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.3,
                    rotate_limit=20,
                    p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 80.0)),
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.3
                ),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        
        return A.Compose(transforms)
    
    @staticmethod
    def create_validation_transforms(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
        """Create validation/test augmentation pipeline"""
        return A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def create_tta_transforms(image_size: Tuple[int, int] = (224, 224)) -> List[A.Compose]:
        """Create test-time augmentation transforms"""
        tta_transforms = []
        
        # Original
        tta_transforms.append(A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Horizontal flip
        tta_transforms.append(A.Compose([
            A.Resize(image_size[1], image_size[0]),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Multiple scales
        for scale in [0.9, 1.1]:
            size = (int(image_size[0] * scale), int(image_size[1] * scale))
            tta_transforms.append(A.Compose([
                A.Resize(size[1], size[0]),
                A.CenterCrop(image_size[1], image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        return tta_transforms


class DataLoaderFactory:
    """Factory for creating professional data loaders with class balancing"""
    
    @staticmethod
    def create_balanced_sampler(labels: List[int], num_classes: int) -> WeightedRandomSampler:
        """Create weighted sampler for class balancing"""
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=labels
        )
        
        # Create sample weights
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    @staticmethod
    def create_data_loaders(
        train_paths: List[str],
        train_labels: List[int],
        val_paths: List[str],
        val_labels: List[int],
        class_names: List[str],
        config: Dict,
        test_paths: Optional[List[str]] = None,
        test_labels: Optional[List[int]] = None
    ) -> Dict[str, DataLoader]:
        """
        Create professional data loaders with augmentation and balancing
        
        Args:
            train_paths: Training image paths
            train_labels: Training labels
            val_paths: Validation image paths
            val_labels: Validation labels
            class_names: List of class names
            config: Configuration dictionary
            test_paths: Optional test image paths
            test_labels: Optional test labels
            
        Returns:
            Dictionary of data loaders
        """
        dataset_config = config['dataset']
        training_config = config['training']
        
        image_size = tuple(dataset_config['image_size'])
        batch_size = dataset_config['batch_size']
        num_workers = dataset_config['num_workers']
        
        # Create augmentation transforms
        aug_config = training_config['augmentation']
        if aug_config['enabled']:
            train_transform = AugmentationFactory.create_training_transforms(
                image_size=image_size,
                severity='medium'
            )
        else:
            train_transform = AugmentationFactory.create_validation_transforms(image_size)
        
        val_transform = AugmentationFactory.create_validation_transforms(image_size)
        
        # Create datasets
        train_dataset = AnomalyDataset(
            image_paths=train_paths,
            labels=train_labels,
            class_names=class_names,
            transform=train_transform,
            image_size=image_size,
            is_training=True,
            mixup_alpha=aug_config.get('mixup_alpha', 0.2),
            cutmix_alpha=aug_config.get('cutmix_alpha', 1.0)
        )
        
        val_dataset = AnomalyDataset(
            image_paths=val_paths,
            labels=val_labels,
            class_names=class_names,
            transform=val_transform,
            image_size=image_size,
            is_training=False
        )
        
        # Create samplers for class balancing
        balance_method = training_config['class_balance']['method']
        if balance_method == 'weighted_sampling':
            train_sampler = DataLoaderFactory.create_balanced_sampler(
                train_labels, len(class_names)
            )
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        loaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # Add test loader if provided
        if test_paths is not None and test_labels is not None:
            test_dataset = AnomalyDataset(
                image_paths=test_paths,
                labels=test_labels,
                class_names=class_names,
                transform=val_transform,
                image_size=image_size,
                is_training=False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False
            )
            
            loaders['test'] = test_loader
        
        return loaders
    
    @staticmethod
    def create_inference_loader(
        image_paths: List[str],
        class_names: List[str],
        config: Dict,
        use_tta: bool = False
    ) -> DataLoader:
        """Create data loader for inference"""
        dataset_config = config['dataset']
        inference_config = config['inference']
        
        image_size = tuple(dataset_config['image_size'])
        batch_size = inference_config['batch_size']
        
        if use_tta:
            # Use TTA transforms
            transforms = AugmentationFactory.create_tta_transforms(image_size)
            # For TTA, we'll need to modify the dataset to return multiple augmented versions
            transform = transforms[0]  # Use first transform for now
        else:
            transform = AugmentationFactory.create_validation_transforms(image_size)
        
        # Create dummy labels for inference
        dummy_labels = [0] * len(image_paths)
        
        dataset = AnomalyDataset(
            image_paths=image_paths,
            labels=dummy_labels,
            class_names=class_names,
            transform=transform,
            image_size=image_size,
            is_training=False
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataset_config['num_workers'],
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )


def load_data_splits(splits_path: str) -> Dict:
    """Load data splits from JSON file"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return splits


def main():
    """Test data loader functionality"""
    from src.utils.config import get_config
    
    config = get_config()
    
    # Load data splits
    try:
        splits = load_data_splits("data/processed/data_splits.json")
        
        # Create data loaders
        loaders = DataLoaderFactory.create_data_loaders(
            train_paths=splits['train']['paths'],
            train_labels=splits['train']['labels'],
            val_paths=splits['val']['paths'],
            val_labels=splits['val']['labels'],
            class_names=config.get('dataset.classes'),
            config=config.config,
            test_paths=splits['test']['paths'],
            test_labels=splits['test']['labels']
        )
        
        print("✅ Data loaders created successfully!")
        print(f"Train batches: {len(loaders['train'])}")
        print(f"Val batches: {len(loaders['val'])}")
        print(f"Test batches: {len(loaders['test'])}")
        
        # Test a batch
        for batch_idx, (images, labels, metadata) in enumerate(loaders['train']):
            print(f"Batch {batch_idx}: Images shape: {images.shape}, Labels: {labels.shape}")
            if batch_idx >= 2:  # Test first 3 batches
                break
                
    except FileNotFoundError:
        print("❌ Data splits not found. Please run analyze_dataset.py first.")


if __name__ == "__main__":
    main()