"""
Dataset Analysis and Preprocessing Module
========================================

Professional data analysis for the UCF-Crime dataset with class imbalance handling,
data augmentation, and multi-camera data preparation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image
import json


class UCFDatasetAnalyzer:
    """Professional dataset analyzer for UCF-Crime dataset"""
    
    def __init__(self, data_root: str, config: Dict):
        """
        Initialize dataset analyzer
        
        Args:
            data_root: Root directory containing raw data
            config: Configuration dictionary
        """
        self.data_root = Path(data_root)
        self.config = config
        self.classes = config['dataset']['classes']
        self.class_distribution = {}
        self.statistics = {}
        
    def analyze_dataset(self) -> Dict:
        """Comprehensive dataset analysis"""
        print("🔍 Starting comprehensive dataset analysis...")
        
        analysis_results = {
            'class_distribution': self._analyze_class_distribution(),
            'image_statistics': self._analyze_image_statistics(),
            'class_balance': self._analyze_class_balance(),
            'data_quality': self._analyze_data_quality(),
            'recommendations': self._generate_recommendations()
        }
        
        self._save_analysis_report(analysis_results)
        self._generate_visualizations()
        
        return analysis_results
    
    def _analyze_class_distribution(self) -> Dict:
        """Analyze class distribution across train/test splits"""
        print("📊 Analyzing class distribution...")
        
        distribution = {}
        total_samples = 0
        
        # Check both Train and Test directories
        for split in ['Train', 'Test']:
            split_path = self.data_root / split
            if not split_path.exists():
                print(f"⚠️  Warning: {split} directory not found at {split_path}")
                continue
                
            distribution[split] = {}
            split_total = 0
            
            for class_name in self.classes:
                class_path = split_path / class_name
                if class_path.exists():
                    image_count = len(list(class_path.glob("*.png")))
                    distribution[split][class_name] = image_count
                    split_total += image_count
                else:
                    distribution[split][class_name] = 0
                    print(f"⚠️  Warning: {class_name} directory not found in {split}")
            
            distribution[split]['total'] = split_total
            total_samples += split_total
        
        # Calculate overall percentages
        overall_distribution = {}
        for class_name in self.classes:
            total_class_count = sum(distribution[split].get(class_name, 0) for split in ['Train', 'Test'] if split in distribution)
            overall_distribution[class_name] = total_class_count
        
        distribution_pct = {
            class_name: (count / total_samples * 100) if total_samples > 0 else 0
            for class_name, count in overall_distribution.items()
        }
        
        self.class_distribution = overall_distribution
        
        return {
            'split_distribution': distribution,
            'overall_distribution': overall_distribution,
            'percentages': distribution_pct,
            'total_samples': total_samples,
            'num_classes': len([c for c in overall_distribution.values() if c > 0])
        }
    
    def _analyze_image_statistics(self) -> Dict:
        """Analyze image properties (size, format, quality)"""
        print("🖼️  Analyzing image statistics...")
        
        sample_size = 1000  # Sample images for analysis
        image_stats = {
            'sizes': [],
            'aspects': [],
            'channels': [],
            'file_sizes': [],
            'corrupted_count': 0
        }
        
        sample_count = 0
        # Sample from both Train and Test directories
        for split in ['Train', 'Test']:
            split_path = self.data_root / split
            if not split_path.exists():
                continue
                
            for class_name in self.classes:
                if sample_count >= sample_size:
                    break
                    
                class_path = split_path / class_name
                if not class_path.exists():
                    continue
                    
                images_per_class = min(sample_size // (len(self.classes) * 2), 50)  # 2 for Train/Test
                image_files = list(class_path.glob("*.png"))[:images_per_class]
                
                for img_path in image_files:
                    if sample_count >= sample_size:
                        break
                        
                    try:
                        # Load image
                        img = cv2.imread(str(img_path))
                        if img is None:
                            image_stats['corrupted_count'] += 1
                            continue
                        
                        h, w, c = img.shape
                        image_stats['sizes'].append((w, h))
                        image_stats['aspects'].append(w / h)
                        image_stats['channels'].append(c)
                        image_stats['file_sizes'].append(img_path.stat().st_size)
                        
                        sample_count += 1
                        
                    except Exception as e:
                        image_stats['corrupted_count'] += 1
                        print(f"Error processing {img_path}: {e}")
                        
            if sample_count >= sample_size:
                break
        
        # Calculate statistics
        if image_stats['sizes']:
            widths = [s[0] for s in image_stats['sizes']]
            heights = [s[1] for s in image_stats['sizes']]
            
            stats_summary = {
                'sample_count': sample_count,
                'width_stats': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths)
                },
                'height_stats': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights)
                },
                'aspect_ratio_stats': {
                    'mean': np.mean(image_stats['aspects']),
                    'std': np.std(image_stats['aspects'])
                },
                'file_size_stats': {
                    'mean_kb': np.mean(image_stats['file_sizes']) / 1024,
                    'std_kb': np.std(image_stats['file_sizes']) / 1024
                },
                'corrupted_images': image_stats['corrupted_count'],
                'unique_sizes': len(set(image_stats['sizes']))
            }
        else:
            stats_summary = {'error': 'No valid images found'}
        
        return stats_summary
    
    def _analyze_class_balance(self) -> Dict:
        """Analyze class imbalance and suggest handling strategies"""
        print("⚖️  Analyzing class balance...")
        
        if not self.class_distribution:
            return {'error': 'No class distribution data available'}
        
        counts = list(self.class_distribution.values())
        valid_counts = [c for c in counts if c > 0]
        
        if not valid_counts:
            return {'error': 'No valid classes found'}
        
        # Calculate imbalance metrics
        max_count = max(valid_counts)
        min_count = min(valid_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Calculate class weights for balanced training
        class_labels = []
        for class_name, count in self.class_distribution.items():
            class_labels.extend([class_name] * count)
        
        unique_classes = list(set(class_labels))
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=class_labels
        )
        
        class_weights_dict = dict(zip(unique_classes, class_weights))
        
        # Identify minority and majority classes
        sorted_classes = sorted(
            self.class_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        balance_analysis = {
            'imbalance_ratio': imbalance_ratio,
            'imbalance_severity': self._classify_imbalance_severity(imbalance_ratio),
            'class_weights': class_weights_dict,
            'majority_classes': sorted_classes[:3],
            'minority_classes': sorted_classes[-3:],
            'suggested_strategies': self._suggest_balance_strategies(imbalance_ratio)
        }
        
        return balance_analysis
    
    def _classify_imbalance_severity(self, ratio: float) -> str:
        """Classify imbalance severity"""
        if ratio < 2:
            return "Balanced"
        elif ratio < 5:
            return "Moderate Imbalance"
        elif ratio < 10:
            return "High Imbalance"
        else:
            return "Extreme Imbalance"
    
    def _suggest_balance_strategies(self, ratio: float) -> List[str]:
        """Suggest strategies for handling class imbalance"""
        strategies = []
        
        if ratio > 2:
            strategies.append("Use class weights in loss function")
            strategies.append("Apply focal loss for hard examples")
        
        if ratio > 5:
            strategies.append("Use stratified sampling")
            strategies.append("Apply data augmentation to minority classes")
        
        if ratio > 10:
            strategies.append("Consider SMOTE or other oversampling")
            strategies.append("Ensemble methods with balanced subsets")
            strategies.append("Cost-sensitive learning")
        
        return strategies
    
    def _analyze_data_quality(self) -> Dict:
        """Analyze data quality issues"""
        print("🔍 Analyzing data quality...")
        
        quality_metrics = {
            'empty_directories': [],
            'very_small_classes': [],
            'potential_duplicates': 0,
            'naming_consistency': True,
            'file_format_consistency': True
        }
        
        # Check for empty or very small classes
        for class_name, count in self.class_distribution.items():
            if count == 0:
                quality_metrics['empty_directories'].append(class_name)
            elif count < 100:  # Threshold for very small classes
                quality_metrics['very_small_classes'].append((class_name, count))
        
        # Check naming consistency (simplified check)
        sample_files = []
        for split in ['Train', 'Test']:
            split_path = self.data_root / split
            if not split_path.exists():
                continue
                
            for class_name in self.classes[:3]:  # Sample first 3 classes
                class_path = split_path / class_name
                if class_path.exists():
                    files = list(class_path.glob("*.png"))[:10]
                    sample_files.extend(files)
        
        # Basic naming pattern check
        naming_patterns = set()
        for file_path in sample_files:
            # Extract naming pattern (simplified)
            name = file_path.stem
            pattern = ''.join(['x' if c.isdigit() else c for c in name])
            naming_patterns.add(pattern)
        
        quality_metrics['naming_patterns'] = list(naming_patterns)
        quality_metrics['naming_consistency'] = len(naming_patterns) <= 2
        
        return quality_metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Based on class distribution
        if hasattr(self, 'class_distribution'):
            total_samples = sum(self.class_distribution.values())
            
            if total_samples < 100000:
                recommendations.append("Consider data augmentation to increase dataset size")
            
            # Check for very small classes
            small_classes = [name for name, count in self.class_distribution.items() if count < 1000]
            if small_classes:
                recommendations.append(f"Apply heavy augmentation to small classes: {small_classes}")
        
        # Training recommendations
        recommendations.extend([
            "Use progressive resizing: start with 64x64, then 128x128, finally 224x224",
            "Apply mixup and cutmix augmentation for better generalization",
            "Use test-time augmentation (TTA) for final inference",
            "Implement cross-validation for robust model evaluation",
            "Consider multi-scale training for better object detection"
        ])
        
        return recommendations
    
    def _save_analysis_report(self, analysis: Dict):
        """Save analysis report to JSON"""
        report_path = Path("data/processed/dataset_analysis_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"📄 Analysis report saved to {report_path}")
    
    def _generate_visualizations(self):
        """Generate visualization plots"""
        print("📈 Generating visualizations...")
        
        # Create plots directory
        plots_dir = Path("data/processed/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Class distribution plot
        plt.figure(figsize=(15, 8))
        
        classes = list(self.class_distribution.keys())
        counts = list(self.class_distribution.values())
        
        # Create bar plot
        plt.subplot(2, 2, 1)
        bars = plt.bar(range(len(classes)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Pie chart for class percentages
        plt.subplot(2, 2, 2)
        valid_classes = [(name, count) for name, count in self.class_distribution.items() if count > 0]
        if valid_classes:
            labels, sizes = zip(*valid_classes)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.title('Class Distribution (Percentage)')
        
        # Class imbalance visualization
        plt.subplot(2, 2, 3)
        sorted_counts = sorted(counts, reverse=True)
        plt.plot(range(len(sorted_counts)), sorted_counts, marker='o', color='red')
        plt.xlabel('Class Rank')
        plt.ylabel('Number of Images')
        plt.title('Class Imbalance (Sorted)')
        plt.yscale('log')
        
        # Class balance heatmap
        plt.subplot(2, 2, 4)
        if len(classes) > 1:
            # Create a matrix showing relative class sizes
            matrix = np.array(counts).reshape(1, -1)
            sns.heatmap(matrix, xticklabels=classes, yticklabels=['Count'], 
                       annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Class Count Heatmap')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {plots_dir}")
    
    def create_stratified_splits(self, val_size: float = 0.15) -> Dict:
        """Create train/validation/test splits from existing Train/Test directories"""
        print("🔄 Creating data splits from Train/Test directories...")
        
        # Load training data (from Train directory)
        train_paths = []
        train_labels = []
        
        train_dir = self.data_root / 'Train'
        if train_dir.exists():
            for class_idx, class_name in enumerate(self.classes):
                class_path = train_dir / class_name
                if class_path.exists():
                    class_images = list(class_path.glob("*.png"))
                    train_paths.extend(class_images)
                    train_labels.extend([class_idx] * len(class_images))
        
        # Load test data (from Test directory)
        test_paths = []
        test_labels = []
        
        test_dir = self.data_root / 'Test'
        if test_dir.exists():
            for class_idx, class_name in enumerate(self.classes):
                class_path = test_dir / class_name
                if class_path.exists():
                    class_images = list(class_path.glob("*.png"))
                    test_paths.extend(class_images)
                    test_labels.extend([class_idx] * len(class_images))
        
        # Split training data into train and validation
        if train_paths and val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                train_paths, train_labels, test_size=val_size, 
                stratify=train_labels, random_state=42
            )
        else:
            X_train, y_train = train_paths, train_labels
            X_val, y_val = [], []
        
        splits = {
            'train': {'paths': X_train, 'labels': y_train},
            'val': {'paths': X_val, 'labels': y_val},
            'test': {'paths': test_paths, 'labels': test_labels}
        }
        
        # Save splits
        splits_path = Path("data/processed/data_splits.json")
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        
        splits_serializable = {
            split_name: {
                'paths': [str(p) for p in split_data['paths']],
                'labels': split_data['labels']
            }
            for split_name, split_data in splits.items()
        }
        
        with open(splits_path, 'w') as f:
            json.dump(splits_serializable, f, indent=2)
        
        print(f"✅ Data splits saved to {splits_path}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(test_paths)}")
        
        return splits


def main():
    """Main function to run dataset analysis"""
    from src.utils.config import get_config
    
    # Load configuration
    config = get_config()
    
    # Initialize analyzer
    analyzer = UCFDatasetAnalyzer(
        data_root="data/raw",
        config=config.config
    )
    
    # Run analysis
    results = analyzer.analyze_dataset()
    
    # Create data splits
    splits = analyzer.create_stratified_splits()
    
    # Print summary
    print("\n" + "="*60)
    print("📋 DATASET ANALYSIS SUMMARY")
    print("="*60)
    
    class_dist = results['class_distribution']
    print(f"Total samples: {class_dist['total_samples']:,}")
    print(f"Number of classes: {class_dist['num_classes']}")
    
    balance = results['class_balance']
    print(f"Class imbalance ratio: {balance['imbalance_ratio']:.2f}")
    print(f"Imbalance severity: {balance['imbalance_severity']}")
    
    print(f"\nTop 3 largest classes:")
    for class_name, count in balance['majority_classes']:
        print(f"  {class_name}: {count:,} images")
    
    print(f"\nBottom 3 smallest classes:")
    for class_name, count in balance['minority_classes']:
        print(f"  {class_name}: {count:,} images")
    
    print(f"\n📋 Recommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n✅ Analysis complete! Check 'data/processed/' for detailed reports and plots.")


if __name__ == "__main__":
    main()