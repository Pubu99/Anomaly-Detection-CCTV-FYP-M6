"""
Enhanced Main Training Script - Technical Report Implementation
=============================================================

This script implements the complete pipeline from the technical report with modern enhancements:
- CNN-LSTM architecture with InceptionV3 + multi-layer LSTM
- Temporal sequence processing (32 frames)
- Advanced loss functions for extreme class imbalance
- Feature pre-computation and caching
- OpenVINO optimization support
- Real-time monitoring and visualization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
import json
import argparse
from typing import Dict, Optional

# Import enhanced modules
from src.models.enhanced_temporal_model import EnhancedTemporalAnomalyModel, create_enhanced_temporal_model
from src.training.enhanced_temporal_train import EnhancedTemporalTrainer
from src.data.enhanced_data_preprocessing import DataPreprocessor
from src.inference.enhanced_real_time_inference import EnhancedRealTimeInference, CameraStream
from src.utils.model_optimization import optimize_model_for_inference
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger


class TechnicalReportPipeline:
    """
    Complete pipeline implementing the technical report approach
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_app_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance optimization (enabled by default)
        self.use_optimization = True
        
        # Pipeline components
        self.preprocessor = None
        self.trainer = None
        self.model = None
        self.inference_engine = None
        
        # Paths
        self.model_dir = Path('models')
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        self.optimized_dir = self.model_dir / 'optimized'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimized_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("🚀 Technical Report Pipeline Initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model Architecture: CNN-LSTM (InceptionV3 + Multi-layer LSTM)")
        self.logger.info(f"Sequence Length: 32 frames")
        self.logger.info(f"Frame Sampling: Every 2nd frame")
        
    def prepare_data(self):
        """
        Prepare data following technical report approach
        """
        self.logger.info("📊 Preparing data with enhanced preprocessing...")
        
        # Initialize data preprocessor
        self.preprocessor = DataPreprocessor(self.config)
        
        # Create data loaders with caching
        self.train_loader, self.val_loader, self.test_loader = self.preprocessor.create_data_loaders(
            use_cache=True,
            cache_dir='data/processed/cache'
        )
        
        self.logger.info("✅ Data preparation completed")
        
        # Print dataset statistics
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"  Training batches: {len(self.train_loader)}")
        self.logger.info(f"  Validation batches: {len(self.val_loader)}")
        self.logger.info(f"  Test batches: {len(self.test_loader)}")
    
    def train_model(self, resume_training: bool = False):
        """
        Train the enhanced temporal model
        """
        self.logger.info("🎯 Starting enhanced temporal model training...")
        
        # Initialize trainer with optimization setting
        if self.use_optimization:
            self.logger.info("🚀 PERFORMANCE OPTIMIZATION ENABLED - Expected 10-20x speedup!")
            self.logger.info("   • Feature pre-extraction and caching")
            self.logger.info("   • Mixed precision training") 
            self.logger.info("   • Optimized data pipeline")
        
        self.trainer = EnhancedTemporalTrainer(config_path=None, use_optimization=self.use_optimization)
        
        # Check for existing checkpoint
        checkpoint_path = self.checkpoint_dir / 'enhanced_temporal_best.pth'
        
        if resume_training and checkpoint_path.exists():
            self.logger.info(f"Resuming training from: {checkpoint_path}")
            # Load checkpoint and continue training
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.trainer.model.load_state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Start training
        start_time = time.time()
        self.model = self.trainer.train()
        training_time = time.time() - start_time
        
        self.logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        self.logger.info(f"Best validation F1: {self.trainer.best_val_f1:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        if self.model is None:
            # Load best model
            checkpoint_path = self.checkpoint_dir / 'enhanced_temporal_best.pth'
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model = create_enhanced_temporal_model(checkpoint['config'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
            else:
                raise ValueError("No trained model found. Train the model first.")
        
        self.logger.info("📈 Starting comprehensive model evaluation...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Evaluate on test set
        test_metrics = self._evaluate_on_dataset(self.test_loader, "Test")
        
        # Save evaluation results
        results = {
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_f1_macro': test_metrics['f1_macro'],
            'per_class_metrics': test_metrics['per_class_metrics'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'evaluation_time': time.time()
        }
        
        results_path = self.checkpoint_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("✅ Model evaluation completed")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
        self.logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        
        return results
    
    def _evaluate_on_dataset(self, data_loader, dataset_name: str) -> Dict:
        """
        Evaluate model on a dataset
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        
        all_preds = []
        all_labels = []
        total_time = 0
        
        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(data_loader):
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                start_time = time.time()
                outputs = self.model(videos)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Get predictions
                _, predicted = torch.max(outputs['main'], 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        # Detailed classification report
        class_names = self.config['dataset']['classes']
        class_report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        self.logger.info(f"{dataset_name} Set Evaluation:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  F1 (Weighted): {f1_weighted:.4f}")
        self.logger.info(f"  F1 (Macro): {f1_macro:.4f}")
        self.logger.info(f"  Average inference time: {total_time/len(data_loader)*1000:.2f} ms/batch")
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'per_class_metrics': class_report,
            'confusion_matrix': conf_matrix,
            'avg_inference_time': total_time / len(data_loader)
        }
    
    def optimize_for_deployment(self, target_platform: str = "cpu"):
        """
        Optimize model for deployment using OpenVINO and other optimizations
        """
        self.logger.info(f"⚡ Optimizing model for {target_platform} deployment...")
        
        checkpoint_path = self.checkpoint_dir / 'enhanced_temporal_best.pth'
        
        if not checkpoint_path.exists():
            raise ValueError("No trained model found. Train the model first.")
        
        # Optimize model
        try:
            optimized_paths = optimize_model_for_inference(str(checkpoint_path), target_platform)
            
            self.logger.info("✅ Model optimization completed")
            self.logger.info("Optimized models:")
            for format_name, path in optimized_paths.items():
                self.logger.info(f"  {format_name.upper()}: {path}")
                
            return optimized_paths
            
        except Exception as e:
            self.logger.error(f"❌ Model optimization failed: {e}")
            raise
    
    def setup_real_time_inference(self, camera_configs: Optional[list] = None):
        """
        Setup real-time inference engine
        """
        self.logger.info("📹 Setting up real-time inference engine...")
        
        # Initialize inference engine
        self.inference_engine = EnhancedRealTimeInference()
        
        # Add default camera if none provided
        if camera_configs is None:
            camera_configs = [
                {
                    'camera_id': 'camera_001',
                    'source': 0,  # Default webcam
                    'fps': 30,
                    'confidence_threshold': 0.75
                }
            ]
        
        # Add camera streams
        for config in camera_configs:
            stream = CameraStream(
                camera_id=config['camera_id'],
                source=config['source'],
                fps=config.get('fps', 30),
                confidence_threshold=config.get('confidence_threshold', 0.75)
            )
            self.inference_engine.add_camera_stream(stream)
        
        self.logger.info(f"✅ Real-time inference setup completed with {len(camera_configs)} cameras")
        
        return self.inference_engine
    
    def run_real_time_demo(self, duration: int = 60):
        """
        Run real-time inference demo
        """
        if self.inference_engine is None:
            self.setup_real_time_inference()
        
        self.logger.info(f"🎥 Starting real-time inference demo for {duration} seconds...")
        
        try:
            # Start processing
            self.inference_engine.start_processing()
            
            # Monitor for specified duration
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(5)
                
                # Print system status
                status = self.inference_engine.get_system_status()
                self.logger.info(
                    f"Demo Status - FPS: {status['performance']['avg_fps']:.1f}, "
                    f"Inference Time: {status['performance']['avg_inference_time_ms']:.1f}ms, "
                    f"Pending Alerts: {status['alerts_pending']}"
                )
            
            self.logger.info("✅ Real-time inference demo completed")
            
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        finally:
            # Stop processing
            if self.inference_engine:
                self.inference_engine.stop_processing()
    
    def run_complete_pipeline(
        self,
        skip_training: bool = False,
        skip_optimization: bool = False,
        skip_demo: bool = False,
        target_platform: str = "cpu"
    ):
        """
        Run the complete pipeline from data preparation to deployment
        """
        self.logger.info("🔥 Starting Complete Technical Report Pipeline")
        self.logger.info("=" * 80)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Data preparation
            self.prepare_data()
            
            # Step 2: Model training
            if not skip_training:
                self.train_model()
            
            # Step 3: Model evaluation
            self.evaluate_model()
            
            # Step 4: Model optimization
            if not skip_optimization:
                self.optimize_for_deployment(target_platform)
            
            # Step 5: Real-time inference setup
            if not skip_demo:
                self.setup_real_time_inference()
                self.run_real_time_demo(duration=30)
            
            # Pipeline completed
            pipeline_time = time.time() - pipeline_start
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 COMPLETE PIPELINE EXECUTION SUCCESSFUL!")
            self.logger.info(f"⏱️  Total Pipeline Time: {pipeline_time/3600:.2f} hours")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            raise


def main():
    """
    Main function with command line arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced Anomaly Detection Pipeline - Technical Report Implementation')
    
    parser.add_argument('--mode', choices=['train', 'evaluate', 'optimize', 'inference', 'complete'], 
                       default='complete', help='Pipeline mode to run')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip-optimization', action='store_true', help='Skip model optimization')
    parser.add_argument('--skip-demo', action='store_true', help='Skip real-time demo')
    parser.add_argument('--target-platform', choices=['cpu', 'gpu', 'edge'], default='cpu', 
                       help='Target platform for optimization')
    parser.add_argument('--demo-duration', type=int, default=30, help='Demo duration in seconds')
    parser.add_argument('--disable-optimization', action='store_true', 
                       help='Disable performance optimization (use for debugging only)')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_app_logger()
    
    logger.info("🚀 Enhanced Anomaly Detection System - Technical Report Implementation")
    logger.info("=" * 80)
    logger.info("FEATURES IMPLEMENTED:")
    logger.info("✅ CNN-LSTM Architecture (InceptionV3 + Multi-layer LSTM)")
    logger.info("✅ Temporal Sequence Processing (32 frames)")
    logger.info("✅ Advanced Data Augmentation")
    logger.info("✅ Class Imbalance Handling")
    logger.info("✅ Feature Pre-computation and Caching")
    logger.info("✅ OpenVINO Optimization Support")
    logger.info("✅ Real-time Multi-threaded Inference")
    logger.info("✅ Comprehensive Evaluation and Monitoring")
    logger.info("=" * 80)
    
    try:
        # Initialize pipeline
        pipeline = TechnicalReportPipeline(config_path=args.config)
        pipeline.use_optimization = not args.disable_optimization
        
        # Run based on mode
        if args.mode == 'train':
            pipeline.prepare_data()
            pipeline.train_model(resume_training=args.resume)
            
        elif args.mode == 'evaluate':
            pipeline.prepare_data()
            pipeline.evaluate_model()
            
        elif args.mode == 'optimize':
            pipeline.optimize_for_deployment(args.target_platform)
            
        elif args.mode == 'inference':
            pipeline.setup_real_time_inference()
            pipeline.run_real_time_demo(duration=args.demo_duration)
            
        elif args.mode == 'complete':
            pipeline.run_complete_pipeline(
                skip_training=args.skip_training,
                skip_optimization=args.skip_optimization,
                skip_demo=args.skip_demo,
                target_platform=args.target_platform
            )
        
        logger.info("🎯 Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()