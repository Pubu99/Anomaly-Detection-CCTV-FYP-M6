#!/usr/bin/env python3
"""
Enhanced Training Script for 95%+ Accuracy
==========================================

This script runs the enhanced training pipeline with all improvements:
- Combined loss function for extreme class imbalance
- Enhanced model architecture with attention mechanisms
- Advanced data augmentation strategies
- Logit adjustment for long-tail distribution
- Test-time augmentation for evaluation
"""

import sys
import os
sys.path.append('.')

import torch
import time
from pathlib import Path
from src.training.train import AnomalyTrainer
from src.utils.logging_config import get_app_logger

def main():
    """Main training function with enhanced pipeline"""
    logger = get_app_logger()
    
    logger.info("🚀 Starting Enhanced Anomaly Detection Training")
    logger.info("=" * 60)
    logger.info("IMPROVEMENTS IMPLEMENTED:")
    logger.info("- ✅ Combined Loss Function (Focal + CE + Contrastive)")
    logger.info("- ✅ Enhanced Model Architecture (Multi-head + Attention)")
    logger.info("- ✅ Advanced Data Augmentation (Heavy for minority classes)")
    logger.info("- ✅ Logit Adjustment for Long-tail Distribution")
    logger.info("- ✅ Test-Time Augmentation for Evaluation")
    logger.info("- ✅ Enhanced EMA and Regularization")
    logger.info("- ✅ Fixed JSON Serialization Issues")
    logger.info("=" * 60)
    
    # Initialize enhanced trainer
    try:
        trainer = AnomalyTrainer()
        logger.info("✅ Enhanced trainer initialized successfully")
        
        # Display training configuration
        logger.info("TRAINING CONFIGURATION:")
        logger.info(f"- Loss Function: Combined (Focal + CE + Contrastive)")
        logger.info(f"- Model: Enhanced EfficientNet-B3 with Multi-head Classification")
        logger.info(f"- Augmentation: Heavy (for extreme class imbalance)")
        logger.info(f"- Epochs: 75 (increased for better convergence)")
        logger.info(f"- Learning Rate: 0.0003 (optimized for stability)")
        logger.info(f"- Weight Decay: 0.08 (enhanced regularization)")
        logger.info(f"- EMA Decay: 0.9995 (improved model averaging)")
        logger.info(f"- Logit Adjustment: Enabled (tau=2.0)")
        logger.info(f"- Device: {trainer.device}")
        
        # Start training
        start_time = time.time()
        logger.info("🎯 Target: 95%+ Test Accuracy")
        logger.info("Starting enhanced training pipeline...")
        
        trainer.train()
        
        # Training completed
        end_time = time.time()
        training_duration = end_time - start_time
        hours = int(training_duration // 3600)
        minutes = int((training_duration % 3600) // 60)
        
        logger.info("=" * 60)
        logger.info("🎉 ENHANCED TRAINING COMPLETED!")
        logger.info(f"⏱️  Total Training Time: {hours}h {minutes}m")
        logger.info(f"🏆 Best Validation F1: {trainer.best_f1:.4f}")
        logger.info(f"🏆 Best Validation Accuracy: {trainer.best_accuracy:.4f}")
        logger.info("=" * 60)
        
        # Final evaluation on test set
        logger.info("🔍 Running final test evaluation with TTA...")
        trainer.evaluate_test_set()
        
        logger.info("✅ Enhanced training pipeline completed successfully!")
        logger.info("📊 Check the results in models/checkpoints/test_results.json")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()