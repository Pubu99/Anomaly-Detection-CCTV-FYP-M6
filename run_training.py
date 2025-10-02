"""
Simple Training Runner for Enhanced Anomaly Detection
===================================================

Quick and easy way to start training your model.
"""

import sys
import subprocess
import os
from pathlib import Path
import time

def check_environment():
    """Check if environment is ready for training"""
    print("🔍 Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if PyTorch is installed
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} found")
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"✅ Training will use: {device}")
    except ImportError:
        print("❌ PyTorch not found. Run: pip install torch torchvision")
        return False
    
    # Check data directory
    data_dir = Path('data/raw')
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please create data structure or run: python enhanced_setup.py")
        return False
    
    # Check for training data
    train_dir = data_dir / 'Train'
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    # Count classes with data
    classes_with_data = 0
    for class_dir in train_dir.iterdir():
        if class_dir.is_dir():
            files = list(class_dir.glob('*'))
            if files:
                classes_with_data += 1
                print(f"✅ Found {len(files)} files in {class_dir.name}")
    
    if classes_with_data == 0:
        print("❌ No training data found")
        print("Please add your dataset to data/raw/Train/[ClassName]/")
        return False
    
    print(f"✅ Found data for {classes_with_data} classes")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install core requirements
        essential_packages = [
            'torch', 'torchvision', 'opencv-python', 'numpy', 
            'pandas', 'scikit-learn', 'matplotlib', 'tqdm', 'albumentations'
        ]
        
        print("Installing essential packages...")
        for package in essential_packages:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✅ Installed {package}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def run_training():
    """Run the training process"""
    print("\n🎯 Starting model training...")
    
    # Check if we have the enhanced training script
    train_script = Path('train_technical_report.py')
    if not train_script.exists():
        print("❌ Enhanced training script not found")
        return False
    
    try:
        # Run training
        print("🚀 Launching training process...")
        print("This will take 2-4 hours depending on your hardware.")
        print("You can monitor progress in the terminal output.")
        
        # Start training
        result = subprocess.run([
            sys.executable, 'train_technical_report.py', 
            '--mode', 'train'
        ], capture_output=False)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False

def quick_test():
    """Run a quick test to verify setup"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        print("Testing imports...")
        import torch
        import cv2
        import numpy as np
        import pandas as pd
        from pathlib import Path
        
        print("✅ All imports successful")
        
        # Test data loading
        data_dir = Path('data/raw/Train')
        if data_dir.exists():
            video_files = []
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    files = list(class_dir.glob('*'))[:2]  # First 2 files only
                    video_files.extend(files)
            
            if video_files:
                print(f"✅ Found {len(video_files)} sample files for testing")
            else:
                print("⚠️  No sample files found")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main training runner"""
    print("🚀 Enhanced Anomaly Detection - Training Runner")
    print("=" * 60)
    
    # Step 1: Environment check
    if not check_environment():
        print("\n🔧 Environment issues found. Trying to fix...")
        
        # Try to install requirements
        if not install_requirements():
            print("❌ Could not fix environment automatically")
            print("Please run: python enhanced_setup.py")
            return
        
        # Check again
        if not check_environment():
            print("❌ Environment still not ready")
            return
    
    # Step 2: Quick test
    if not quick_test():
        print("❌ System test failed")
        print("Please check your installation")
        return
    
    # Step 3: Training options
    print("\n🎯 Training Options:")
    print("1. Quick Training (10 epochs, ~1 hour)")
    print("2. Full Training (30 epochs, ~3 hours)")  
    print("3. Custom Training")
    print("4. Test Only (no training)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("🚀 Starting Quick Training...")
        subprocess.run([
            sys.executable, 'train_technical_report.py', 
            '--mode', 'train', '--epochs', '10'
        ])
        
    elif choice == "2":
        print("🚀 Starting Full Training...")
        subprocess.run([
            sys.executable, 'train_technical_report.py', 
            '--mode', 'complete'
        ])
        
    elif choice == "3":
        epochs = input("Enter number of epochs (default 30): ").strip() or "30"
        batch_size = input("Enter batch size (default 32): ").strip() or "32"
        
        print(f"🚀 Starting Custom Training ({epochs} epochs, batch size {batch_size})...")
        subprocess.run([
            sys.executable, 'train_technical_report.py', 
            '--mode', 'train', '--epochs', epochs, '--batch-size', batch_size
        ])
        
    elif choice == "4":
        print("🧪 Running system test only...")
        subprocess.run([sys.executable, 'test_system.py'])
        
    else:
        print("❌ Invalid choice")
        return
    
    print("\n✅ Training runner completed!")

if __name__ == "__main__":
    main()