"""
Enhanced Anomaly Detection System - Setup Script
==============================================

Automated setup script for the enhanced anomaly detection system
"""

import sys
import subprocess
import os
from pathlib import Path
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor
        ))
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🎮 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ CUDA available: {device_name} ({memory_gb:.1f} GB)")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU mode")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet - CUDA check will be performed after installation")
        return False

def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    print("\n🔥 Installing PyTorch...")
    
    # Detect CUDA version (simplified)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected - installing CUDA version")
            cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ]
        else:
            print("ℹ️  No NVIDIA GPU detected - installing CPU version")
            cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
    except FileNotFoundError:
        print("ℹ️  nvidia-smi not found - installing CPU version")
        cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio']
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ PyTorch installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False

def install_requirements():
    """Install other requirements"""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path('requirements.txt')
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        subprocess.run(cmd, check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Requirements installation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'data/raw',
        'data/processed',
        'data/processed/cache',
        'models/checkpoints',
        'models/optimized',
        'models/openvino',
        'logs/plots',
        'alerts'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def download_sample_data():
    """Download or setup sample data"""
    print("\n📊 Setting up sample data...")
    
    data_dir = Path('data/raw')
    if not (data_dir / 'Train').exists():
        print("ℹ️  Sample data not found.")
        print("📋 To use your own data:")
        print("   1. Place training videos in: data/raw/Train/[ClassName]/")
        print("   2. Place test videos in: data/raw/Test/[ClassName]/")
        print("   3. Supported classes: Abuse, Assault, Burglary, Fighting, Normal, etc.")
        
        # Create sample structure
        sample_classes = ['Normal', 'Fighting', 'Assault', 'Robbery']
        for split in ['Train', 'Test']:
            for class_name in sample_classes:
                (data_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        print("✅ Sample directory structure created")
    else:
        print("✅ Data directory structure exists")

def run_system_test():
    """Run system test"""
    print("\n🧪 Running system test...")
    
    try:
        result = subprocess.run([sys.executable, 'test_system.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System test passed!")
            print(result.stdout)
            return True
        else:
            print("❌ System test failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Could not run system test: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Enhanced Anomaly Detection System - Setup")
    print("=" * 60)
    
    steps = [
        ("Python Version Check", check_python_version),
        ("PyTorch Installation", install_pytorch),
        ("Requirements Installation", install_requirements),
        ("Directory Creation", create_directories),
        ("Sample Data Setup", download_sample_data),
        ("CUDA Check", check_cuda),
        ("System Test", run_system_test),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        try:
            result = step_func()
            results[step_name] = result
            if not result and step_name in ["Python Version Check"]:
                print(f"❌ Critical step failed: {step_name}")
                break
        except Exception as e:
            print(f"❌ Step failed with exception: {e}")
            results[step_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Setup Summary:")
    
    success_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for step_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {step_name}: {status}")
    
    print(f"\nOverall: {success_count}/{total_count} steps completed")
    
    if success_count == total_count:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next Steps:")
        print("1. Add your dataset to data/raw/Train/ and data/raw/Test/")
        print("2. Run: python train_technical_report.py --mode complete")
        print("3. Or start with training: python train_technical_report.py --mode train")
    else:
        print("\n⚠️  Setup completed with issues.")
        print("Please resolve the failed steps and run setup again.")
        print("\n🔧 Troubleshooting:")
        print("- Ensure you have Python 3.8+")
        print("- Check internet connection for package downloads")
        print("- Install Visual Studio Build Tools (Windows)")
        print("- Try: pip install --upgrade pip setuptools wheel")

if __name__ == "__main__":
    main()