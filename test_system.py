"""
System Test Script for Enhanced Anomaly Detection
================================================

Quick test script to verify installation and functionality
"""

import sys
import importlib
from pathlib import Path
import torch
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Module Imports...")
    
    required_modules = [
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'pandas',
        'sklearn',
        'albumentations',
        'tqdm'
    ]
    
    optional_modules = [
        'openvino',
        'onnx', 
        'onnxruntime',
        'decord'
    ]
    
    results = {}
    
    # Test required modules
    for module in required_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ OK"
        except ImportError as e:
            results[module] = f"❌ FAILED: {e}"
    
    # Test optional modules  
    for module in optional_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ OK (Optional)"
        except ImportError:
            results[module] = "⚠️ NOT INSTALLED (Optional)"
    
    print("\nImport Results:")
    for module, status in results.items():
        print(f"  {module}: {status}")
    
    return results

def test_custom_modules():
    """Test custom modules"""
    print("\n🔧 Testing Custom Modules...")
    
    custom_modules = [
        'src.models.enhanced_temporal_model',
        'src.utils.config',
        'src.utils.logging_config',
        'src.data.enhanced_data_preprocessing'
    ]
    
    results = {}
    
    for module in custom_modules:
        try:
            importlib.import_module(module)
            results[module] = "✅ OK"
        except ImportError as e:
            results[module] = f"❌ FAILED: {e}"
    
    print("\nCustom Module Results:")
    for module, status in results.items():
        print(f"  {module}: {status}")
    
    return results

def test_model_creation():
    """Test model creation"""
    print("\n🏗️ Testing Model Creation...")
    
    try:
        from src.models.enhanced_temporal_model import create_enhanced_temporal_model
        
        config = {
            'model': {
                'temporal': {
                    'num_classes': 14,
                    'max_seq_length': 32,
                    'use_attention': True
                }
            }
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_enhanced_temporal_model(config)
        model.to(device)
        
        # Test forward pass
        dummy_input = torch.randn(1, 32, 3, 299, 299).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print("✅ Model creation and forward pass successful")
        print(f"   Device: {device}")
        print(f"   Output shape: {output['main'].shape}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing"""
    print("\n📊 Testing Data Preprocessing...")
    
    try:
        from src.data.enhanced_data_preprocessing import VideoSequenceExtractor
        
        extractor = VideoSequenceExtractor(
            max_frames=8,  # Small for testing
            target_size=(64, 64)  # Small for testing
        )
        
        print("✅ Data preprocessing modules loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_system_resources():
    """Test system resources"""
    print("\n💻 Testing System Resources...")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {'✅ YES' if cuda_available else '❌ NO (CPU only)'}")
    
    if cuda_available:
        print(f"   CUDA Device: {torch.cuda.get_device_name()}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"   RAM Available: {memory.available / 1e9:.1f} GB / {memory.total / 1e9:.1f} GB")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Enhanced Anomaly Detection System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Custom Modules", test_custom_modules), 
        ("System Resources", test_system_resources),
        ("Model Creation", test_model_creation),
        ("Data Preprocessing", test_data_preprocessing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    all_passed = True
    for test_name, result in results.items():
        if isinstance(result, dict):
            # For import tests, check if any required failed
            failed_required = any("❌ FAILED" in status for module, status in result.items() 
                                if "Optional" not in status)
            status = "✅ PASSED" if not failed_required else "❌ FAILED"
        else:
            status = "✅ PASSED" if result else "❌ FAILED"
        
        print(f"   {test_name}: {status}")
        if not result or (isinstance(result, dict) and failed_required):
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
        print("\nNext steps:")
        print("1. Run: python train_technical_report.py --mode complete")
        print("2. Or start with: python train_technical_report.py --mode train")
    else:
        print("⚠️ SOME TESTS FAILED. Check the output above and install missing dependencies.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    main()