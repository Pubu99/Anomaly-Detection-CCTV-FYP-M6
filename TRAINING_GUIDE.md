# 🚀 Complete Training Guide - Enhanced Anomaly Detection System

This guide shows you exactly how to run the model training from start to finish.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (3.9-3.10 recommended)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but recommended)
- **RAM**: 16GB+ recommended for training
- **Storage**: 50GB+ for dataset and models

### CUDA Installation (For GPU Training)
If you have an NVIDIA GPU, install CUDA Toolkit 11.8 or 12.1 from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

---

## 🔧 Step 1: Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Create new environment
conda create -n anomaly-detection python=3.9
conda activate anomaly-detection

# Verify environment
python --version  # Should show Python 3.9.x
```

### Option B: Using Virtual Environment
```bash
# Create virtual environment
python -m venv anomaly-env

# Activate (Windows)
anomaly-env\Scripts\activate

# Activate (Linux/Mac)
source anomaly-env/bin/activate
```

---

## 📦 Step 2: Install Dependencies

### Method 1: Automatic Installation (Recommended)
```bash
# Run the automated setup script
python enhanced_setup.py

# This will:
# ✅ Check Python version
# ✅ Install PyTorch with CUDA support (if available)
# ✅ Install all requirements
# ✅ Create necessary directories
# ✅ Run system tests
```

### Method 2: Manual Installation
```bash
# Install PyTorch (CUDA version - adjust for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Verify installation
python test_system.py
```

---

## 📊 Step 3: Prepare Your Dataset

### Dataset Structure
Your data should be organized like this:
```
data/raw/
├── Train/
│   ├── Abuse/
│   │   ├── video1.mp4
│   │   ├── video2.avi
│   │   └── ...
│   ├── Assault/
│   ├── Burglary/
│   ├── Fighting/
│   ├── NormalVideos/
│   └── ... (other classes)
└── Test/
    ├── Abuse/
    ├── Assault/
    └── ... (same structure as Train)
```

### Supported Video Formats
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`
- **Note**: For UCF-Crime dataset, you might have images (.png, .jpg) extracted from videos

### Creating Sample Data Structure
```bash
# If you don't have data yet, create the directory structure
python -c "
from pathlib import Path
classes = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'NormalVideos', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
for split in ['Train', 'Test']:
    for cls in classes:
        Path(f'data/raw/{split}/{cls}').mkdir(parents=True, exist_ok=True)
print('✅ Directory structure created')
"
```

---

## 🎯 Step 4: Training Options

### Option A: Complete Pipeline (Recommended for First Run)
```bash
# Run everything: data prep → training → evaluation → optimization
python train_technical_report.py --mode complete

# This takes 2-4 hours total:
# - Data preprocessing: 30 minutes
# - Model training: 2-3 hours 
# - Evaluation: 15 minutes
# - Model optimization: 15 minutes
```

### Option B: Step-by-Step Training
```bash
# Step 1: Data preparation and preprocessing
python train_technical_report.py --mode prepare

# Step 2: Train the model
python train_technical_report.py --mode train

# Step 3: Evaluate the model
python train_technical_report.py --mode evaluate

# Step 4: Optimize for deployment
python train_technical_report.py --mode optimize --target-platform cpu
```

### Option C: Resume Training (If Interrupted)
```bash
# Resume from the last checkpoint
python train_technical_report.py --mode train --resume
```

---

## ⚙️ Step 5: Training Configuration (Optional)

### Quick Configuration Changes
Edit `config/config.yaml` to customize training:

```yaml
# For faster training (lower accuracy)
training:
  epochs: 10              # Reduce from 30
  batch_size: 16          # Reduce if GPU memory issues

# For better accuracy (longer training)  
training:
  epochs: 50              # Increase epochs
  learning_rate: 0.00005  # Lower learning rate
```

### Advanced Configuration
```bash
# Custom configuration file
cp config/config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
python train_technical_report.py --mode train --config config/my_config.yaml
```

---

## 📈 Step 6: Monitor Training Progress

### Real-time Monitoring
Training will show progress like this:
```
🚀 Starting Enhanced Temporal Anomaly Detection Training
================================================================
IMPROVEMENTS IMPLEMENTED:
- ✅ CNN-LSTM Architecture (InceptionV3 + Multi-layer LSTM)  
- ✅ Temporal Sequence Processing (32 frames)
- ✅ Advanced Data Augmentation
- ✅ Feature Pre-computation and Caching

Epoch 1/30: 100%|██████████| 150/150 [12:34<00:00, Loss: 2.456, Val F1: 0.678]
Epoch 2/30: 100%|██████████| 150/150 [11:23<00:00, Loss: 1.234, Val F1: 0.745]
...
```

### Training Curves
Plots are automatically saved to `logs/plots/`:
- `training_history_epoch_X.png` - Loss and accuracy curves
- `confusion_matrix.png` - Model performance breakdown

### Logs and Checkpoints
- **Logs**: `logs/training.log`
- **Checkpoints**: `models/checkpoints/`
- **Best Model**: `models/checkpoints/enhanced_temporal_best.pth`

---

## 🎯 Step 7: Training Results

### What to Expect
After successful training, you'll see:
```
🎉 ENHANCED TRAINING COMPLETED!
⏱️  Total Training Time: 2h 34m
🏆 Best Validation F1: 0.9156
🏆 Best Validation Accuracy: 0.9323
===============================================
✅ Enhanced training pipeline completed successfully!
📊 Check the results in models/checkpoints/test_results.json
```

### Performance Files Created
- `models/checkpoints/enhanced_temporal_best.pth` - Best model weights
- `models/checkpoints/test_results.json` - Detailed evaluation metrics
- `logs/plots/training_history_epoch_30.png` - Training curves
- `models/optimized/` - Optimized models for deployment

---

## 🧪 Step 8: Test Your Trained Model

### Quick Test
```bash
# Test the trained model
python -c "
from src.models.enhanced_temporal_model import create_enhanced_temporal_model
import torch
from pathlib import Path

# Load trained model
checkpoint_path = Path('models/checkpoints/enhanced_temporal_best.pth')
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f'✅ Model loaded - Validation F1: {checkpoint[\"val_f1\"]:.4f}')
    print(f'✅ Training completed at epoch: {checkpoint[\"epoch\"]}')
else:
    print('❌ No trained model found - run training first')
"
```

### Real-time Inference Test
```bash
# Test real-time inference (30 seconds)
python train_technical_report.py --mode inference --demo-duration 30
```

---

## 🔧 Common Issues and Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
# Edit config/config.yaml:
dataset:
  batch_size: 16  # Reduce from 32

# Or use CPU training
python train_technical_report.py --mode train --device cpu
```

### Issue 2: ImportError for modules
```bash
# Solution: Install missing dependencies
pip install -r requirements.txt

# Or run setup again
python enhanced_setup.py
```

### Issue 3: No data found
```bash
# Solution: Check data structure
ls data/raw/Train/
# Should show class folders: Abuse, Assault, Fighting, etc.

# Create sample structure if empty
python enhanced_setup.py
```

### Issue 4: Slow training
```bash
# Solution 1: Enable feature caching (automatic in enhanced version)
# Solution 2: Reduce epochs for testing
python train_technical_report.py --mode train --epochs 5

# Solution 3: Use smaller model
# Edit config.yaml: change epochs from 30 to 10
```

### Issue 5: Training stalls or NaN loss
```bash
# Solution: Lower learning rate
# Edit config/config.yaml:
training:
  learning_rate: 0.00005  # Reduce from 0.0001
```

---

## 📊 Expected Training Times

| Hardware | Batch Size | Time per Epoch | Total (30 epochs) |
|----------|------------|----------------|-------------------|
| **RTX 4090** | 32 | 4-6 minutes | 2-3 hours |
| **RTX 3080** | 32 | 6-8 minutes | 3-4 hours |
| **RTX 3070** | 16 | 8-10 minutes | 4-5 hours |
| **CPU Only** | 8 | 25-30 minutes | 12-15 hours |

---

## 🚀 Quick Start Commands

For immediate training (assuming you have data ready):

```bash
# 1. Setup (one time)
python enhanced_setup.py

# 2. Train model
python train_technical_report.py --mode complete

# 3. Test results  
python train_technical_report.py --mode inference --demo-duration 30
```

That's it! Your model will be trained and ready for deployment.

---

## 📞 Need Help?

If you encounter issues:

1. **Check system test**: `python test_system.py`
2. **Check logs**: Look in `logs/training.log`
3. **Reduce complexity**: Start with fewer epochs for testing
4. **Check GPU memory**: Use `nvidia-smi` to monitor GPU usage
5. **Verify data**: Ensure your dataset is properly structured

The enhanced system is designed to be robust and provide helpful error messages to guide you through any issues!