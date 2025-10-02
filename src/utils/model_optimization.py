"""
Model Optimization and Conversion Utilities
==========================================

Utilities for optimizing models for inference:
- PyTorch to ONNX conversion
- ONNX to OpenVINO IR conversion
- TensorRT optimization
- Model quantization
"""

import torch
import torch.onnx
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional, Union

# Try to import optimization libraries
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import openvino as ov
    from openvino.tools import mo
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from src.models.enhanced_temporal_model import EnhancedTemporalAnomalyModel
from src.utils.logging_config import get_app_logger


class ModelOptimizer:
    """
    Model optimization utility for enhanced inference performance
    """
    
    def __init__(self, model: EnhancedTemporalAnomalyModel, device: torch.device):
        self.model = model
        self.device = device
        self.logger = get_app_logger()
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Model metadata
        self.input_shape = (1, 32, 3, 299, 299)  # [batch, sequence, channels, height, width]
        self.output_dir = Path('models/optimized')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_onnx(
        self,
        output_path: Optional[str] = None,
        opset_version: int = 11,
        dynamic_axes: bool = True
    ) -> str:
        """
        Export PyTorch model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic input axes
            
        Returns:
            Path to saved ONNX model
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX is not available. Install with: pip install onnx onnxruntime")
        
        if output_path is None:
            output_path = self.output_dir / 'enhanced_temporal_model.onnx'
        else:
            output_path = Path(output_path)
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Define dynamic axes for flexible batch size and sequence length
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        try:
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
            
            # Verify the exported model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"✅ ONNX export successful: {output_path}")
            
            # Test inference speed
            self._benchmark_onnx(output_path)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ ONNX export failed: {e}")
            raise
    
    def convert_to_openvino(
        self,
        onnx_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        precision: str = "FP16"
    ) -> str:
        """
        Convert ONNX model to OpenVINO IR format
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory for OpenVINO IR
            precision: Model precision (FP32, FP16, INT8)
            
        Returns:
            Path to OpenVINO IR model
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is not available. Install with: pip install openvino")
        
        if onnx_path is None:
            onnx_path = self.output_dir / 'enhanced_temporal_model.onnx'
            if not Path(onnx_path).exists():
                self.logger.info("ONNX model not found, creating...")
                onnx_path = self.export_to_onnx()
        
        if output_dir is None:
            output_dir = self.output_dir / 'openvino'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Converting to OpenVINO IR format: {precision}")
        
        try:
            # Initialize OpenVINO Model Optimizer
            core = ov.Core()
            
            # Read ONNX model
            model = core.read_model(onnx_path)
            
            # Configure precision
            if precision == "FP16":
                # Compress model to FP16
                from openvino.tools import mo
                compressed_model = mo.compress_model_weights(model)
                model = compressed_model
            
            # Save the model
            ir_path = output_dir / "enhanced_temporal_model.xml"
            ov.save_model(model, str(ir_path))
            
            self.logger.info(f"✅ OpenVINO conversion successful: {ir_path}")
            
            # Test inference speed
            self._benchmark_openvino(str(ir_path))
            
            return str(ir_path)
            
        except Exception as e:
            self.logger.error(f"❌ OpenVINO conversion failed: {e}")
            raise
    
    def quantize_model(
        self,
        model_path: str,
        calibration_data: np.ndarray,
        output_path: Optional[str] = None
    ) -> str:
        """
        Quantize model to INT8 for faster inference
        
        Args:
            model_path: Path to model (ONNX or OpenVINO)
            calibration_data: Calibration data for quantization
            output_path: Output path for quantized model
            
        Returns:
            Path to quantized model
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO is not available for quantization")
        
        self.logger.info("Starting INT8 quantization...")
        
        try:
            import openvino.tools.pot as pot
            
            # Configuration for quantization
            config = {
                "model": {
                    "model_name": "enhanced_temporal_model",
                    "model": model_path,
                    "weights": model_path.replace('.xml', '.bin')
                },
                "engine": {
                    "type": "accuracy_checker"
                },
                "compression": {
                    "algorithms": [
                        {
                            "name": "DefaultQuantization",
                            "params": {
                                "preset": "performance",
                                "stat_subset_size": 300
                            }
                        }
                    ]
                }
            }
            
            # Run quantization
            # Note: This is a simplified version. In practice, you'd need proper calibration data
            self.logger.warning("INT8 quantization requires proper calibration dataset")
            
            return model_path  # Return original for now
            
        except Exception as e:
            self.logger.error(f"❌ Quantization failed: {e}")
            raise
    
    def _benchmark_onnx(self, onnx_path: str, num_runs: int = 100):
        """Benchmark ONNX model inference speed"""
        if not ONNX_AVAILABLE:
            return
        
        try:
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(onnx_path), providers=providers)
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                _ = session.run(None, {input_name: dummy_input})
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            fps = 1.0 / avg_time
            
            self.logger.info(f"ONNX Benchmark - Avg time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
            
        except Exception as e:
            self.logger.warning(f"ONNX benchmark failed: {e}")
    
    def _benchmark_openvino(self, ir_path: str, num_runs: int = 100):
        """Benchmark OpenVINO model inference speed"""
        if not OPENVINO_AVAILABLE:
            return
        
        try:
            # Initialize OpenVINO
            core = ov.Core()
            model = core.read_model(ir_path)
            
            # Compile for CPU (change to GPU if available)
            compiled_model = core.compile_model(model, "CPU")
            
            # Get input/output info
            input_layer = compiled_model.input(0)
            output_layer = compiled_model.output(0)
            
            # Prepare input
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                _ = compiled_model([dummy_input])[output_layer]
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                _ = compiled_model([dummy_input])[output_layer]
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            fps = 1.0 / avg_time
            
            self.logger.info(f"OpenVINO Benchmark - Avg time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
            
        except Exception as e:
            self.logger.warning(f"OpenVINO benchmark failed: {e}")
    
    def optimize_for_deployment(self, target_platform: str = "cpu") -> Dict[str, str]:
        """
        Complete optimization pipeline for deployment
        
        Args:
            target_platform: Target platform (cpu, gpu, edge)
            
        Returns:
            Dictionary of optimized model paths
        """
        results = {}
        
        self.logger.info(f"🚀 Starting model optimization for {target_platform}")
        
        try:
            # 1. Export to ONNX
            if ONNX_AVAILABLE:
                onnx_path = self.export_to_onnx()
                results['onnx'] = onnx_path
            
            # 2. Convert to OpenVINO (best for CPU inference)
            if OPENVINO_AVAILABLE and target_platform in ['cpu', 'edge']:
                precision = "FP16" if target_platform == "edge" else "FP32"
                openvino_path = self.convert_to_openvino(precision=precision)
                results['openvino'] = openvino_path
            
            # 3. TensorRT optimization (for NVIDIA GPUs)
            if TENSORRT_AVAILABLE and target_platform == "gpu":
                # TensorRT optimization would go here
                self.logger.info("TensorRT optimization not implemented yet")
            
            # Save optimization report
            self._save_optimization_report(results)
            
            self.logger.info("✅ Model optimization completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Model optimization failed: {e}")
            raise
    
    def _save_optimization_report(self, results: Dict[str, str]):
        """Save optimization report"""
        report = {
            'timestamp': time.time(),
            'original_model': 'EnhancedTemporalAnomalyModel',
            'input_shape': list(self.input_shape),
            'optimized_models': results,
            'libraries_available': {
                'onnx': ONNX_AVAILABLE,
                'openvino': OPENVINO_AVAILABLE,
                'tensorrt': TENSORRT_AVAILABLE
            }
        }
        
        report_path = self.output_dir / 'optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Optimization report saved: {report_path}")


def optimize_model_for_inference(model_checkpoint_path: str, target_platform: str = "cpu"):
    """
    Utility function to optimize a trained model for inference
    
    Args:
        model_checkpoint_path: Path to trained model checkpoint
        target_platform: Target deployment platform
    """
    logger = get_app_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load trained model
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        from src.models.enhanced_temporal_model import create_enhanced_temporal_model
        model = create_enhanced_temporal_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Create optimizer
        optimizer = ModelOptimizer(model, device)
        
        # Optimize for deployment
        results = optimizer.optimize_for_deployment(target_platform)
        
        logger.info("🎯 Model optimization completed!")
        logger.info("Optimized models:")
        for format_name, path in results.items():
            logger.info(f"  - {format_name.upper()}: {path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "models/checkpoints/enhanced_temporal_best.pth"
    
    if Path(checkpoint_path).exists():
        optimize_model_for_inference(checkpoint_path, "cpu")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the model first using the enhanced training script.")