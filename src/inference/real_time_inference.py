"""
Real-time Multi-Camera Inference Engine
=======================================

Professional real-time inference system with optimized frame processing,
multi-camera synchronization, and intelligent alert generation.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import asyncio
import threading
import queue
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json

# Import custom modules
from src.models.hybrid_model import HybridAnomalyModel, AnomalyResult, create_model
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger, PerformanceTimer


@dataclass
class CameraConfig:
    """Configuration for individual camera"""
    camera_id: str
    camera_url: str  # RTSP URL or camera index
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    position: Tuple[float, float] = (0.0, 0.0)  # x, y coordinates
    weight: float = 1.0  # Importance weight for this camera
    enabled: bool = True


@dataclass
class FrameData:
    """Structure for frame data with metadata"""
    frame: np.ndarray
    timestamp: float
    camera_id: str
    frame_id: int
    processed: bool = False
    anomalies: List[AnomalyResult] = field(default_factory=list)


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    batch_size: int = 4
    max_fps: int = 30
    frame_sampling_rate: int = 3  # Process every Nth frame
    buffer_size: int = 100
    confidence_threshold: float = 0.75
    alert_cooldown: float = 5.0  # Seconds between same alerts
    max_concurrent_cameras: int = 10


class FrameProcessor:
    """Optimized frame processor with batching and caching"""
    
    def __init__(self, model: HybridAnomalyModel, config: InferenceConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = get_app_logger()
        
        # Preprocessing pipeline
        self.image_size = (224, 224)  # Model input size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Performance tracking
        self.inference_times = []
        self.frame_count = 0
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model input
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, self.image_size)
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - self.mean) / self.std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def process_batch(self, frames: List[FrameData]) -> List[FrameData]:
        """
        Process a batch of frames
        
        Args:
            frames: List of frame data
            
        Returns:
            Processed frames with anomaly results
        """
        if not frames:
            return []
        
        start_time = time.time()
        
        # Preprocess all frames
        batch_tensors = []
        camera_ids = []
        
        for frame_data in frames:
            try:
                tensor = self.preprocess_frame(frame_data.frame)
                batch_tensors.append(tensor)
                camera_ids.append(frame_data.camera_id)
            except Exception as e:
                self.logger.error(f"Error preprocessing frame from {frame_data.camera_id}: {e}")
                continue
        
        if not batch_tensors:
            return frames
        
        # Batch inference
        try:
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            with torch.no_grad():
                # Get predictions
                predictions = self.model.predict_anomaly(batch_tensor, camera_ids)
                
                # Update frame data with results
                for i, (frame_data, prediction) in enumerate(zip(frames[:len(predictions)], predictions)):
                    # Update timestamp
                    prediction.timestamp = frame_data.timestamp
                    
                    # Add to frame data
                    if prediction.confidence >= self.config.confidence_threshold:
                        frame_data.anomalies.append(prediction)
                    
                    frame_data.processed = True
                
        except Exception as e:
            self.logger.error(f"Error during batch inference: {e}")
            return frames
        
        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += len(frames)
        
        # Log performance periodically
        if self.frame_count % 100 == 0:
            avg_time = np.mean(self.inference_times[-100:])
            fps = len(frames) / inference_time
            self.logger.info(f"Batch processing: {fps:.1f} FPS, Avg time: {avg_time:.3f}s")
        
        return frames


class CameraManager:
    """Manages multiple camera streams with synchronization"""
    
    def __init__(self, cameras: List[CameraConfig]):
        self.cameras = {cam.camera_id: cam for cam in cameras}
        self.camera_streams = {}
        self.frame_queues = {}
        self.logger = get_app_logger()
        
        # Initialize queues for each camera
        for camera_id in self.cameras:
            self.frame_queues[camera_id] = queue.Queue(maxsize=50)
    
    def start_camera_stream(self, camera_id: str) -> bool:
        """Start streaming from a specific camera"""
        if camera_id not in self.cameras:
            self.logger.error(f"Camera {camera_id} not configured")
            return False
        
        camera_config = self.cameras[camera_id]
        
        try:
            # Try to parse as integer (local camera)
            camera_source = int(camera_config.camera_url)
        except ValueError:
            # Use as string (RTSP URL)
            camera_source = camera_config.camera_url
        
        try:
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                self.logger.error(f"Failed to open camera {camera_id}")
                return False
            
            # Configure camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, camera_config.fps)
            
            self.camera_streams[camera_id] = cap
            
            # Start capture thread
            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id, cap),
                daemon=True
            )
            thread.start()
            
            self.logger.info(f"Camera {camera_id} started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting camera {camera_id}: {e}")
            return False
    
    def _capture_frames(self, camera_id: str, cap: cv2.VideoCapture):
        """Capture frames from camera in separate thread"""
        frame_id = 0
        
        while True:
            try:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Create frame data
                frame_data = FrameData(
                    frame=frame,
                    timestamp=time.time(),
                    camera_id=camera_id,
                    frame_id=frame_id
                )
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queues[camera_id].put_nowait(frame_data)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queues[camera_id].get_nowait()
                        self.frame_queues[camera_id].put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                frame_id += 1
                
            except Exception as e:
                self.logger.error(f"Error capturing frame from camera {camera_id}: {e}")
                time.sleep(1)
    
    def get_synchronized_frames(self, max_frames: int = 4) -> List[FrameData]:
        """Get synchronized frames from all active cameras"""
        frames = []
        current_time = time.time()
        
        for camera_id in self.cameras:
            if camera_id in self.camera_streams and not self.frame_queues[camera_id].empty():
                try:
                    frame_data = self.frame_queues[camera_id].get_nowait()
                    
                    # Check if frame is recent (within 1 second)
                    if current_time - frame_data.timestamp < 1.0:
                        frames.append(frame_data)
                    
                    if len(frames) >= max_frames:
                        break
                        
                except queue.Empty:
                    continue
        
        return frames
    
    def stop_all_cameras(self):
        """Stop all camera streams"""
        for camera_id, cap in self.camera_streams.items():
            try:
                cap.release()
                self.logger.info(f"Camera {camera_id} stopped")
            except Exception as e:
                self.logger.error(f"Error stopping camera {camera_id}: {e}")
        
        self.camera_streams.clear()


class AlertManager:
    """Manages alert generation and cooldown"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.logger = get_app_logger()
        self.last_alerts = {}  # camera_id -> {alert_type -> timestamp}
        self.alert_callbacks = []
    
    def register_alert_callback(self, callback: Callable[[AnomalyResult], None]):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def should_generate_alert(self, anomaly: AnomalyResult) -> bool:
        """Check if alert should be generated based on cooldown"""
        camera_id = anomaly.camera_id
        alert_type = anomaly.anomaly_class
        current_time = time.time()
        
        # Initialize camera alerts if not exists
        if camera_id not in self.last_alerts:
            self.last_alerts[camera_id] = {}
        
        # Check cooldown
        if alert_type in self.last_alerts[camera_id]:
            last_time = self.last_alerts[camera_id][alert_type]
            if current_time - last_time < self.config.alert_cooldown:
                return False
        
        # Update last alert time
        self.last_alerts[camera_id][alert_type] = current_time
        return True
    
    def generate_alert(self, anomaly: AnomalyResult):
        """Generate alert and notify callbacks"""
        if self.should_generate_alert(anomaly):
            self.logger.warning(
                f"ALERT: {anomaly.anomaly_class} detected on {anomaly.camera_id} "
                f"(confidence: {anomaly.confidence:.3f}, severity: {anomaly.severity})"
            )
            
            # Notify all callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}")


class RealTimeInferenceEngine:
    """Main real-time inference engine coordinating all components"""
    
    def __init__(self, model_path: str, cameras: List[CameraConfig], config_path: Optional[str] = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            cameras: List of camera configurations
            config_path: Optional configuration file path
        """
        self.logger = get_app_logger()
        self.config = get_config() if config_path is None else get_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize components
        self.inference_config = InferenceConfig(
            batch_size=self.config.get('inference.batch_size', 4),
            max_fps=self.config.get('inference.max_fps', 30),
            frame_sampling_rate=self.config.get('inference.frame_sampling_rate', 3),
            confidence_threshold=self.config.get('model.fusion.confidence_threshold', 0.75)
        )
        
        self.frame_processor = FrameProcessor(self.model, self.inference_config, self.device)
        self.camera_manager = CameraManager(cameras)
        self.alert_manager = AlertManager(self.inference_config)
        
        # State management
        self.running = False
        self.frame_skip_counter = 0
        
        # Performance metrics
        self.start_time = None
        self.total_frames_processed = 0
        self.total_alerts_generated = 0
    
    def _load_model(self, model_path: str) -> HybridAnomalyModel:
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            model = create_model(self.config.config)
            
            # Load weights
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            model.eval()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def register_alert_callback(self, callback: Callable[[AnomalyResult], None]):
        """Register callback for alert notifications"""
        self.alert_manager.register_alert_callback(callback)
    
    def start_inference(self):
        """Start real-time inference"""
        self.logger.info("Starting real-time inference...")
        
        # Start all cameras
        for camera_id in self.camera_manager.cameras:
            self.camera_manager.start_camera_stream(camera_id)
        
        self.running = True
        self.start_time = time.time()
        
        # Main inference loop
        try:
            while self.running:
                # Get synchronized frames
                frames = self.camera_manager.get_synchronized_frames(
                    max_frames=self.inference_config.batch_size
                )
                
                if not frames:
                    time.sleep(0.01)  # Short sleep if no frames
                    continue
                
                # Frame sampling (process every Nth frame)
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.inference_config.frame_sampling_rate != 0:
                    continue
                
                # Process frames
                with PerformanceTimer("frame_processing"):
                    processed_frames = self.frame_processor.process_batch(frames)
                
                # Check for anomalies and generate alerts
                for frame_data in processed_frames:
                    for anomaly in frame_data.anomalies:
                        self.alert_manager.generate_alert(anomaly)
                        self.total_alerts_generated += 1
                
                self.total_frames_processed += len(processed_frames)
                
                # Log performance periodically
                if self.total_frames_processed % 100 == 0:
                    self._log_performance()
                
        except KeyboardInterrupt:
            self.logger.info("Inference stopped by user")
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
        finally:
            self.stop_inference()
    
    def _log_performance(self):
        """Log performance metrics"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            fps = self.total_frames_processed / elapsed_time
            
            self.logger.info(
                f"Performance: {fps:.1f} FPS, "
                f"Processed: {self.total_frames_processed} frames, "
                f"Alerts: {self.total_alerts_generated}, "
                f"Runtime: {elapsed_time:.1f}s"
            )
    
    def stop_inference(self):
        """Stop real-time inference"""
        self.logger.info("Stopping inference...")
        self.running = False
        self.camera_manager.stop_all_cameras()
        self._log_performance()
    
    async def start_async_inference(self):
        """Start inference in async mode"""
        loop = asyncio.get_event_loop()
        
        # Run inference in thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(executor, self.start_inference)
            await future


def create_demo_cameras() -> List[CameraConfig]:
    """Create demo camera configurations"""
    return [
        CameraConfig(
            camera_id="camera_1",
            camera_url="0",  # Default camera
            position=(0.0, 0.0),
            weight=1.0
        ),
        CameraConfig(
            camera_id="camera_2", 
            camera_url="1",  # Second camera if available
            position=(10.0, 0.0),
            weight=0.8
        )
    ]


def alert_callback(anomaly: AnomalyResult):
    """Example alert callback function"""
    print(f"🚨 ALERT: {anomaly.anomaly_class} detected!")
    print(f"   Camera: {anomaly.camera_id}")
    print(f"   Confidence: {anomaly.confidence:.3f}")
    print(f"   Severity: {anomaly.severity}")
    print(f"   Timestamp: {time.ctime(anomaly.timestamp)}")


def main():
    """Main function for testing inference engine"""
    # Create demo cameras
    cameras = create_demo_cameras()
    
    # Initialize inference engine
    try:
        engine = RealTimeInferenceEngine(
            model_path="models/checkpoints/best_model.pth",
            cameras=cameras
        )
        
        # Register alert callback
        engine.register_alert_callback(alert_callback)
        
        # Start inference
        engine.start_inference()
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()