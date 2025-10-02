"""
Enhanced Real-time Inference Engine with OpenVINO Support
========================================================

Implements the real-time processing pipeline from the technical report:
- Multi-threaded video processing
- OpenVINO optimization for faster inference
- Sliding window temporal processing
- Intelligent alert generation
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import asyncio
from collections import deque

# Try to import OpenVINO
try:
    import openvino as ov
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available. Using PyTorch inference.")

# Import custom modules
from src.models.enhanced_temporal_model import EnhancedTemporalAnomalyModel, VideoProcessor
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger


@dataclass
class CameraStream:
    """Camera stream configuration"""
    camera_id: str
    source: Union[str, int]  # RTSP URL or camera index
    fps: int = 30
    resolution: Tuple[int, int] = (1920, 1080)
    enabled: bool = True
    position: Tuple[float, float] = (0.0, 0.0)
    
    # Processing parameters
    frame_skip: int = 2  # Process every Nth frame (as per technical report)
    buffer_size: int = 100
    
    # Alert parameters
    alert_cooldown: float = 5.0
    confidence_threshold: float = 0.75


@dataclass
class AlertInfo:
    """Alert information structure"""
    timestamp: float
    camera_id: str
    anomaly_type: str
    confidence: float
    severity: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    frame: Optional[np.ndarray] = None
    description: str = ""


class VideoPlayer:
    """
    Enhanced video player with threading support (from technical report)
    """
    
    def __init__(
        self, 
        source: Union[str, int], 
        fps: Optional[int] = None, 
        flip: bool = False,
        skip_first_frames: int = 0,
        buffer_size: int = 100
    ):
        self.source = source
        self.fps = fps
        self.flip = flip
        self.skip_first_frames = skip_first_frames
        self.buffer_size = buffer_size
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        
        # Get video properties
        self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output_fps = fps if fps is not None else self.input_fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame buffer and threading
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.frame_count = 0
        
        # Skip initial frames
        for _ in range(skip_first_frames):
            ret, _ = self.cap.read()
            if not ret:
                break
    
    def start(self):
        """Start the video capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the video capture thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip if requested
            if self.flip:
                frame = cv2.flip(frame, 1)
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put((frame, self.frame_count), block=False)
                self.frame_count += 1
            except queue.Full:
                # Remove oldest frame if buffer is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, self.frame_count), block=False)
                    self.frame_count += 1
                except queue.Empty:
                    pass
            
            # Control frame rate
            if self.output_fps > 0:
                time.sleep(1.0 / self.output_fps)
    
    def next(self) -> Tuple[Optional[np.ndarray], int]:
        """Get next frame"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, -1


class OpenVINOOptimizer:
    """
    OpenVINO optimization for faster inference (from technical report)
    """
    
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = Path(model_path)
        self.device = device
        self.core = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        
        if OPENVINO_AVAILABLE:
            self.initialize_openvino()
    
    def initialize_openvino(self):
        """Initialize OpenVINO inference engine"""
        try:
            # Initialize OpenVINO Runtime
            self.core = ov.Core()
            
            # Read the model
            model = self.core.read_model(self.model_path)
            
            # Compile the model
            self.compiled_model = self.core.compile_model(model, self.device)
            
            # Get input and output layers
            self.input_layer = self.compiled_model.input(0)
            self.output_layer = self.compiled_model.output(0)
            
            print(f"OpenVINO model loaded successfully on {self.device}")
            print(f"Input shape: {self.input_layer.shape}")
            print(f"Output shape: {self.output_layer.shape}")
            
        except Exception as e:
            print(f"Failed to initialize OpenVINO: {e}")
            self.core = None
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference using OpenVINO"""
        if self.compiled_model is None:
            raise RuntimeError("OpenVINO model not initialized")
        
        # Run inference
        result = self.compiled_model([input_data])[self.output_layer]
        return result


class EnhancedRealTimeInference:
    """
    Enhanced real-time inference engine with multi-camera support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_app_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self.load_model()
        
        # Initialize OpenVINO if available
        self.openvino_optimizer = None
        if OPENVINO_AVAILABLE:
            self.setup_openvino()
        
        # Camera streams
        self.camera_streams: Dict[str, CameraStream] = {}
        self.video_players: Dict[str, VideoPlayer] = {}
        self.video_processors: Dict[str, VideoProcessor] = {}
        
        # Alert system
        self.alert_queue = queue.Queue()
        self.last_alerts: Dict[str, float] = {}  # Camera ID -> timestamp
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        
        # Processing control
        self.running = False
        self.processing_threads = []
        
    def load_model(self) -> EnhancedTemporalAnomalyModel:
        """Load the trained model"""
        model_path = Path('models/checkpoints/enhanced_temporal_best.pth')
        
        if not model_path.exists():
            self.logger.warning("Enhanced temporal model not found, using default model")
            # Return a default model
            from src.models.enhanced_temporal_model import create_enhanced_temporal_model
            model = create_enhanced_temporal_model(self.config)
        else:
            # Load trained model
            checkpoint = torch.load(model_path, map_location=self.device)
            from src.models.enhanced_temporal_model import create_enhanced_temporal_model
            model = create_enhanced_temporal_model(checkpoint['config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            
            self.logger.info(f"Loaded model with validation F1: {checkpoint['val_f1']:.4f}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def setup_openvino(self):
        """Setup OpenVINO optimization"""
        openvino_model_path = Path('models/openvino/enhanced_temporal_model.xml')
        
        if openvino_model_path.exists():
            try:
                self.openvino_optimizer = OpenVINOOptimizer(str(openvino_model_path))
                self.logger.info("OpenVINO optimization enabled")
            except Exception as e:
                self.logger.warning(f"Failed to setup OpenVINO: {e}")
        else:
            self.logger.warning("OpenVINO model not found, using PyTorch inference")
    
    def add_camera_stream(self, stream_config: CameraStream):
        """Add a camera stream for processing"""
        self.camera_streams[stream_config.camera_id] = stream_config
        
        # Initialize video player
        player = VideoPlayer(
            source=stream_config.source,
            fps=stream_config.fps,
            skip_first_frames=0,
            buffer_size=stream_config.buffer_size
        )
        self.video_players[stream_config.camera_id] = player
        
        # Initialize video processor
        processor = VideoProcessor(self.model, self.device)
        self.video_processors[stream_config.camera_id] = processor
        
        self.logger.info(f"Added camera stream: {stream_config.camera_id}")
    
    def process_camera_stream(self, camera_id: str):
        """
        Process frames from a single camera stream
        """
        stream = self.camera_streams[camera_id]
        player = self.video_players[camera_id]
        processor = self.video_processors[camera_id]
        
        frame_count = 0
        
        try:
            player.start()
            
            while self.running:
                start_time = time.time()
                
                # Get next frame
                frame, frame_id = player.next()
                if frame is None:
                    continue
                
                # Skip frames according to configuration (technical report approach)
                if frame_count % stream.frame_skip != 0:
                    frame_count += 1
                    continue
                
                frame_count += 1
                
                # Process frame
                inference_start = time.time()
                result = processor.add_frame(frame)
                inference_time = time.time() - inference_start
                
                if result is not None:
                    # Check if anomaly detected
                    if result['is_anomaly'] and result['confidence'] > stream.confidence_threshold:
                        # Check alert cooldown
                        current_time = time.time()
                        last_alert_time = self.last_alerts.get(camera_id, 0)
                        
                        if current_time - last_alert_time > stream.alert_cooldown:
                            # Generate alert
                            alert = AlertInfo(
                                timestamp=current_time,
                                camera_id=camera_id,
                                anomaly_type=result['predicted_class'],
                                confidence=result['confidence'],
                                severity=result['severity'],
                                frame=frame.copy(),
                                description=f"Detected {result['predicted_class']} with {result['confidence']:.2f} confidence"
                            )
                            
                            # Add to alert queue
                            try:
                                self.alert_queue.put_nowait(alert)
                                self.last_alerts[camera_id] = current_time
                                
                                self.logger.warning(
                                    f"🚨 ANOMALY DETECTED - Camera: {camera_id}, "
                                    f"Type: {result['predicted_class']}, "
                                    f"Confidence: {result['confidence']:.3f}, "
                                    f"Severity: {result['severity']}"
                                )
                            except queue.Full:
                                pass
                
                # Update performance metrics
                total_time = time.time() - start_time
                self.frame_times.append(total_time)
                self.inference_times.append(inference_time)
                
        except Exception as e:
            self.logger.error(f"Error processing camera {camera_id}: {e}")
        finally:
            player.stop()
    
    def start_processing(self):
        """Start processing all camera streams"""
        self.running = True
        
        # Start processing thread for each camera
        for camera_id in self.camera_streams:
            thread = threading.Thread(
                target=self.process_camera_stream,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        # Start alert handler thread
        alert_thread = threading.Thread(target=self.handle_alerts, daemon=True)
        alert_thread.start()
        self.processing_threads.append(alert_thread)
        
        # Start performance monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_performance, daemon=True)
        monitor_thread.start()
        self.processing_threads.append(monitor_thread)
        
        self.logger.info(f"Started processing {len(self.camera_streams)} camera streams")
    
    def stop_processing(self):
        """Stop processing all camera streams"""
        self.running = False
        
        # Wait for all threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=5.0)
        
        self.processing_threads.clear()
        self.logger.info("Stopped all camera stream processing")
    
    def handle_alerts(self):
        """Handle alerts in separate thread"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                # Process alert (save to database, send notifications, etc.)
                self.process_alert(alert)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error handling alert: {e}")
    
    def process_alert(self, alert: AlertInfo):
        """
        Process and handle an alert
        """
        # Save alert image
        if alert.frame is not None:
            alert_dir = Path('alerts') / alert.camera_id
            alert_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(alert.timestamp))
            image_path = alert_dir / f'{timestamp_str}_{alert.anomaly_type}.jpg'
            
            cv2.imwrite(str(image_path), alert.frame)
        
        # Log alert details
        alert_data = {
            'timestamp': alert.timestamp,
            'camera_id': alert.camera_id,
            'anomaly_type': alert.anomaly_type,
            'confidence': alert.confidence,
            'severity': alert.severity,
            'description': alert.description
        }
        
        # Save to alerts log
        alerts_log = Path('alerts/alerts_log.jsonl')
        alerts_log.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alerts_log, 'a') as f:
            json.dump(alert_data, f)
            f.write('\n')
        
        # Here you could add:
        # - Send to database
        # - Send push notifications
        # - Trigger other security systems
        # - Send to web dashboard via WebSocket
    
    def monitor_performance(self):
        """Monitor system performance"""
        while self.running:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                
                if self.frame_times and self.inference_times:
                    avg_frame_time = np.mean(self.frame_times)
                    avg_inference_time = np.mean(self.inference_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    
                    self.logger.info(
                        f"Performance - FPS: {fps:.1f}, "
                        f"Avg Frame Time: {avg_frame_time*1000:.1f}ms, "
                        f"Avg Inference Time: {avg_inference_time*1000:.1f}ms"
                    )
            
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'running': self.running,
            'num_cameras': len(self.camera_streams),
            'active_streams': [
                cam_id for cam_id, stream in self.camera_streams.items()
                if stream.enabled
            ],
            'performance': {
                'avg_fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
                'avg_inference_time_ms': np.mean(self.inference_times) * 1000 if self.inference_times else 0
            },
            'alerts_pending': self.alert_queue.qsize(),
            'openvino_enabled': self.openvino_optimizer is not None
        }
        
        return status


# Example usage and testing
def main():
    """Example usage of the enhanced real-time inference engine"""
    logger = get_app_logger()
    
    # Create inference engine
    inference_engine = EnhancedRealTimeInference()
    
    # Add camera streams (examples)
    # Camera 1: Webcam
    stream1 = CameraStream(
        camera_id="camera_001",
        source=0,  # Webcam
        fps=30,
        confidence_threshold=0.7
    )
    inference_engine.add_camera_stream(stream1)
    
    # Camera 2: RTSP stream (example)
    # stream2 = CameraStream(
    #     camera_id="camera_002",
    #     source="rtsp://192.168.1.100:554/stream",
    #     fps=25,
    #     confidence_threshold=0.75
    # )
    # inference_engine.add_camera_stream(stream2)
    
    try:
        # Start processing
        logger.info("🎥 Starting enhanced real-time inference...")
        inference_engine.start_processing()
        
        # Run for demonstration (in real deployment, this would run indefinitely)
        time.sleep(30)
        
        # Print system status
        status = inference_engine.get_system_status()
        logger.info(f"System Status: {status}")
        
    except KeyboardInterrupt:
        logger.info("Stopping inference engine...")
    finally:
        # Clean shutdown
        inference_engine.stop_processing()
        logger.info("Enhanced real-time inference stopped.")


if __name__ == "__main__":
    main()