"""
Hybrid Multi-Camera Anomaly Detection Model
==========================================

Professional implementation combining YOLO object detection with deep anomaly classification
for real-time multi-camera surveillance systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B3_Weights, ResNet50_Weights
import timm
from ultralytics import YOLO

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import cv2


@dataclass
class DetectionResult:
    """Structure for detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


@dataclass
class AnomalyResult:
    """Structure for anomaly detection results"""
    anomaly_class: str
    confidence: float
    severity: str  # low, medium, high, critical
    timestamp: float
    camera_id: str
    bbox: Optional[Tuple[int, int, int, int]] = None
    objects_detected: Optional[List[DetectionResult]] = None


class ObjectDetector(nn.Module):
    """Professional YOLO-based object detector with weapon detection"""
    
    def __init__(self, model_version: str = "yolov8n", confidence_threshold: float = 0.25):
        """
        Initialize object detector
        
        Args:
            model_version: YOLO model version (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence_threshold: Confidence threshold for detections
        """
        super().__init__()
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        self.yolo = YOLO(f"{model_version}.pt")
        
        # Define dangerous object classes (COCO dataset class IDs)
        self.dangerous_objects = {
            # Weapons and dangerous items
            'knife': [46],  # knife (approximate, may need custom training)
            'scissors': [87],  # scissors  
            'bottle': [39],  # bottle (can be weapon)
            'baseball bat': None,  # Not in COCO, needs custom training
            'gun': None,  # Not in COCO, needs custom training
        }
        
        # High-risk objects that indicate anomalies
        self.risk_objects = {
            'person': 0,
            'car': 2,
            'truck': 7,
            'bus': 5,
            'motorcycle': 3,
            'bicycle': 1,
            'fire hydrant': 10,
            'bottle': 39,
            'knife': 46,
            'scissors': 87
        }
    
    def forward(self, images: torch.Tensor) -> List[List[DetectionResult]]:
        """
        Forward pass for object detection
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            List of detections for each image
        """
        batch_results = []
        
        # Convert torch tensor to numpy for YOLO
        images_np = images.cpu().numpy()
        
        for img_idx in range(images_np.shape[0]):
            # Convert from CHW to HWC and denormalize
            img = images_np[img_idx].transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            
            # Run YOLO detection
            results = self.yolo(img, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        bbox = tuple(boxes[i].astype(int))
                        confidence = float(confidences[i])
                        class_id = int(class_ids[i])
                        class_name = self.yolo.names[class_id]
                        
                        detection = DetectionResult(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
            
            batch_results.append(detections)
        
        return batch_results
    
    def detect_dangerous_objects(self, detections: List[DetectionResult]) -> Dict[str, float]:
        """
        Analyze detections for dangerous objects
        
        Args:
            detections: List of object detections
            
        Returns:
            Dictionary of danger scores for different categories
        """
        danger_scores = {
            'weapon_risk': 0.0,
            'crowd_risk': 0.0,
            'vehicle_risk': 0.0,
            'general_risk': 0.0
        }
        
        person_count = 0
        vehicle_count = 0
        weapon_indicators = 0
        
        for detection in detections:
            class_name = detection.class_name
            confidence = detection.confidence
            
            # Count persons for crowd analysis
            if class_name == 'person':
                person_count += 1
            
            # Count vehicles
            elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_count += 1
            
            # Check for weapon indicators
            elif class_name in ['knife', 'scissors', 'bottle']:
                weapon_indicators += confidence
            
            # General risk objects
            if class_name in self.risk_objects:
                danger_scores['general_risk'] += confidence * 0.1
        
        # Calculate specific risk scores
        danger_scores['crowd_risk'] = min(person_count * 0.1, 1.0)
        danger_scores['vehicle_risk'] = min(vehicle_count * 0.15, 1.0)
        danger_scores['weapon_risk'] = min(weapon_indicators, 1.0)
        
        return danger_scores


class AnomalyClassifier(nn.Module):
    """Advanced anomaly classifier with attention mechanism"""
    
    def __init__(
        self,
        num_classes: int = 14,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize anomaly classifier
        
        Args:
            num_classes: Number of anomaly classes
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Create backbone
        if backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(
                backbone, 
                pretrained=pretrained,
                num_classes=0,  # Remove head
                global_pool=''  # Remove global pooling
            )
            feature_dim = self.backbone.num_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention mechanism
        self.attention = SpatialAttention(feature_dim)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim // 4, num_classes),
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            logits: Class logits [B, num_classes]
            confidence: Confidence scores [B, 1]
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply spatial attention
        attended_features = self.attention(features)
        
        # Global pooling
        pooled_features = self.global_pool(attended_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Feature fusion
        fused_features = self.feature_fusion(pooled_features)
        
        # Classification
        logits = self.classifier(fused_features)
        confidence = self.confidence_head(fused_features)
        
        return logits, confidence


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention"""
        # Generate attention map
        attention = self.conv1(x)
        attention = F.relu(attention, inplace=True)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention
        return x * attention


class MultiCameraFusionModule(nn.Module):
    """Multi-camera fusion with temporal consistency"""
    
    def __init__(self, num_cameras: int = 3, feature_dim: int = 512):
        """
        Initialize multi-camera fusion
        
        Args:
            num_cameras: Maximum number of cameras
            feature_dim: Feature dimension for each camera
        """
        super().__init__()
        self.num_cameras = num_cameras
        self.feature_dim = feature_dim
        
        # Camera-specific encoders
        self.camera_encoders = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_cameras)
        ])
        
        # Cross-camera attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal consistency module
        self.temporal_fusion = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Final fusion
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        camera_features: List[torch.Tensor], 
        temporal_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse multi-camera features
        
        Args:
            camera_features: List of features from each camera [B, feature_dim]
            temporal_features: Optional temporal features [B, seq_len, feature_dim]
            
        Returns:
            fused_score: Combined anomaly score [B, 1]
            attention_weights: Camera attention weights [B, num_cameras]
        """
        batch_size = camera_features[0].size(0)
        
        # Encode camera-specific features
        encoded_features = []
        for i, features in enumerate(camera_features):
            if i < len(self.camera_encoders):
                encoded = self.camera_encoders[i](features)
                encoded_features.append(encoded)
        
        # Stack features for attention
        if encoded_features:
            stacked_features = torch.stack(encoded_features, dim=1)  # [B, num_cameras, feature_dim]
            
            # Apply cross-camera attention
            attended_features, attention_weights = self.cross_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Average across cameras
            averaged_features = attended_features.mean(dim=1)  # [B, feature_dim]
        else:
            # Get device from the first camera encoder parameters
            device = next(self.camera_encoders[0].parameters()).device
            dtype = next(self.camera_encoders[0].parameters()).dtype
            averaged_features = torch.zeros(batch_size, self.feature_dim, device=device, dtype=dtype)
            attention_weights = torch.zeros(batch_size, self.num_cameras, self.feature_dim, device=device, dtype=dtype)
        
        # Apply temporal consistency if available
        if temporal_features is not None:
            # Add current features to temporal sequence
            temporal_input = torch.cat([temporal_features, averaged_features.unsqueeze(1)], dim=1)
            temporal_output, _ = self.temporal_fusion(temporal_input)
            final_features = temporal_output[:, -1, :]  # Take last output
        else:
            # Use a dummy LSTM pass for consistency
            dummy_input = averaged_features.unsqueeze(1)
            final_features, _ = self.temporal_fusion(dummy_input)
            final_features = final_features.squeeze(1)
        
        # Generate final score
        fused_score = self.fusion_head(final_features)
        
        # Extract attention weights (simplified)
        camera_attention = attention_weights.mean(dim=-1)  # [B, num_cameras]
        
        return fused_score, camera_attention


class HybridAnomalyModel(nn.Module):
    """Complete hybrid model for multi-camera anomaly detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize hybrid model
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        # Extract configuration
        model_config = config['model']
        yolo_config = model_config['yolo']
        classifier_config = model_config['anomaly_classifier']
        fusion_config = model_config['fusion']
        
        # Initialize components
        self.object_detector = ObjectDetector(
            model_version=yolo_config['version'],
            confidence_threshold=yolo_config['confidence_threshold']
        )
        
        self.anomaly_classifier = AnomalyClassifier(
            num_classes=classifier_config['num_classes'],
            backbone=classifier_config['backbone'],
            pretrained=classifier_config['pretrained'],
            dropout=classifier_config['dropout']
        )
        
        # Calculate the correct feature dimension for multi-camera fusion
        # The feature_fusion output dimension is feature_dim // 4
        if classifier_config['backbone'].startswith('efficientnet'):
            import timm
            temp_model = timm.create_model(classifier_config['backbone'], pretrained=False, num_classes=0)
            backbone_feature_dim = temp_model.num_features
        elif classifier_config['backbone'] == 'resnet50':
            backbone_feature_dim = 2048  # ResNet50 feature dimension
        else:
            backbone_feature_dim = 1536  # Default fallback
        
        fusion_feature_dim = backbone_feature_dim // 4  # This matches feature_fusion output
        
        self.multi_camera_fusion = MultiCameraFusionModule(
            num_cameras=3,  # Default to 3 cameras
            feature_dim=fusion_feature_dim
        )
        
        # Scoring weights from config
        self.scoring_weights = fusion_config['scoring_weights']
        self.confidence_threshold = fusion_config['confidence_threshold']
        
        # Class names for anomaly detection
        self.class_names = config['dataset']['classes']
        
    def forward(
        self, 
        images: torch.Tensor,
        camera_ids: Optional[List[str]] = None,
        return_objects: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for hybrid model
        
        Args:
            images: Input images [B, C, H, W]
            camera_ids: Optional camera identifiers
            return_objects: Whether to return object detections
            
        Returns:
            Dictionary containing predictions and scores
        """
        batch_size = images.size(0)
        
        # Object detection
        if return_objects:
            object_detections = self.object_detector(images)
            danger_scores = []
            for detections in object_detections:
                danger_score = self.object_detector.detect_dangerous_objects(detections)
                danger_scores.append(danger_score)
        else:
            object_detections = None
            danger_scores = None
        
        # Anomaly classification
        anomaly_logits, anomaly_confidence = self.anomaly_classifier(images)
        anomaly_probs = F.softmax(anomaly_logits, dim=1)
        
        # Extract features for fusion (from classifier's feature fusion layer)
        with torch.no_grad():
            features = self.anomaly_classifier.backbone(images)
            attended_features = self.anomaly_classifier.attention(features)
            pooled_features = self.anomaly_classifier.global_pool(attended_features)
            pooled_features = pooled_features.view(batch_size, -1)
            fusion_features = self.anomaly_classifier.feature_fusion(pooled_features)
        
        # Multi-camera fusion (simulated for single camera, can be extended)
        camera_features = [fusion_features]  # Single camera for now
        fused_score, camera_attention = self.multi_camera_fusion(camera_features)
        
        # Combine scores
        final_scores = anomaly_confidence * 0.7 + fused_score * 0.3
        
        results = {
            'anomaly_logits': anomaly_logits,
            'anomaly_probs': anomaly_probs,
            'anomaly_confidence': anomaly_confidence,
            'fused_score': fused_score,
            'final_scores': final_scores,
            'camera_attention': camera_attention
        }
        
        if return_objects:
            results['object_detections'] = object_detections
            results['danger_scores'] = danger_scores
        
        return results
    
    def predict_anomaly(
        self, 
        images: torch.Tensor, 
        camera_ids: Optional[List[str]] = None
    ) -> List[AnomalyResult]:
        """
        Predict anomalies with full result structure
        
        Args:
            images: Input images
            camera_ids: Camera identifiers
            
        Returns:
            List of anomaly results
        """
        self.eval()
        with torch.no_grad():
            # Get model predictions
            results = self.forward(images, camera_ids, return_objects=True)
            
            anomaly_results = []
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # Get predictions for this image
                probs = results['anomaly_probs'][i]
                confidence = results['final_scores'][i].item()
                
                # Get predicted class
                predicted_class_idx = torch.argmax(probs).item()
                predicted_class = self.class_names[predicted_class_idx]
                class_confidence = probs[predicted_class_idx].item()
                
                # Determine severity
                if confidence >= 0.9:
                    severity = "critical"
                elif confidence >= 0.75:
                    severity = "high"
                elif confidence >= 0.6:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Create result
                camera_id = camera_ids[i] if camera_ids else f"camera_{i}"
                
                anomaly_result = AnomalyResult(
                    anomaly_class=predicted_class,
                    confidence=confidence,
                    severity=severity,
                    timestamp=0.0,  # Will be set by caller
                    camera_id=camera_id,
                    objects_detected=results['object_detections'][i] if 'object_detections' in results else None
                )
                
                anomaly_results.append(anomaly_result)
            
            return anomaly_results
    
    def train(self, mode: bool = True):
        """
        Override train method to handle YOLO component properly
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        # Set PyTorch modules to train/eval mode
        self.anomaly_classifier.train(mode)
        self.multi_camera_fusion.train(mode)
        
        # For YOLO, we don't call train() as it triggers YOLO's training pipeline
        # YOLO model is typically kept in eval mode for feature extraction
        # during our custom training loop
        
        return self
    
    def eval(self):
        """Override eval method to handle YOLO component properly"""
        return self.train(False)


def create_model(config: Dict) -> HybridAnomalyModel:
    """Factory function to create hybrid model"""
    return HybridAnomalyModel(config)


# Utility functions for model optimization
def optimize_inference(model: nn.Module) -> nn.Module:
    """Optimize model for inference using TorchScript"""
    model.eval()
    return torch.jit.script(model)


def load_pretrained_weights(model: HybridAnomalyModel, weights_path: str):
    """Load pretrained weights"""
    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def main():
    """Test model creation and forward pass"""
    from src.utils.config import get_config
    
    config = get_config()
    
    # Create model
    model = create_model(config.config)
    print(f"✅ Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        results = model(dummy_input, return_objects=True)
        print(f"✅ Forward pass successful!")
        print(f"Anomaly logits shape: {results['anomaly_logits'].shape}")
        print(f"Final scores shape: {results['final_scores'].shape}")
        
        # Test prediction
        predictions = model.predict_anomaly(dummy_input)
        print(f"✅ Prediction successful!")
        for i, pred in enumerate(predictions):
            print(f"Sample {i}: {pred.anomaly_class} (confidence: {pred.confidence:.3f}, severity: {pred.severity})")


if __name__ == "__main__":
    main()