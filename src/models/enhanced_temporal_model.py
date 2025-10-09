"""
Enhanced Temporal Anomaly Detection Model
=========================================

Implements the CNN-LSTM architecture from the technical report with modern enhancements:
- InceptionV3 feature extraction with temporal LSTM processing
- Multi-scale feature fusion
- Attention mechanisms for better temporal understanding
- OpenVINO optimization support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2
import time


class InceptionV3FeatureExtractor(nn.Module):
    """
    Enhanced InceptionV3 feature extractor based on technical report
    """
    
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pre-trained InceptionV3
        self.inception = models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True
        )
        
        # Remove the final classification layer
        self.inception.fc = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.inception.parameters():
                param.requires_grad = False
        
        # Global average pooling to get fixed-size features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature dimension: 2048 (InceptionV3 output)
        self.feature_dim = 2048
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor [B, C, H, W] or [B*T, C, H, W] for video sequences
            
        Returns:
            Feature tensor [B, feature_dim] or [B*T, feature_dim]
        """
        # Ensure input is the right size for InceptionV3 (299x299)
        if x.shape[-1] != 299 or x.shape[-2] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Extract features using InceptionV3
        # When training with aux_logits=True, inception returns (main_output, aux_output)
        features = self.inception(x)
        if isinstance(features, tuple):
            features = features[0]  # Use only the main output
        
        return features


class TemporalAttention(nn.Module):
    """
    Multi-head attention mechanism for temporal sequence modeling
    """
    
    def __init__(self, feature_dim: int = 2048, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention
        
        Args:
            x: Input tensor [B, T, feature_dim]
            
        Returns:
            Attended features [B, T, feature_dim]
        """
        B, T, D = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        
        # Output projection
        output = self.output_proj(attn_output)
        
        return output + x  # Residual connection


class EnhancedLSTMClassifier(nn.Module):
    """
    Enhanced LSTM classifier with attention and multi-scale processing
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 14,
        dropout: float = 0.4,
        use_attention: bool = True,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # Temporal attention if enabled
        if use_attention:
            self.temporal_attention = TemporalAttention(feature_dim)
        
        # Multi-layer LSTM
        self.lstm_layers = nn.ModuleList()
        input_dim = feature_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if i < len(hidden_dims) - 1 else 0
            )
            self.lstm_layers.append(lstm)
            
            # Update input dimension for next layer
            input_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Classification head
        final_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
        
        # Auxiliary heads for multi-scale prediction
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(hidden_dims[i] * (2 if bidirectional else 1), num_classes)
            for i in range(len(hidden_dims))
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSTM classifier
        
        Args:
            x: Input features [B, T, feature_dim]
            
        Returns:
            Dictionary containing main and auxiliary predictions
        """
        # Apply temporal attention if enabled
        if self.use_attention:
            x = self.temporal_attention(x)
        
        # Process through LSTM layers
        lstm_outputs = []
        current_input = x
        
        for i, lstm in enumerate(self.lstm_layers):
            output, (hidden, cell) = lstm(current_input)
            lstm_outputs.append(output)
            current_input = output
        
        # Main prediction from final LSTM output
        # Use the last timestep for classification
        final_output = lstm_outputs[-1][:, -1, :]  # [B, hidden_dim]
        main_pred = self.classifier(final_output)
        
        # Auxiliary predictions for multi-scale learning
        aux_preds = []
        for i, (output, aux_classifier) in enumerate(zip(lstm_outputs, self.aux_classifiers)):
            aux_pred = aux_classifier(output[:, -1, :])
            aux_preds.append(aux_pred)
        
        return {
            'main': main_pred,
            'auxiliary': aux_preds,
            'features': final_output
        }


class EnhancedTemporalAnomalyModel(nn.Module):
    """
    Complete temporal anomaly detection model combining CNN and LSTM
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        max_seq_length: int = 32,
        feature_dim: int = 2048,
        lstm_hidden_dims: List[int] = [512, 256],
        dropout: float = 0.4,
        use_attention: bool = True,
        freeze_cnn: bool = True,
        feature_based: bool = False  # NEW: Support for pre-extracted features
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.feature_dim = feature_dim
        self.feature_based = feature_based
        
        # Feature extractor (CNN) - only needed if not using pre-extracted features
        if not feature_based:
            self.feature_extractor = InceptionV3FeatureExtractor(
                pretrained=True,
                freeze_backbone=freeze_cnn
            )
        
        # Temporal classifier (LSTM + Attention)
        self.temporal_classifier = EnhancedLSTMClassifier(
            feature_dim=feature_dim,
            hidden_dims=lstm_hidden_dims,
            num_classes=num_classes,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Class names (from technical report)
        self.class_names = [
            'Abuse', 'Arson', 'Assault', 'Burglary', 'Explosion',
            'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting',
            'Shoplifting', 'Stealing', 'Vandalism', 'Arrest'
        ]
        
    def extract_features_batch(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video batch
        
        Args:
            video_batch: [B, T, C, H, W]
            
        Returns:
            Features: [B, T, feature_dim]
        """
        if not hasattr(self, 'feature_extractor'):
            raise RuntimeError("Feature extractor not available in feature-based mode")
            
        B, T, C, H, W = video_batch.shape
        
        # Reshape to process all frames together
        frames = video_batch.view(B * T, C, H, W)
        
        # Extract features
        with torch.no_grad() if self.training else torch.enable_grad():
            features = self.feature_extractor(frames)  # [B*T, feature_dim]
        
        # Reshape back to sequence format
        features = features.view(B, T, self.feature_dim)
        
        return features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model
        
        Args:
            x: Input video tensor [B, T, C, H, W] or pre-computed features [B, T, feature_dim]
            
        Returns:
            Predictions and features
        """
        # Check if input is raw video or pre-computed features
        if x.dim() == 5:  # Raw video: [B, T, C, H, W]
            features = self.extract_features_batch(x)
        else:  # Pre-computed features: [B, T, feature_dim]
            features = x
        
        # Temporal classification
        predictions = self.temporal_classifier(features)
        
        return predictions
    
    def predict_single_video(self, video: torch.Tensor, threshold: float = 0.5) -> Dict:
        """
        Predict anomaly for single video with detailed output
        
        Args:
            video: Single video tensor [T, C, H, W] or [1, T, C, H, W]
            threshold: Confidence threshold
            
        Returns:
            Detailed prediction results
        """
        self.eval()
        
        # Ensure batch dimension
        if video.dim() == 4:
            video = video.unsqueeze(0)
        
        with torch.no_grad():
            # Get predictions
            outputs = self.forward(video)
            main_pred = outputs['main']
            
            # Get probabilities
            probs = F.softmax(main_pred, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
            # Convert to numpy for processing
            confidence = confidence.item()
            predicted_class = predicted_class.item()
            class_probs = probs[0].cpu().numpy()
            
            # Get top 3 predictions
            top_indices = np.argsort(class_probs)[::-1][:3]
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'confidence': float(class_probs[idx]),
                    'class_id': int(idx)
                }
                for idx in top_indices
            ]
            
            # Determine if anomaly is detected
            is_anomaly = (confidence > threshold) and (predicted_class != 6)  # 6 is 'Normal'
            
            # Determine severity based on confidence and class
            severity = 'low'
            if confidence > 0.8:
                if predicted_class in [4, 9]:  # Explosion, Shooting
                    severity = 'critical'
                elif predicted_class in [2, 3, 5, 8]:  # Assault, Burglary, Fighting, Robbery
                    severity = 'high'
                else:
                    severity = 'medium'
            elif confidence > 0.65:
                severity = 'medium'
            
            return {
                'is_anomaly': is_anomaly,
                'predicted_class': self.class_names[predicted_class],
                'confidence': confidence,
                'severity': severity,
                'top_predictions': top_predictions,
                'class_probabilities': class_probs.tolist(),
                'features': outputs['features'][0].cpu().numpy()
            }


def create_enhanced_temporal_model(config: Dict) -> EnhancedTemporalAnomalyModel:
    """
    Factory function to create enhanced temporal model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured model instance
    """
    model_config = config.get('model', {}).get('temporal', {})
    
    model = EnhancedTemporalAnomalyModel(
        num_classes=model_config.get('num_classes', 14),
        max_seq_length=model_config.get('max_seq_length', 32),
        feature_dim=model_config.get('feature_dim', 2048),
        lstm_hidden_dims=model_config.get('lstm_hidden_dims', [512, 256]),
        dropout=model_config.get('dropout', 0.4),
        use_attention=model_config.get('use_attention', True),
        freeze_cnn=model_config.get('freeze_cnn', True),
        feature_based=model_config.get('feature_based', False)  # Support for optimization
    )
    
    return model


# Video processing utilities for real-time inference
class VideoProcessor:
    """
    Video processor for real-time inference similar to technical report
    """
    
    def __init__(self, model: EnhancedTemporalAnomalyModel, device: torch.device):
        self.model = model
        self.device = device
        self.frame_buffer = []
        self.max_seq_length = model.max_seq_length
        
        # Preprocessing parameters
        self.img_size = (299, 299)  # InceptionV3 input size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, self.img_size)
        
        # Normalize
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - self.mean) / self.std
        
        # Convert to tensor and add channel dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1)
        
        return frame_tensor
    
    def add_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Add frame to buffer and predict if sequence is complete"""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        
        # Keep only the last max_seq_length frames
        if len(self.frame_buffer) > self.max_seq_length:
            self.frame_buffer = self.frame_buffer[-self.max_seq_length:]
        
        # Predict if we have enough frames
        if len(self.frame_buffer) == self.max_seq_length:
            # Stack frames and add batch dimension
            video_tensor = torch.stack(self.frame_buffer).unsqueeze(0).to(self.device)
            
            # Get prediction
            result = self.model.predict_single_video(video_tensor)
            
            # Clear buffer for next sequence (or use sliding window)
            self.frame_buffer = self.frame_buffer[1:]  # Sliding window
            
            return result
        
        return None


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    config = {
        'model': {
            'temporal': {
                'num_classes': 14,
                'max_seq_length': 32,
                'use_attention': True
            }
        }
    }
    
    model = create_enhanced_temporal_model(config).to(device)
    
    # Test with random input
    batch_size = 2
    seq_length = 32
    test_input = torch.randn(batch_size, seq_length, 3, 299, 299).to(device)
    
    # Forward pass
    outputs = model(test_input)
    print(f"Main prediction shape: {outputs['main'].shape}")
    print(f"Number of auxiliary predictions: {len(outputs['auxiliary'])}")
    
    print("Enhanced Temporal Model created successfully!")