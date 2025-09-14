"""
Multi-Camera Fusion and Intelligent Scoring System
==================================================

Advanced system for combining multiple camera feeds with spatial-temporal analysis,
intelligent scoring, and emergency contact selection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import time
import math
from enum import Enum

from src.models.hybrid_model import AnomalyResult, DetectionResult
from src.utils.logging_config import get_app_logger


class SeverityLevel(Enum):
    """Severity levels for anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyType(Enum):
    """Types of emergency contacts"""
    POLICE = "police"
    MEDICAL = "medical"
    FIRE = "fire"
    SECURITY = "security"
    NONE = "none"


@dataclass
class CameraScore:
    """Individual camera scoring data"""
    camera_id: str
    anomaly_score: float
    confidence: float
    spatial_weight: float
    temporal_consistency: float
    object_density: float
    movement_intensity: float
    timestamp: float


@dataclass
class FusedDetection:
    """Fused detection result from multiple cameras"""
    anomaly_class: str
    combined_score: float
    severity: SeverityLevel
    emergency_type: EmergencyType
    contributing_cameras: List[str]
    spatial_center: Tuple[float, float]
    temporal_window: float
    confidence_distribution: Dict[str, float]
    recommended_actions: List[str]
    timestamp: float


class SpatialAnalyzer:
    """Analyzes spatial relationships between cameras"""
    
    def __init__(self, camera_positions: Dict[str, Tuple[float, float]]):
        """
        Initialize spatial analyzer
        
        Args:
            camera_positions: Dictionary mapping camera_id to (x, y) position
        """
        self.camera_positions = camera_positions
        self.logger = get_app_logger()
        
        # Calculate distance matrix between cameras
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Calculate distance matrix between all camera pairs"""
        distance_matrix = {}
        camera_ids = list(self.camera_positions.keys())
        
        for i, cam1 in enumerate(camera_ids):
            for j, cam2 in enumerate(camera_ids):
                if i != j:
                    pos1 = self.camera_positions[cam1]
                    pos2 = self.camera_positions[cam2]
                    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    distance_matrix[(cam1, cam2)] = distance
                else:
                    distance_matrix[(cam1, cam2)] = 0.0
        
        return distance_matrix
    
    def calculate_spatial_correlation(
        self, 
        camera_scores: List[CameraScore],
        correlation_radius: float = 10.0
    ) -> Dict[str, float]:
        """
        Calculate spatial correlation scores for each camera
        
        Args:
            camera_scores: List of camera scores
            correlation_radius: Maximum distance for correlation
            
        Returns:
            Dictionary of spatial correlation scores
        """
        correlations = {}
        
        for score in camera_scores:
            camera_id = score.camera_id
            correlation = 0.0
            total_weight = 0.0
            
            # Check correlation with other cameras
            for other_score in camera_scores:
                if other_score.camera_id == camera_id:
                    continue
                
                # Get distance between cameras
                distance = self.distance_matrix.get((camera_id, other_score.camera_id), float('inf'))
                
                if distance <= correlation_radius:
                    # Calculate correlation weight (inverse distance)
                    weight = 1.0 / (1.0 + distance)
                    
                    # Weighted correlation based on anomaly scores
                    correlation += weight * other_score.anomaly_score * score.anomaly_score
                    total_weight += weight
            
            # Normalize correlation
            if total_weight > 0:
                correlations[camera_id] = correlation / total_weight
            else:
                correlations[camera_id] = 0.0
        
        return correlations
    
    def find_spatial_clusters(
        self, 
        camera_scores: List[CameraScore],
        cluster_radius: float = 5.0,
        min_score: float = 0.5
    ) -> List[List[str]]:
        """
        Find spatial clusters of high-scoring cameras
        
        Args:
            camera_scores: List of camera scores
            cluster_radius: Maximum distance for clustering
            min_score: Minimum score to consider for clustering
            
        Returns:
            List of camera clusters
        """
        # Filter high-scoring cameras
        high_score_cameras = [
            score.camera_id for score in camera_scores 
            if score.anomaly_score >= min_score
        ]
        
        if not high_score_cameras:
            return []
        
        # Simple clustering algorithm
        clusters = []
        visited = set()
        
        for camera_id in high_score_cameras:
            if camera_id in visited:
                continue
            
            # Start new cluster
            cluster = [camera_id]
            visited.add(camera_id)
            queue = [camera_id]
            
            while queue:
                current_cam = queue.pop(0)
                
                # Find nearby cameras
                for other_cam in high_score_cameras:
                    if other_cam in visited:
                        continue
                    
                    distance = self.distance_matrix.get((current_cam, other_cam), float('inf'))
                    
                    if distance <= cluster_radius:
                        cluster.append(other_cam)
                        visited.add(other_cam)
                        queue.append(other_cam)
            
            if len(cluster) > 1:  # Only keep clusters with multiple cameras
                clusters.append(cluster)
        
        return clusters


class TemporalAnalyzer:
    """Analyzes temporal consistency and patterns"""
    
    def __init__(self, window_size: int = 30, consistency_threshold: float = 0.7):
        """
        Initialize temporal analyzer
        
        Args:
            window_size: Number of frames to consider for temporal analysis
            consistency_threshold: Threshold for temporal consistency
        """
        self.window_size = window_size
        self.consistency_threshold = consistency_threshold
        self.logger = get_app_logger()
        
        # Store temporal data for each camera
        self.temporal_data = {}  # camera_id -> deque of scores
        
    def update_temporal_data(self, camera_scores: List[CameraScore]):
        """Update temporal data with new scores"""
        for score in camera_scores:
            camera_id = score.camera_id
            
            if camera_id not in self.temporal_data:
                self.temporal_data[camera_id] = deque(maxlen=self.window_size)
            
            self.temporal_data[camera_id].append({
                'score': score.anomaly_score,
                'timestamp': score.timestamp,
                'confidence': score.confidence
            })
    
    def calculate_temporal_consistency(self, camera_id: str) -> float:
        """
        Calculate temporal consistency for a camera
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Temporal consistency score [0, 1]
        """
        if camera_id not in self.temporal_data:
            return 0.0
        
        data = list(self.temporal_data[camera_id])
        
        if len(data) < 3:
            return 0.0
        
        # Calculate variance in scores
        scores = [d['score'] for d in data]
        mean_score = np.mean(scores)
        
        if mean_score == 0:
            return 0.0
        
        # Consistency based on coefficient of variation
        std_score = np.std(scores)
        cv = std_score / mean_score if mean_score > 0 else float('inf')
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = 1.0 / (1.0 + cv)
        
        return consistency
    
    def detect_temporal_patterns(self, camera_id: str) -> Dict[str, float]:
        """
        Detect temporal patterns in camera data
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Dictionary of pattern scores
        """
        if camera_id not in self.temporal_data:
            return {}
        
        data = list(self.temporal_data[camera_id])
        
        if len(data) < 5:
            return {}
        
        scores = np.array([d['score'] for d in data])
        timestamps = np.array([d['timestamp'] for d in data])
        
        patterns = {}
        
        # Trend analysis
        if len(scores) > 1:
            # Linear trend
            time_diff = timestamps[-1] - timestamps[0]
            score_diff = scores[-1] - scores[0]
            
            if time_diff > 0:
                patterns['trend'] = score_diff / time_diff
            else:
                patterns['trend'] = 0.0
        
        # Sudden spike detection
        if len(scores) >= 3:
            recent_avg = np.mean(scores[-3:])
            historical_avg = np.mean(scores[:-3]) if len(scores) > 3 else 0
            
            patterns['sudden_spike'] = max(0, recent_avg - historical_avg)
        
        # Persistence (how long high scores are maintained)
        high_score_count = np.sum(scores > 0.7)
        patterns['persistence'] = high_score_count / len(scores)
        
        return patterns


class IntelligentScoring:
    """Advanced scoring system combining multiple factors"""
    
    def __init__(self, config: Dict):
        """
        Initialize intelligent scoring
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_app_logger()
        
        # Scoring weights
        self.weights = {
            'base_score': 0.4,
            'spatial_correlation': 0.2,
            'temporal_consistency': 0.2,
            'object_context': 0.1,
            'movement_intensity': 0.1
        }
        
        # Emergency contact mapping
        self.emergency_mapping = {
            'Shooting': EmergencyType.POLICE,
            'Robbery': EmergencyType.POLICE,
            'Assault': EmergencyType.POLICE,
            'Fighting': EmergencyType.POLICE,
            'Abuse': EmergencyType.POLICE,
            'RoadAccidents': EmergencyType.MEDICAL,
            'Explosion': EmergencyType.FIRE,
            'Arson': EmergencyType.FIRE,
            'Burglary': EmergencyType.SECURITY,
            'Vandalism': EmergencyType.SECURITY,
            'Shoplifting': EmergencyType.SECURITY,
            'Stealing': EmergencyType.SECURITY,
            'Arrest': EmergencyType.NONE,
            'NormalVideos': EmergencyType.NONE
        }
        
        # Action recommendations
        self.action_recommendations = {
            EmergencyType.POLICE: [
                "Contact local police immediately",
                "Secure the area",
                "Preserve video evidence",
                "Assist victims if safe to do so"
            ],
            EmergencyType.MEDICAL: [
                "Call emergency medical services",
                "Clear access routes for ambulances",
                "Check for injuries",
                "Provide first aid if trained"
            ],
            EmergencyType.FIRE: [
                "Contact fire department immediately",
                "Evacuate the area",
                "Check for additional fire hazards",
                "Monitor for smoke and gases"
            ],
            EmergencyType.SECURITY: [
                "Alert security personnel",
                "Monitor suspect movement",
                "Secure valuable items",
                "Document incident details"
            ],
            EmergencyType.NONE: [
                "Continue monitoring",
                "Document incident",
                "Review footage if needed"
            ]
        }
    
    def calculate_combined_score(
        self,
        camera_scores: List[CameraScore],
        spatial_correlations: Dict[str, float],
        temporal_consistencies: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate combined scores for all cameras
        
        Args:
            camera_scores: Individual camera scores
            spatial_correlations: Spatial correlation scores
            temporal_consistencies: Temporal consistency scores
            
        Returns:
            Dictionary of combined scores
        """
        combined_scores = {}
        
        for score in camera_scores:
            camera_id = score.camera_id
            
            # Base anomaly score
            base_score = score.anomaly_score * self.weights['base_score']
            
            # Spatial correlation component
            spatial_score = spatial_correlations.get(camera_id, 0.0) * self.weights['spatial_correlation']
            
            # Temporal consistency component
            temporal_score = temporal_consistencies.get(camera_id, 0.0) * self.weights['temporal_consistency']
            
            # Object context component
            object_score = score.object_density * self.weights['object_context']
            
            # Movement intensity component
            movement_score = score.movement_intensity * self.weights['movement_intensity']
            
            # Combine all components
            combined_score = (
                base_score + spatial_score + temporal_score + 
                object_score + movement_score
            )
            
            # Apply camera-specific weight
            combined_score *= score.spatial_weight
            
            combined_scores[camera_id] = min(combined_score, 1.0)  # Cap at 1.0
        
        return combined_scores
    
    def determine_severity(self, score: float, anomaly_class: str) -> SeverityLevel:
        """
        Determine severity level based on score and anomaly class
        
        Args:
            score: Combined anomaly score
            anomaly_class: Type of anomaly
            
        Returns:
            Severity level
        """
        # Critical anomalies that are always high priority
        critical_classes = {'Shooting', 'Explosion', 'RoadAccidents'}
        high_priority_classes = {'Robbery', 'Assault', 'Fighting', 'Arson'}
        
        if anomaly_class in critical_classes:
            if score >= 0.6:
                return SeverityLevel.CRITICAL
            elif score >= 0.4:
                return SeverityLevel.HIGH
            else:
                return SeverityLevel.MEDIUM
        
        elif anomaly_class in high_priority_classes:
            if score >= 0.8:
                return SeverityLevel.CRITICAL
            elif score >= 0.6:
                return SeverityLevel.HIGH
            elif score >= 0.4:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
        
        else:
            # Standard severity mapping
            if score >= 0.9:
                return SeverityLevel.CRITICAL
            elif score >= 0.7:
                return SeverityLevel.HIGH
            elif score >= 0.5:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.LOW
    
    def get_emergency_type(self, anomaly_class: str) -> EmergencyType:
        """Get emergency contact type for anomaly class"""
        return self.emergency_mapping.get(anomaly_class, EmergencyType.SECURITY)
    
    def get_recommended_actions(self, emergency_type: EmergencyType) -> List[str]:
        """Get recommended actions for emergency type"""
        return self.action_recommendations.get(emergency_type, [])


class MultiCameraFusionSystem:
    """Main fusion system coordinating all components"""
    
    def __init__(
        self, 
        camera_positions: Dict[str, Tuple[float, float]],
        config: Dict
    ):
        """
        Initialize fusion system
        
        Args:
            camera_positions: Dictionary mapping camera_id to position
            config: Configuration dictionary
        """
        self.camera_positions = camera_positions
        self.config = config
        self.logger = get_app_logger()
        
        # Initialize components
        self.spatial_analyzer = SpatialAnalyzer(camera_positions)
        self.temporal_analyzer = TemporalAnalyzer(
            window_size=config.get('fusion', {}).get('temporal_window', 30)
        )
        self.intelligent_scoring = IntelligentScoring(config)
        
        # Fusion parameters
        self.confidence_threshold = config.get('fusion', {}).get('confidence_threshold', 0.75)
        self.min_cameras_for_fusion = 2
        
    def process_camera_results(
        self,
        anomaly_results: List[AnomalyResult]
    ) -> List[FusedDetection]:
        """
        Process results from multiple cameras and generate fused detections
        
        Args:
            anomaly_results: List of anomaly results from different cameras
            
        Returns:
            List of fused detections
        """
        if not anomaly_results:
            return []
        
        # Convert to camera scores
        camera_scores = self._convert_to_camera_scores(anomaly_results)
        
        # Update temporal data
        self.temporal_analyzer.update_temporal_data(camera_scores)
        
        # Calculate spatial correlations
        spatial_correlations = self.spatial_analyzer.calculate_spatial_correlation(camera_scores)
        
        # Calculate temporal consistencies
        temporal_consistencies = {}
        for score in camera_scores:
            temporal_consistencies[score.camera_id] = \
                self.temporal_analyzer.calculate_temporal_consistency(score.camera_id)
        
        # Calculate combined scores
        combined_scores = self.intelligent_scoring.calculate_combined_score(
            camera_scores, spatial_correlations, temporal_consistencies
        )
        
        # Find spatial clusters
        clusters = self.spatial_analyzer.find_spatial_clusters(camera_scores)
        
        # Generate fused detections
        fused_detections = []
        
        # Process clusters
        for cluster in clusters:
            cluster_scores = [combined_scores[cam_id] for cam_id in cluster]
            max_score = max(cluster_scores)
            
            if max_score >= self.confidence_threshold:
                # Find the anomaly class with highest confidence in cluster
                cluster_anomalies = [
                    result for result in anomaly_results 
                    if result.camera_id in cluster
                ]
                
                if cluster_anomalies:
                    best_anomaly = max(cluster_anomalies, key=lambda x: x.confidence)
                    
                    fused_detection = self._create_fused_detection(
                        cluster, best_anomaly, max_score, combined_scores
                    )
                    fused_detections.append(fused_detection)
        
        # Process individual high-confidence detections not in clusters
        for result in anomaly_results:
            if result.camera_id not in [cam for cluster in clusters for cam in cluster]:
                individual_score = combined_scores.get(result.camera_id, 0.0)
                
                if individual_score >= self.confidence_threshold:
                    fused_detection = self._create_fused_detection(
                        [result.camera_id], result, individual_score, combined_scores
                    )
                    fused_detections.append(fused_detection)
        
        return fused_detections
    
    def _convert_to_camera_scores(self, anomaly_results: List[AnomalyResult]) -> List[CameraScore]:
        """Convert anomaly results to camera scores"""
        camera_scores = []
        
        for result in anomaly_results:
            # Calculate additional metrics
            object_density = len(result.objects_detected) / 10.0 if result.objects_detected else 0.0
            movement_intensity = min(object_density * 0.5, 1.0)  # Simplified calculation
            
            # Get spatial weight from camera position
            spatial_weight = 1.0  # Default weight, can be configured per camera
            
            score = CameraScore(
                camera_id=result.camera_id,
                anomaly_score=result.confidence,
                confidence=result.confidence,
                spatial_weight=spatial_weight,
                temporal_consistency=0.0,  # Will be calculated
                object_density=object_density,
                movement_intensity=movement_intensity,
                timestamp=result.timestamp
            )
            
            camera_scores.append(score)
        
        return camera_scores
    
    def _create_fused_detection(
        self,
        camera_cluster: List[str],
        best_anomaly: AnomalyResult,
        combined_score: float,
        all_scores: Dict[str, float]
    ) -> FusedDetection:
        """Create fused detection from cluster data"""
        # Calculate spatial center
        if len(camera_cluster) == 1:
            spatial_center = self.camera_positions.get(camera_cluster[0], (0.0, 0.0))
        else:
            positions = [self.camera_positions.get(cam_id, (0.0, 0.0)) for cam_id in camera_cluster]
            spatial_center = (
                np.mean([pos[0] for pos in positions]),
                np.mean([pos[1] for pos in positions])
            )
        
        # Determine severity and emergency type
        severity = self.intelligent_scoring.determine_severity(combined_score, best_anomaly.anomaly_class)
        emergency_type = self.intelligent_scoring.get_emergency_type(best_anomaly.anomaly_class)
        
        # Get confidence distribution
        confidence_distribution = {
            cam_id: all_scores.get(cam_id, 0.0) for cam_id in camera_cluster
        }
        
        # Get recommended actions
        recommended_actions = self.intelligent_scoring.get_recommended_actions(emergency_type)
        
        return FusedDetection(
            anomaly_class=best_anomaly.anomaly_class,
            combined_score=combined_score,
            severity=severity,
            emergency_type=emergency_type,
            contributing_cameras=camera_cluster,
            spatial_center=spatial_center,
            temporal_window=30.0,  # From config
            confidence_distribution=confidence_distribution,
            recommended_actions=recommended_actions,
            timestamp=time.time()
        )


def main():
    """Test the fusion system"""
    from src.utils.config import get_config
    
    config = get_config()
    
    # Demo camera positions
    camera_positions = {
        "camera_1": (0.0, 0.0),
        "camera_2": (5.0, 0.0),
        "camera_3": (10.0, 5.0)
    }
    
    # Initialize fusion system
    fusion_system = MultiCameraFusionSystem(camera_positions, config.config)
    
    # Create demo anomaly results
    demo_results = [
        AnomalyResult(
            anomaly_class="Robbery",
            confidence=0.85,
            severity="high",
            timestamp=time.time(),
            camera_id="camera_1"
        ),
        AnomalyResult(
            anomaly_class="Robbery",
            confidence=0.78,
            severity="high", 
            timestamp=time.time(),
            camera_id="camera_2"
        )
    ]
    
    # Process results
    fused_detections = fusion_system.process_camera_results(demo_results)
    
    print("🔍 Fusion System Test Results:")
    for detection in fused_detections:
        print(f"  Anomaly: {detection.anomaly_class}")
        print(f"  Score: {detection.combined_score:.3f}")
        print(f"  Severity: {detection.severity.value}")
        print(f"  Emergency Type: {detection.emergency_type.value}")
        print(f"  Cameras: {detection.contributing_cameras}")
        print(f"  Actions: {detection.recommended_actions}")
        print("  ---")


if __name__ == "__main__":
    main()