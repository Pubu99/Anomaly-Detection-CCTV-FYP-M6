"""
API Schemas for Multi-Camera Anomaly Detection System
=====================================================

Pydantic models for API request/response validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


# Enums
class CameraStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class EmergencyType(str, Enum):
    POLICE = "police"
    MEDICAL = "medical"
    FIRE = "fire"
    SECURITY = "security"


class ModelType(str, Enum):
    HYBRID = "hybrid"
    YOLO = "yolo"
    EFFICIENTNET = "efficientnet"


class TrainingStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Camera Schemas
class CameraBase(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    name: str = Field(..., description="Human-readable camera name")
    location: str = Field(..., description="Camera location description")
    camera_url: str = Field(..., description="Camera URL or device index")
    position_x: float = Field(default=0.0, description="X coordinate position")
    position_y: float = Field(default=0.0, description="Y coordinate position")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Camera importance weight")
    resolution_width: int = Field(default=1920, gt=0, description="Camera resolution width")
    resolution_height: int = Field(default=1080, gt=0, description="Camera resolution height")
    fps: int = Field(default=30, gt=0, le=120, description="Camera FPS")
    enabled: bool = Field(default=True, description="Whether camera is enabled")


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    camera_url: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution_width: Optional[int] = Field(None, gt=0)
    resolution_height: Optional[int] = Field(None, gt=0)
    fps: Optional[int] = Field(None, gt=0, le=120)
    enabled: Optional[bool] = None


class CameraResponse(CameraBase):
    id: int
    status: CameraStatus
    created_at: datetime
    last_seen: Optional[datetime] = None
    last_frame_time: Optional[datetime] = None
    total_frames_processed: int = 0
    error_count: int = 0
    
    class Config:
        from_attributes = True


class CameraPerformanceResponse(BaseModel):
    camera_id: str
    timestamp: datetime
    fps: float
    frame_drops: int
    avg_inference_time: float
    detection_count: int
    alert_count: int
    false_positive_rate: float
    connection_errors: int
    quality_score: float
    
    class Config:
        from_attributes = True


# Alert Schemas
class AlertBase(BaseModel):
    camera_id: str
    anomaly_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: AlertSeverity
    bbox: Optional[List[float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2]")
    objects_detected: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertCreate(AlertBase):
    pass


class AlertUpdate(BaseModel):
    status: Optional[AlertStatus] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None


class AlertResponse(AlertBase):
    id: int
    status: AlertStatus
    frame_path: Optional[str] = None
    video_clip_path: Optional[str] = None
    emergency_type: Optional[EmergencyType] = None
    emergency_contact_sent: bool = False
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    class Config:
        from_attributes = True


# Feedback Schemas
class FeedbackBase(BaseModel):
    detection_id: str = Field(..., description="Detection ID to provide feedback for")
    is_correct: bool = Field(..., description="Whether the detection was correct")
    correct_label: Optional[str] = Field(None, description="Correct label if detection was wrong")
    confidence_rating: float = Field(..., ge=0.0, le=1.0, description="User confidence in feedback")
    comments: Optional[str] = Field(None, description="Additional comments")
    feedback_type: str = Field(default="manual", description="Type of feedback")


class FeedbackCreate(FeedbackBase):
    user_id: Optional[str] = None


class FeedbackResponse(FeedbackBase):
    id: int
    alert_id: Optional[int] = None
    user_id: Optional[str] = None
    created_at: datetime
    processed: bool = False
    
    class Config:
        from_attributes = True


# Model Schemas
class ModelVersionBase(BaseModel):
    version: str
    model_type: ModelType = ModelType.HYBRID
    model_path: str
    config_path: Optional[str] = None
    training_data_size: Optional[int] = None
    training_duration: Optional[float] = None
    training_epochs: Optional[int] = None


class ModelVersionCreate(ModelVersionBase):
    created_by: Optional[str] = None


class ModelVersionResponse(ModelVersionBase):
    id: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    validation_loss: Optional[float] = None
    is_active: bool = False
    deployment_date: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    created_by: Optional[str] = None
    
    class Config:
        from_attributes = True


# Training Schemas
class TrainingConfigSchema(BaseModel):
    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=32, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)
    optimizer: str = Field(default="adam")
    loss_function: str = Field(default="focal")
    data_augmentation: bool = Field(default=True)
    validation_split: float = Field(default=0.2, gt=0, lt=1)
    early_stopping: bool = Field(default=True)
    patience: int = Field(default=10, gt=0)


class TrainingSessionCreate(BaseModel):
    trigger_type: str = Field(..., description="What triggered the training")
    training_config: TrainingConfigSchema
    data_version: Optional[str] = None


class TrainingSessionResponse(BaseModel):
    id: int
    session_id: str
    trigger_type: str
    data_version: Optional[str] = None
    training_config: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: TrainingStatus
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: Optional[int] = None
    best_accuracy: Optional[float] = None
    best_loss: Optional[float] = None
    final_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_model_path: Optional[str] = None
    
    class Config:
        from_attributes = True


# Emergency Contact Schemas
class EmergencyContactBase(BaseModel):
    contact_type: EmergencyType
    name: str
    phone: str
    email: Optional[str] = None
    department: Optional[str] = None
    priority: int = Field(default=1, ge=1)
    location_area: Optional[str] = None
    availability_hours: Optional[Dict[str, Any]] = None
    response_time: Optional[float] = Field(None, gt=0)


class EmergencyContactCreate(EmergencyContactBase):
    pass


class EmergencyContactResponse(EmergencyContactBase):
    id: int
    is_active: bool = True
    created_at: datetime
    last_contacted: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# System Schemas
class SystemMetricsResponse(BaseModel):
    timestamp: datetime
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    network_io: Optional[Dict[str, Any]] = None
    active_cameras: Optional[int] = None
    total_fps: Optional[float] = None
    average_inference_time: Optional[float] = None
    queue_size: Optional[int] = None
    error_rate: Optional[float] = None
    alert_rate: Optional[float] = None
    
    class Config:
        from_attributes = True


class SystemStatsResponse(BaseModel):
    total_cameras: int
    active_cameras: int
    total_alerts_24h: int
    critical_alerts_24h: int
    system_uptime: float
    average_fps: float
    model_accuracy: float
    last_retrain: Optional[datetime] = None
    error_rate_24h: float = 0.0
    false_positive_rate: float = 0.0


class SystemConfigurationBase(BaseModel):
    config_key: str
    config_value: Dict[str, Any]
    config_type: str
    description: Optional[str] = None


class SystemConfigurationCreate(SystemConfigurationBase):
    updated_by: Optional[str] = None


class SystemConfigurationResponse(SystemConfigurationBase):
    id: int
    is_active: bool = True
    updated_at: datetime
    updated_by: Optional[str] = None
    
    class Config:
        from_attributes = True


# Real-time Schemas
class RealTimeAlert(BaseModel):
    type: str = "real_time_alert"
    alert: AlertResponse
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RealTimeUpdate(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CameraFrame(BaseModel):
    camera_id: str
    frame_data: str  # Base64 encoded image
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


# Inference Control Schemas
class InferenceConfig(BaseModel):
    model_path: Optional[str] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, gt=0)
    frame_skip: int = Field(default=1, ge=1)
    batch_size: int = Field(default=4, gt=0)
    gpu_enabled: bool = Field(default=True)


class InferenceStatus(BaseModel):
    running: bool
    cameras_active: int
    total_fps: float
    queue_size: int
    model_version: str
    uptime: float


# Batch Operations
class BatchCameraUpdate(BaseModel):
    camera_ids: List[str]
    updates: CameraUpdate


class BatchAlertUpdate(BaseModel):
    alert_ids: List[int]
    updates: AlertUpdate


# Analytics Schemas
class AnalyticsQuery(BaseModel):
    start_date: datetime
    end_date: datetime
    camera_ids: Optional[List[str]] = None
    anomaly_classes: Optional[List[str]] = None
    severity_levels: Optional[List[AlertSeverity]] = None
    group_by: Optional[str] = Field(None, description="hour, day, week, month")


class AnalyticsResponse(BaseModel):
    period: str
    camera_id: Optional[str] = None
    anomaly_class: Optional[str] = None
    severity: Optional[AlertSeverity] = None
    count: int
    avg_confidence: float
    false_positive_rate: Optional[float] = None


# Error Schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    field: str
    message: str
    value: Any


# Success Schemas
class SuccessResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)