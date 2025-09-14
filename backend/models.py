"""
Database Models for Multi-Camera Anomaly Detection System
=========================================================

SQLAlchemy models for cameras, alerts, feedback, and system data.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Camera(Base):
    """Camera configuration and status"""
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    camera_url = Column(String, nullable=False)
    status = Column(String, default="inactive")  # active, inactive, error, maintenance
    position_x = Column(Float, default=0.0)
    position_y = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime)
    last_frame_time = Column(DateTime)
    total_frames_processed = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    
    # Relationships
    alerts = relationship("Alert", back_populates="camera")
    performance_metrics = relationship("CameraPerformance", back_populates="camera")


class Alert(Base):
    """Alert/detection records"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), index=True)
    anomaly_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    severity = Column(String, nullable=False)  # low, medium, high, critical
    status = Column(String, default="active")  # active, acknowledged, resolved, false_positive
    bbox = Column(JSON)  # [x1, y1, x2, y2]
    objects_detected = Column(JSON)  # List of detected objects
    metadata = Column(JSON)  # Additional detection metadata
    frame_path = Column(String)  # Path to saved frame image
    video_clip_path = Column(String)  # Path to video clip if saved
    emergency_type = Column(String)  # police, medical, fire, security
    emergency_contact_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String)
    resolved_at = Column(DateTime)
    resolved_by = Column(String)
    
    # Relationships
    camera = relationship("Camera", back_populates="alerts")
    feedback = relationship("UserFeedback", back_populates="alert")


class UserFeedback(Base):
    """User feedback on detections for continuous learning"""
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), index=True)
    detection_id = Column(String, index=True)  # External detection ID
    user_id = Column(String)  # User who provided feedback
    is_correct = Column(Boolean, nullable=False)
    correct_label = Column(String)  # Correct label if detection was wrong
    confidence_rating = Column(Float, nullable=False)  # User confidence 0-1
    comments = Column(Text)
    feedback_type = Column(String, default="manual")  # manual, automated, expert
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)  # Whether used in retraining
    
    # Relationships
    alert = relationship("Alert", back_populates="feedback")


class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, nullable=False)
    model_type = Column(String, default="hybrid")  # hybrid, yolo, efficientnet
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    model_path = Column(String, nullable=False)
    config_path = Column(String)
    training_data_size = Column(Integer)
    training_duration = Column(Float)  # Hours
    training_epochs = Column(Integer)
    validation_loss = Column(Float)
    is_active = Column(Boolean, default=False)
    deployment_date = Column(DateTime)
    performance_metrics = Column(JSON)  # Additional metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)
    
    # Relationships
    deployments = relationship("ModelDeployment", back_populates="model_version")


class ModelDeployment(Base):
    """Model deployment history"""
    __tablename__ = "model_deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    deployment_type = Column(String)  # production, staging, testing
    status = Column(String)  # active, rolled_back, failed
    deployed_at = Column(DateTime, default=datetime.utcnow)
    rolled_back_at = Column(DateTime)
    performance_baseline = Column(JSON)
    rollback_reason = Column(Text)
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="deployments")


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_usage = Column(Float)  # Percentage
    memory_usage = Column(Float)  # Percentage
    gpu_usage = Column(Float)  # Percentage
    gpu_memory_usage = Column(Float)  # Percentage
    disk_usage = Column(Float)  # Percentage
    network_io = Column(JSON)  # Network I/O stats
    active_cameras = Column(Integer)
    total_fps = Column(Float)
    average_inference_time = Column(Float)  # Milliseconds
    queue_size = Column(Integer)
    error_rate = Column(Float)  # Errors per minute
    alert_rate = Column(Float)  # Alerts per minute


class CameraPerformance(Base):
    """Per-camera performance metrics"""
    __tablename__ = "camera_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, ForeignKey("cameras.camera_id"), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    fps = Column(Float)  # Actual FPS
    frame_drops = Column(Integer)  # Dropped frames in interval
    avg_inference_time = Column(Float)  # Average inference time (ms)
    detection_count = Column(Integer)  # Detections in interval
    alert_count = Column(Integer)  # Alerts generated in interval
    false_positive_rate = Column(Float)  # Based on feedback
    connection_errors = Column(Integer)  # Connection errors in interval
    quality_score = Column(Float)  # Overall quality score
    
    # Relationships
    camera = relationship("Camera", back_populates="performance_metrics")


class TrainingSession(Base):
    """Training session records"""
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, nullable=False)
    trigger_type = Column(String)  # scheduled, feedback_threshold, manual
    data_version = Column(String)
    training_config = Column(JSON)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String)  # running, completed, failed, cancelled
    progress = Column(Float, default=0.0)  # 0-100
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    best_accuracy = Column(Float)
    best_loss = Column(Float)
    final_metrics = Column(JSON)
    error_message = Column(Text)
    output_model_path = Column(String)
    logs_path = Column(String)


class EmergencyContact(Base):
    """Emergency contact information"""
    __tablename__ = "emergency_contacts"
    
    id = Column(Integer, primary_key=True, index=True)
    contact_type = Column(String, nullable=False)  # police, medical, fire, security
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    email = Column(String)
    department = Column(String)
    priority = Column(Integer, default=1)  # 1=primary, 2=secondary, etc.
    location_area = Column(String)  # Specific area they cover
    availability_hours = Column(JSON)  # When they're available
    response_time = Column(Float)  # Expected response time in minutes
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_contacted = Column(DateTime)


class SystemConfiguration(Base):
    """System configuration settings"""
    __tablename__ = "system_configuration"
    
    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String, unique=True, nullable=False)
    config_value = Column(JSON, nullable=False)
    config_type = Column(String)  # detection, alerts, training, system
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
    updated_by = Column(String)


class AuditLog(Base):
    """System audit log"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String, index=True)
    action = Column(String, nullable=False)  # login, logout, config_change, etc.
    resource_type = Column(String)  # camera, alert, model, etc.
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    status = Column(String)  # success, failure, warning