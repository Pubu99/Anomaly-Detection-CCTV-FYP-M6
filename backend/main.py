"""
FastAPI Backend for Multi-Camera Anomaly Detection System
=========================================================

Professional REST API with real-time WebSocket streaming, database management,
user feedback collection, and model retraining triggers.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

# Pydantic models for request/response
from pydantic import BaseModel, Field
from enum import Enum

# Database imports
import asyncpg
import motor.motor_asyncio
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Import our modules
from src.inference.real_time_inference import RealTimeInferenceEngine, CameraConfig
from src.inference.multi_camera_fusion import MultiCameraFusionSystem, FusedDetection
from src.models.hybrid_model import AnomalyResult
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger


# Pydantic Models
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


class CameraRequest(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier")
    camera_url: str = Field(..., description="Camera URL or device index")
    name: str = Field(..., description="Human-readable camera name")
    location: str = Field(..., description="Camera location description")
    position_x: float = Field(default=0.0, description="X coordinate position")
    position_y: float = Field(default=0.0, description="Y coordinate position")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Camera importance weight")
    resolution_width: int = Field(default=1920, description="Camera resolution width")
    resolution_height: int = Field(default=1080, description="Camera resolution height")
    fps: int = Field(default=30, description="Camera FPS")
    enabled: bool = Field(default=True, description="Whether camera is enabled")


class CameraResponse(BaseModel):
    id: int
    camera_id: str
    name: str
    location: str
    status: CameraStatus
    position_x: float
    position_y: float
    weight: float
    resolution_width: int
    resolution_height: int
    fps: int
    enabled: bool
    created_at: datetime
    last_seen: Optional[datetime]


class AlertRequest(BaseModel):
    camera_id: str
    anomaly_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: AlertSeverity
    bbox: Optional[List[float]] = None
    objects_detected: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    id: int
    camera_id: str
    anomaly_class: str
    confidence: float
    severity: AlertSeverity
    status: str
    bbox: Optional[List[float]]
    objects_detected: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]


class FeedbackRequest(BaseModel):
    detection_id: str = Field(..., description="Detection ID to provide feedback for")
    is_correct: bool = Field(..., description="Whether the detection was correct")
    correct_label: Optional[str] = Field(None, description="Correct label if detection was wrong")
    confidence_rating: float = Field(..., ge=0.0, le=1.0, description="User confidence in feedback")
    comments: Optional[str] = Field(None, description="Additional comments")


class SystemStatsResponse(BaseModel):
    total_cameras: int
    active_cameras: int
    total_alerts_24h: int
    critical_alerts_24h: int
    system_uptime: float
    average_fps: float
    model_accuracy: float
    last_retrain: Optional[datetime]


# Database Models
Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True)
    name = Column(String)
    location = Column(String)
    camera_url = Column(String)
    status = Column(String, default="inactive")
    position_x = Column(Float, default=0.0)
    position_y = Column(Float, default=0.0)
    weight = Column(Float, default=1.0)
    resolution_width = Column(Integer, default=1920)
    resolution_height = Column(Integer, default=1080)
    fps = Column(Integer, default=30)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime)


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    anomaly_class = Column(String)
    confidence = Column(Float)
    severity = Column(String)
    status = Column(String, default="active")
    bbox = Column(JSON)
    objects_detected = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)


class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String, index=True)
    is_correct = Column(Boolean)
    correct_label = Column(String)
    confidence_rating = Column(Float)
    comments = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True)
    accuracy = Column(Float)
    f1_score = Column(Float)
    model_path = Column(String)
    training_data_size = Column(Integer)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database setup
DATABASE_URL = "sqlite:///./anomaly_detection.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# FastAPI app initialization
app = FastAPI(
    title="Multi-Camera Anomaly Detection API",
    description="Professional API for real-time anomaly detection in surveillance systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = get_config()
logger = get_app_logger()
inference_engine: Optional[RealTimeInferenceEngine] = None
fusion_system: Optional[MultiCameraFusionSystem] = None
websocket_connections: List[WebSocket] = []

# Security
security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (simplified for demo)"""
    # In production, implement proper JWT verification
    token = credentials.credentials
    if token != "demo-token":  # Replace with proper validation
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return token


# WebSocket Manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


websocket_manager = WebSocketManager()


# API Routes

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Anomaly Detection API...")
    
    global inference_engine, fusion_system
    
    try:
        # Initialize camera positions (can be loaded from database)
        camera_positions = {
            "camera_1": (0.0, 0.0),
            "camera_2": (10.0, 0.0),
            "camera_3": (5.0, 10.0)
        }
        
        # Initialize fusion system
        fusion_system = MultiCameraFusionSystem(camera_positions, config.config)
        
        logger.info("API services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Anomaly Detection API...")
    
    global inference_engine
    if inference_engine:
        inference_engine.stop_inference()


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


# Camera Management
@app.post("/api/cameras", response_model=CameraResponse)
async def create_camera(camera: CameraRequest, db: Session = Depends(get_db)):
    """Create a new camera"""
    # Check if camera already exists
    existing = db.query(Camera).filter(Camera.camera_id == camera.camera_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Camera already exists")
    
    db_camera = Camera(
        camera_id=camera.camera_id,
        name=camera.name,
        location=camera.location,
        camera_url=camera.camera_url,
        position_x=camera.position_x,
        position_y=camera.position_y,
        weight=camera.weight,
        resolution_width=camera.resolution_width,
        resolution_height=camera.resolution_height,
        fps=camera.fps,
        enabled=camera.enabled
    )
    
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    
    return CameraResponse(**db_camera.__dict__)


@app.get("/api/cameras", response_model=List[CameraResponse])
async def get_cameras(db: Session = Depends(get_db)):
    """Get all cameras"""
    cameras = db.query(Camera).all()
    return [CameraResponse(**camera.__dict__) for camera in cameras]


@app.get("/api/cameras/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """Get specific camera"""
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return CameraResponse(**camera.__dict__)


@app.put("/api/cameras/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: str, camera_update: CameraRequest, db: Session = Depends(get_db)):
    """Update camera configuration"""
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    for field, value in camera_update.dict().items():
        setattr(camera, field, value)
    
    db.commit()
    db.refresh(camera)
    
    return CameraResponse(**camera.__dict__)


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """Delete camera"""
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    db.delete(camera)
    db.commit()
    
    return {"message": "Camera deleted successfully"}


# Alert Management
@app.post("/api/alerts", response_model=AlertResponse)
async def create_alert(alert: AlertRequest, db: Session = Depends(get_db)):
    """Create a new alert"""
    db_alert = Alert(
        camera_id=alert.camera_id,
        anomaly_class=alert.anomaly_class,
        confidence=alert.confidence,
        severity=alert.severity.value,
        bbox=alert.bbox,
        objects_detected=alert.objects_detected,
        metadata=alert.metadata
    )
    
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    
    # Broadcast alert via WebSocket
    alert_message = {
        "type": "new_alert",
        "alert": AlertResponse(**db_alert.__dict__).dict()
    }
    await websocket_manager.broadcast_message(json.dumps(alert_message))
    
    return AlertResponse(**db_alert.__dict__)


@app.get("/api/alerts", response_model=List[AlertResponse])
async def get_alerts(
    camera_id: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get alerts with optional filtering"""
    query = db.query(Alert)
    
    if camera_id:
        query = query.filter(Alert.camera_id == camera_id)
    if severity:
        query = query.filter(Alert.severity == severity.value)
    if status:
        query = query.filter(Alert.status == status)
    
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    return [AlertResponse(**alert.__dict__) for alert in alerts]


@app.patch("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged_at = datetime.utcnow()
    alert.status = "acknowledged"
    db.commit()
    
    return {"message": "Alert acknowledged"}


@app.patch("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    """Resolve an alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.resolved_at = datetime.utcnow()
    alert.status = "resolved"
    db.commit()
    
    return {"message": "Alert resolved"}


# User Feedback
@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit user feedback for a detection"""
    db_feedback = UserFeedback(
        detection_id=feedback.detection_id,
        is_correct=feedback.is_correct,
        correct_label=feedback.correct_label,
        confidence_rating=feedback.confidence_rating,
        comments=feedback.comments
    )
    
    db.add(db_feedback)
    db.commit()
    
    logger.info(f"Feedback received for detection {feedback.detection_id}")
    
    # Check if we have enough feedback to trigger retraining
    feedback_count = db.query(UserFeedback).count()
    retrain_threshold = config.get('continuous_learning.feedback_threshold', 100)
    
    if feedback_count >= retrain_threshold:
        # Trigger retraining (implement as background task)
        logger.info("Feedback threshold reached, scheduling model retraining")
    
    return {"message": "Feedback submitted successfully"}


@app.get("/api/feedback")
async def get_feedback(limit: int = 100, db: Session = Depends(get_db)):
    """Get user feedback"""
    feedback = db.query(UserFeedback).order_by(UserFeedback.created_at.desc()).limit(limit).all()
    return feedback


# System Statistics
@app.get("/api/stats", response_model=SystemStatsResponse)
async def get_system_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    # Calculate statistics
    total_cameras = db.query(Camera).count()
    active_cameras = db.query(Camera).filter(Camera.status == "active").count()
    
    # Alerts in last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)
    total_alerts_24h = db.query(Alert).filter(Alert.created_at >= yesterday).count()
    critical_alerts_24h = db.query(Alert).filter(
        Alert.created_at >= yesterday,
        Alert.severity == "critical"
    ).count()
    
    # Get latest model version
    latest_model = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
    
    return SystemStatsResponse(
        total_cameras=total_cameras,
        active_cameras=active_cameras,
        total_alerts_24h=total_alerts_24h,
        critical_alerts_24h=critical_alerts_24h,
        system_uptime=time.time(),  # Simplified
        average_fps=25.0,  # Would calculate from actual data
        model_accuracy=latest_model.accuracy if latest_model else 0.0,
        last_retrain=latest_model.created_at if latest_model else None
    )


# Real-time WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Echo back or handle specific commands
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# Inference Control
@app.post("/api/inference/start")
async def start_inference():
    """Start real-time inference"""
    global inference_engine
    
    if inference_engine and inference_engine.running:
        raise HTTPException(status_code=400, detail="Inference already running")
    
    # Get cameras from database
    db = SessionLocal()
    cameras = db.query(Camera).filter(Camera.enabled == True).all()
    db.close()
    
    if not cameras:
        raise HTTPException(status_code=400, detail="No enabled cameras found")
    
    # Convert to CameraConfig objects
    camera_configs = [
        CameraConfig(
            camera_id=cam.camera_id,
            camera_url=cam.camera_url,
            resolution=(cam.resolution_width, cam.resolution_height),
            fps=cam.fps,
            position=(cam.position_x, cam.position_y),
            weight=cam.weight
        )
        for cam in cameras
    ]
    
    try:
        inference_engine = RealTimeInferenceEngine(
            model_path="models/checkpoints/best_model.pth",
            cameras=camera_configs
        )
        
        # Register alert callback
        inference_engine.register_alert_callback(handle_inference_alert)
        
        # Start inference in background
        asyncio.create_task(inference_engine.start_async_inference())
        
        return {"message": "Inference started successfully"}
        
    except Exception as e:
        logger.error(f"Error starting inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inference/stop")
async def stop_inference():
    """Stop real-time inference"""
    global inference_engine
    
    if not inference_engine or not inference_engine.running:
        raise HTTPException(status_code=400, detail="Inference not running")
    
    inference_engine.stop_inference()
    return {"message": "Inference stopped successfully"}


async def handle_inference_alert(anomaly: AnomalyResult):
    """Handle alerts from inference engine"""
    # Create alert in database
    db = SessionLocal()
    
    try:
        db_alert = Alert(
            camera_id=anomaly.camera_id,
            anomaly_class=anomaly.anomaly_class,
            confidence=anomaly.confidence,
            severity=anomaly.severity,
            metadata={"timestamp": anomaly.timestamp}
        )
        
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        
        # Broadcast via WebSocket
        alert_message = {
            "type": "real_time_alert",
            "alert": AlertResponse(**db_alert.__dict__).dict()
        }
        await websocket_manager.broadcast_message(json.dumps(alert_message))
        
    except Exception as e:
        logger.error(f"Error handling inference alert: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)