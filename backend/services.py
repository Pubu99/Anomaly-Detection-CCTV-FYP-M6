"""
Business Logic Services for Multi-Camera Anomaly Detection System
=================================================================

Service layer containing business logic for cameras, alerts, and inference management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

# Import models and schemas
from .models import (
    Camera, Alert, UserFeedback, ModelVersion, SystemMetrics,
    CameraPerformance, TrainingSession, EmergencyContact
)
from .schemas import (
    CameraCreate, CameraUpdate, AlertCreate, FeedbackCreate,
    AlertSeverity, CameraStatus, AlertStatus
)

logger = logging.getLogger(__name__)


class CameraService:
    """Service for camera management operations"""
    
    @staticmethod
    def create_camera(db: Session, camera_data: CameraCreate) -> Camera:
        """Create a new camera"""
        # Check if camera already exists
        existing = db.query(Camera).filter(Camera.camera_id == camera_data.camera_id).first()
        if existing:
            raise ValueError(f"Camera with ID {camera_data.camera_id} already exists")
        
        db_camera = Camera(**camera_data.dict())
        db.add(db_camera)
        db.commit()
        db.refresh(db_camera)
        
        logger.info(f"Created camera: {camera_data.camera_id}")
        return db_camera
    
    @staticmethod
    def get_camera(db: Session, camera_id: str) -> Optional[Camera]:
        """Get camera by ID"""
        return db.query(Camera).filter(Camera.camera_id == camera_id).first()
    
    @staticmethod
    def get_cameras(db: Session, skip: int = 0, limit: int = 100) -> List[Camera]:
        """Get all cameras with pagination"""
        return db.query(Camera).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_active_cameras(db: Session) -> List[Camera]:
        """Get all active cameras"""
        return db.query(Camera).filter(
            and_(Camera.enabled == True, Camera.status == CameraStatus.ACTIVE)
        ).all()
    
    @staticmethod
    def update_camera(db: Session, camera_id: str, camera_data: CameraUpdate) -> Optional[Camera]:
        """Update camera configuration"""
        camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        if not camera:
            return None
        
        update_data = camera_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(camera, field, value)
        
        db.commit()
        db.refresh(camera)
        
        logger.info(f"Updated camera: {camera_id}")
        return camera
    
    @staticmethod
    def update_camera_status(db: Session, camera_id: str, status: CameraStatus) -> Optional[Camera]:
        """Update camera status"""
        camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        if camera:
            camera.status = status.value
            camera.last_seen = datetime.utcnow()
            db.commit()
            db.refresh(camera)
        return camera
    
    @staticmethod
    def delete_camera(db: Session, camera_id: str) -> bool:
        """Delete camera"""
        camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        if camera:
            db.delete(camera)
            db.commit()
            logger.info(f"Deleted camera: {camera_id}")
            return True
        return False
    
    @staticmethod
    def increment_frame_count(db: Session, camera_id: str) -> None:
        """Increment frame count for camera"""
        camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        if camera:
            camera.total_frames_processed += 1
            camera.last_frame_time = datetime.utcnow()
            db.commit()
    
    @staticmethod
    def increment_error_count(db: Session, camera_id: str) -> None:
        """Increment error count for camera"""
        camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        if camera:
            camera.error_count += 1
            db.commit()


class AlertService:
    """Service for alert management operations"""
    
    @staticmethod
    def create_alert(db: Session, alert_data: AlertCreate, emergency_type: Optional[str] = None) -> Alert:
        """Create a new alert"""
        db_alert = Alert(
            **alert_data.dict(),
            emergency_type=emergency_type
        )
        
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        
        logger.info(f"Created alert: {db_alert.id} for camera {alert_data.camera_id}")
        
        # Trigger emergency notifications if critical
        if alert_data.severity == AlertSeverity.CRITICAL:
            EmergencyService.handle_critical_alert(db, db_alert)
        
        return db_alert
    
    @staticmethod
    def get_alert(db: Session, alert_id: int) -> Optional[Alert]:
        """Get alert by ID"""
        return db.query(Alert).filter(Alert.id == alert_id).first()
    
    @staticmethod
    def get_alerts(
        db: Session,
        camera_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts with filtering"""
        query = db.query(Alert)
        
        if camera_id:
            query = query.filter(Alert.camera_id == camera_id)
        if severity:
            query = query.filter(Alert.severity == severity.value)
        if status:
            query = query.filter(Alert.status == status.value)
        if start_date:
            query = query.filter(Alert.created_at >= start_date)
        if end_date:
            query = query.filter(Alert.created_at <= end_date)
        
        return query.order_by(desc(Alert.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_recent_alerts(db: Session, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return db.query(Alert).filter(Alert.created_at >= cutoff).order_by(desc(Alert.created_at)).all()
    
    @staticmethod
    def acknowledge_alert(db: Session, alert_id: int, acknowledged_by: str) -> Optional[Alert]:
        """Acknowledge an alert"""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.status = AlertStatus.ACKNOWLEDGED.value
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            db.commit()
            db.refresh(alert)
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return alert
    
    @staticmethod
    def resolve_alert(db: Session, alert_id: int, resolved_by: str) -> Optional[Alert]:
        """Resolve an alert"""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.status = AlertStatus.RESOLVED.value
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            db.commit()
            db.refresh(alert)
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return alert
    
    @staticmethod
    def mark_false_positive(db: Session, alert_id: int, marked_by: str) -> Optional[Alert]:
        """Mark alert as false positive"""
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if alert:
            alert.status = AlertStatus.FALSE_POSITIVE.value
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = marked_by
            db.commit()
            db.refresh(alert)
            logger.info(f"Alert {alert_id} marked as false positive by {marked_by}")
        return alert
    
    @staticmethod
    def get_alert_statistics(db: Session, days: int = 30) -> Dict[str, Any]:
        """Get alert statistics for the last N days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        total_alerts = db.query(Alert).filter(Alert.created_at >= cutoff).count()
        critical_alerts = db.query(Alert).filter(
            and_(Alert.created_at >= cutoff, Alert.severity == AlertSeverity.CRITICAL.value)
        ).count()
        resolved_alerts = db.query(Alert).filter(
            and_(Alert.created_at >= cutoff, Alert.status == AlertStatus.RESOLVED.value)
        ).count()
        false_positives = db.query(Alert).filter(
            and_(Alert.created_at >= cutoff, Alert.status == AlertStatus.FALSE_POSITIVE.value)
        ).count()
        
        return {
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "resolved_alerts": resolved_alerts,
            "false_positives": false_positives,
            "resolution_rate": resolved_alerts / total_alerts if total_alerts > 0 else 0,
            "false_positive_rate": false_positives / total_alerts if total_alerts > 0 else 0
        }


class FeedbackService:
    """Service for user feedback management"""
    
    @staticmethod
    def create_feedback(db: Session, feedback_data: FeedbackCreate) -> UserFeedback:
        """Create user feedback"""
        db_feedback = UserFeedback(**feedback_data.dict())
        db.add(db_feedback)
        db.commit()
        db.refresh(db_feedback)
        
        logger.info(f"Feedback created for detection: {feedback_data.detection_id}")
        
        # Check if we should trigger retraining
        FeedbackService.check_retraining_threshold(db)
        
        return db_feedback
    
    @staticmethod
    def get_feedback(db: Session, skip: int = 0, limit: int = 100) -> List[UserFeedback]:
        """Get feedback with pagination"""
        return db.query(UserFeedback).order_by(desc(UserFeedback.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_unprocessed_feedback(db: Session) -> List[UserFeedback]:
        """Get feedback that hasn't been used for training"""
        return db.query(UserFeedback).filter(UserFeedback.processed == False).all()
    
    @staticmethod
    def mark_feedback_processed(db: Session, feedback_ids: List[int]) -> None:
        """Mark feedback as processed"""
        db.query(UserFeedback).filter(UserFeedback.id.in_(feedback_ids)).update(
            {UserFeedback.processed: True}, synchronize_session=False
        )
        db.commit()
        logger.info(f"Marked {len(feedback_ids)} feedback items as processed")
    
    @staticmethod
    def check_retraining_threshold(db: Session, threshold: int = 100) -> bool:
        """Check if we have enough feedback to trigger retraining"""
        unprocessed_count = db.query(UserFeedback).filter(UserFeedback.processed == False).count()
        
        if unprocessed_count >= threshold:
            logger.info(f"Retraining threshold reached: {unprocessed_count} unprocessed feedback items")
            # Trigger retraining (implement in TrainingService)
            return True
        return False
    
    @staticmethod
    def get_feedback_statistics(db: Session) -> Dict[str, Any]:
        """Get feedback statistics"""
        total_feedback = db.query(UserFeedback).count()
        correct_feedback = db.query(UserFeedback).filter(UserFeedback.is_correct == True).count()
        processed_feedback = db.query(UserFeedback).filter(UserFeedback.processed == True).count()
        
        return {
            "total_feedback": total_feedback,
            "correct_feedback": correct_feedback,
            "incorrect_feedback": total_feedback - correct_feedback,
            "accuracy_rate": correct_feedback / total_feedback if total_feedback > 0 else 0,
            "processed_feedback": processed_feedback,
            "pending_feedback": total_feedback - processed_feedback
        }


class EmergencyService:
    """Service for emergency contact and notification management"""
    
    @staticmethod
    def get_emergency_contacts(db: Session, contact_type: Optional[str] = None) -> List[EmergencyContact]:
        """Get emergency contacts by type"""
        query = db.query(EmergencyContact).filter(EmergencyContact.is_active == True)
        if contact_type:
            query = query.filter(EmergencyContact.contact_type == contact_type)
        return query.order_by(EmergencyContact.priority).all()
    
    @staticmethod
    def handle_critical_alert(db: Session, alert: Alert) -> None:
        """Handle critical alert by notifying emergency contacts"""
        if not alert.emergency_type:
            # Determine emergency type based on anomaly class
            alert.emergency_type = EmergencyService.determine_emergency_type(alert.anomaly_class)
        
        contacts = EmergencyService.get_emergency_contacts(db, alert.emergency_type)
        
        for contact in contacts[:2]:  # Notify top 2 contacts
            try:
                EmergencyService.send_emergency_notification(contact, alert)
                contact.last_contacted = datetime.utcnow()
                alert.emergency_contact_sent = True
            except Exception as e:
                logger.error(f"Failed to notify emergency contact {contact.id}: {e}")
        
        db.commit()
    
    @staticmethod
    def determine_emergency_type(anomaly_class: str) -> str:
        """Determine emergency type based on anomaly class"""
        emergency_mapping = {
            "shooting": "police",
            "fighting": "police",
            "assault": "police",
            "robbery": "police",
            "burglary": "police",
            "vandalism": "police",
            "explosion": "fire",
            "arson": "fire",
            "accident": "medical",
            "shoplifting": "security",
            "stealing": "security"
        }
        return emergency_mapping.get(anomaly_class.lower(), "security")
    
    @staticmethod
    def send_emergency_notification(contact: EmergencyContact, alert: Alert) -> None:
        """Send emergency notification (implement with actual notification service)"""
        # This would integrate with actual notification services
        message = f"CRITICAL ALERT: {alert.anomaly_class} detected at camera {alert.camera_id} with {alert.confidence:.2f} confidence"
        
        logger.info(f"Emergency notification sent to {contact.name}: {message}")
        # TODO: Implement actual SMS/email/call functionality


class PerformanceService:
    """Service for system performance monitoring"""
    
    @staticmethod
    def record_system_metrics(db: Session, metrics: Dict[str, Any]) -> SystemMetrics:
        """Record system performance metrics"""
        db_metrics = SystemMetrics(**metrics)
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)
        return db_metrics
    
    @staticmethod
    def record_camera_performance(db: Session, camera_id: str, performance: Dict[str, Any]) -> CameraPerformance:
        """Record camera performance metrics"""
        db_performance = CameraPerformance(camera_id=camera_id, **performance)
        db.add(db_performance)
        db.commit()
        db.refresh(db_performance)
        return db_performance
    
    @staticmethod
    def get_system_performance(db: Session, hours: int = 24) -> List[SystemMetrics]:
        """Get system performance metrics for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return db.query(SystemMetrics).filter(SystemMetrics.timestamp >= cutoff).order_by(SystemMetrics.timestamp).all()
    
    @staticmethod
    def get_camera_performance(db: Session, camera_id: str, hours: int = 24) -> List[CameraPerformance]:
        """Get camera performance metrics for the last N hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return db.query(CameraPerformance).filter(
            and_(CameraPerformance.camera_id == camera_id, CameraPerformance.timestamp >= cutoff)
        ).order_by(CameraPerformance.timestamp).all()
    
    @staticmethod
    def calculate_system_health(db: Session) -> Dict[str, Any]:
        """Calculate overall system health score"""
        # Get recent metrics
        recent_metrics = PerformanceService.get_system_performance(db, hours=1)
        
        if not recent_metrics:
            return {"health_score": 0, "status": "unknown"}
        
        latest = recent_metrics[-1]
        
        # Calculate health score based on various factors
        cpu_score = max(0, 100 - (latest.cpu_usage or 0))
        memory_score = max(0, 100 - (latest.memory_usage or 0))
        gpu_score = max(0, 100 - (latest.gpu_usage or 0))
        error_score = max(0, 100 - (latest.error_rate or 0) * 10)
        
        health_score = (cpu_score + memory_score + gpu_score + error_score) / 4
        
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "health_score": health_score,
            "status": status,
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "gpu_usage": latest.gpu_usage,
            "error_rate": latest.error_rate,
            "active_cameras": latest.active_cameras,
            "total_fps": latest.total_fps
        }


class ModelService:
    """Service for model version management"""
    
    @staticmethod
    def create_model_version(db: Session, model_data: Dict[str, Any]) -> ModelVersion:
        """Create new model version"""
        db_model = ModelVersion(**model_data)
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        
        logger.info(f"Created model version: {model_data['version']}")
        return db_model
    
    @staticmethod
    def get_active_model(db: Session) -> Optional[ModelVersion]:
        """Get currently active model version"""
        return db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    @staticmethod
    def set_active_model(db: Session, version: str) -> Optional[ModelVersion]:
        """Set model version as active"""
        # Deactivate current active model
        db.query(ModelVersion).filter(ModelVersion.is_active == True).update(
            {ModelVersion.is_active: False}
        )
        
        # Activate new model
        model = db.query(ModelVersion).filter(ModelVersion.version == version).first()
        if model:
            model.is_active = True
            model.deployment_date = datetime.utcnow()
            db.commit()
            db.refresh(model)
            logger.info(f"Activated model version: {version}")
        
        return model
    
    @staticmethod
    def get_model_versions(db: Session) -> List[ModelVersion]:
        """Get all model versions"""
        return db.query(ModelVersion).order_by(desc(ModelVersion.created_at)).all()


# Service factory
class ServiceFactory:
    """Factory for creating service instances"""
    
    @staticmethod
    def get_camera_service() -> CameraService:
        return CameraService()
    
    @staticmethod
    def get_alert_service() -> AlertService:
        return AlertService()
    
    @staticmethod
    def get_feedback_service() -> FeedbackService:
        return FeedbackService()
    
    @staticmethod
    def get_emergency_service() -> EmergencyService:
        return EmergencyService()
    
    @staticmethod
    def get_performance_service() -> PerformanceService:
        return PerformanceService()
    
    @staticmethod
    def get_model_service() -> ModelService:
        return ModelService()