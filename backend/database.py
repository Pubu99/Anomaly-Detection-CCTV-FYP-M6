"""
Database Configuration for Multi-Camera Anomaly Detection System
================================================================

Database connection, session management, and initialization utilities.
"""

import os
from typing import Generator, Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging

# Import models
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration class"""
    
    def __init__(self):
        self.database_url = self._get_database_url()
        self.engine = None
        self.session_local = None
        self._initialize_database()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use default"""
        # Production: Use PostgreSQL
        postgres_url = os.getenv("DATABASE_URL")
        if postgres_url:
            return postgres_url
        
        # Development: Use SQLite
        db_path = os.getenv("DB_PATH", "./data/anomaly_detection.db")
        return f"sqlite:///{db_path}"
    
    def _initialize_database(self):
        """Initialize database engine and session"""
        if "sqlite" in self.database_url:
            # SQLite configuration
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=os.getenv("DB_ECHO", "false").lower() == "true"
            )
        else:
            # PostgreSQL configuration
            self.engine = create_engine(
                self.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=os.getenv("DB_ECHO", "false").lower() == "true"
            )
        
        self.session_local = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping database tables: {e}")
            raise
    
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session"""
        session = self.session_local()
        try:
            yield session
        finally:
            session.close()
    
    def get_session_sync(self) -> Session:
        """Get synchronous database session"""
        return self.session_local()


# Global database instance
db_config = DatabaseConfig()


def get_database() -> DatabaseConfig:
    """Get database configuration instance"""
    return db_config


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    yield from db_config.get_session()


def init_db():
    """Initialize database with tables"""
    db_config.create_tables()


def reset_db():
    """Reset database (drop and recreate tables)"""
    db_config.drop_tables()
    db_config.create_tables()


# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.db_config.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_table_counts(self) -> dict:
        """Get record counts for all tables"""
        counts = {}
        try:
            with self.db_config.get_session() as session:
                # Import here to avoid circular imports
                from .models import (
                    Camera, Alert, UserFeedback, ModelVersion, 
                    SystemMetrics, CameraPerformance, TrainingSession,
                    EmergencyContact, SystemConfiguration, AuditLog
                )
                
                counts = {
                    "cameras": session.query(Camera).count(),
                    "alerts": session.query(Alert).count(),
                    "user_feedback": session.query(UserFeedback).count(),
                    "model_versions": session.query(ModelVersion).count(),
                    "system_metrics": session.query(SystemMetrics).count(),
                    "camera_performance": session.query(CameraPerformance).count(),
                    "training_sessions": session.query(TrainingSession).count(),
                    "emergency_contacts": session.query(EmergencyContact).count(),
                    "system_configuration": session.query(SystemConfiguration).count(),
                    "audit_logs": session.query(AuditLog).count()
                }
        except Exception as e:
            logger.error(f"Error getting table counts: {e}")
        
        return counts
    
    def backup_database(self, backup_path: str) -> bool:
        """Backup database (SQLite only)"""
        if "sqlite" not in self.db_config.database_url:
            logger.warning("Database backup only supported for SQLite")
            return False
        
        try:
            import shutil
            db_path = self.db_config.database_url.replace("sqlite:///", "")
            shutil.copy2(db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup (SQLite only)"""
        if "sqlite" not in self.db_config.database_url:
            logger.warning("Database restore only supported for SQLite")
            return False
        
        try:
            import shutil
            db_path = self.db_config.database_url.replace("sqlite:///", "")
            shutil.copy2(backup_path, db_path)
            logger.info(f"Database restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> dict:
        """Clean up old records to manage database size"""
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        try:
            with self.db_config.get_session() as session:
                # Import models
                from .models import SystemMetrics, CameraPerformance, AuditLog
                
                # Delete old system metrics
                deleted_metrics = session.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < cutoff_date
                ).delete()
                
                # Delete old camera performance records
                deleted_performance = session.query(CameraPerformance).filter(
                    CameraPerformance.timestamp < cutoff_date
                ).delete()
                
                # Delete old audit logs (keep for longer - 90 days)
                audit_cutoff = datetime.utcnow() - timedelta(days=90)
                deleted_audit = session.query(AuditLog).filter(
                    AuditLog.timestamp < audit_cutoff
                ).delete()
                
                session.commit()
                
                deleted_counts = {
                    "system_metrics": deleted_metrics,
                    "camera_performance": deleted_performance,
                    "audit_logs": deleted_audit
                }
                
                logger.info(f"Database cleanup completed: {deleted_counts}")
                
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
        
        return deleted_counts


# Migration utilities
def run_migrations():
    """Run database migrations"""
    logger.info("Running database migrations...")
    
    # Check current database version
    try:
        with db_config.get_session() as session:
            # Add migration logic here as needed
            # For now, just ensure all tables exist
            init_db()
            logger.info("Database migrations completed")
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise


# Initialize default data
def init_default_data():
    """Initialize database with default data"""
    logger.info("Initializing default data...")
    
    try:
        with db_config.get_session() as session:
            # Import models
            from .models import EmergencyContact, SystemConfiguration
            
            # Check if emergency contacts exist
            if session.query(EmergencyContact).count() == 0:
                # Add default emergency contacts
                default_contacts = [
                    EmergencyContact(
                        contact_type="police",
                        name="Local Police Department",
                        phone="911",
                        priority=1,
                        response_time=5.0
                    ),
                    EmergencyContact(
                        contact_type="medical",
                        name="Emergency Medical Services",
                        phone="911",
                        priority=1,
                        response_time=8.0
                    ),
                    EmergencyContact(
                        contact_type="fire",
                        name="Fire Department",
                        phone="911",
                        priority=1,
                        response_time=6.0
                    ),
                    EmergencyContact(
                        contact_type="security",
                        name="Security Team",
                        phone="555-0123",
                        priority=1,
                        response_time=3.0
                    )
                ]
                
                for contact in default_contacts:
                    session.add(contact)
            
            # Check if system configuration exists
            if session.query(SystemConfiguration).count() == 0:
                # Add default system configurations
                default_configs = [
                    SystemConfiguration(
                        config_key="detection.confidence_threshold",
                        config_value={"value": 0.5},
                        config_type="detection",
                        description="Minimum confidence threshold for detections"
                    ),
                    SystemConfiguration(
                        config_key="alerts.cooldown_minutes",
                        config_value={"value": 5},
                        config_type="alerts",
                        description="Cooldown period between similar alerts"
                    ),
                    SystemConfiguration(
                        config_key="training.batch_size",
                        config_value={"value": 32},
                        config_type="training",
                        description="Default batch size for training"
                    ),
                    SystemConfiguration(
                        config_key="system.max_cameras",
                        config_value={"value": 50},
                        config_type="system",
                        description="Maximum number of cameras supported"
                    )
                ]
                
                for config in default_configs:
                    session.add(config)
            
            session.commit()
            logger.info("Default data initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing default data: {e}")
        raise


if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    init_default_data()
    
    # Show database info
    db_manager = DatabaseManager(db_config)
    print("Database Health:", db_manager.health_check())
    print("Table Counts:", db_manager.get_table_counts())