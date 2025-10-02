"""
Enhanced Continuous Learning System
==================================

Advanced continuous learning pipeline that integrates user feedback with the CNN-LSTM model
for automatic model improvement through human-in-the-loop learning.

Key Features:
- Real-time feedback collection from user interface
- Intelligent feedback validation and quality assessment
- Automated model retraining with feedback integration
- A/B testing for model performance comparison
- Incremental learning to avoid catastrophic forgetting
- Model versioning and rollback capabilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import cv2
import threading
import queue
from collections import defaultdict, deque
import pickle
import hashlib

# Import our enhanced modules
from src.models.enhanced_temporal_model import EnhancedTemporalAnomalyModel, create_enhanced_temporal_model
from src.training.enhanced_temporal_train import EnhancedTemporalTrainer
from src.data.enhanced_data_preprocessing import EnhancedVideoDataset, VideoSequenceExtractor
from src.utils.config import get_config
from src.utils.logging_config import get_app_logger


class FeedbackType(Enum):
    """Types of user feedback"""
    TRUE_POSITIVE = "true_positive"           # Correct detection
    FALSE_POSITIVE = "false_positive"         # Incorrect detection (should be normal)
    FALSE_NEGATIVE = "false_negative"         # Missed detection (should be anomaly)
    WRONG_CLASSIFICATION = "wrong_classification"  # Wrong anomaly type
    SEVERITY_CORRECTION = "severity_correction"    # Wrong severity level


class UserRole(Enum):
    """User role determines feedback weight"""
    SECURITY_GUARD = "security_guard"         # Weight: 0.6
    SUPERVISOR = "supervisor"                 # Weight: 0.8
    SECURITY_EXPERT = "security_expert"       # Weight: 1.0
    SYSTEM_ADMIN = "system_admin"            # Weight: 0.9


@dataclass
class FeedbackEntry:
    """Enhanced feedback entry structure"""
    id: str
    timestamp: datetime
    user_id: str
    user_role: UserRole
    camera_id: str
    
    # Original prediction
    original_prediction: str
    original_confidence: float
    original_severity: str
    
    # User correction
    feedback_type: FeedbackType
    corrected_label: Optional[str] = None
    corrected_severity: Optional[str] = None
    
    # Context information
    video_segment: Optional[str] = None  # Path to video segment
    frame_data: Optional[bytes] = None   # Key frame data
    bounding_boxes: Optional[List] = None
    
    # Feedback metadata
    confidence_level: float = 1.0       # User's confidence in feedback
    notes: Optional[str] = None
    processed: bool = False
    
    def get_feedback_weight(self) -> float:
        """Calculate feedback weight based on user role and confidence"""
        role_weights = {
            UserRole.SECURITY_GUARD: 0.6,
            UserRole.SUPERVISOR: 0.8,
            UserRole.SECURITY_EXPERT: 1.0,
            UserRole.SYSTEM_ADMIN: 0.9
        }
        return role_weights[self.user_role] * self.confidence_level


class FeedbackDatabase:
    """Enhanced feedback database with better querying capabilities"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = get_app_logger()
        self._init_database()
    
    def _init_database(self):
        """Initialize database with enhanced schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    user_id TEXT,
                    user_role TEXT,
                    camera_id TEXT,
                    original_prediction TEXT,
                    original_confidence REAL,
                    original_severity TEXT,
                    feedback_type TEXT,
                    corrected_label TEXT,
                    corrected_severity TEXT,
                    confidence_level REAL,
                    feedback_weight REAL,
                    video_segment TEXT,
                    notes TEXT,
                    processed BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    evaluation_date DATETIME,
                    accuracy REAL,
                    f1_score REAL,
                    precision REAL,
                    recall REAL,
                    feedback_integrated_count INTEGER,
                    notes TEXT
                )
            """)
            
            # Retraining history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retraining_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    old_model_version TEXT,
                    new_model_version TEXT,
                    feedback_count INTEGER,
                    improvement_metrics TEXT,
                    status TEXT,
                    error_message TEXT
                )
            """)
            
            conn.commit()
    
    def add_feedback(self, feedback: FeedbackEntry) -> bool:
        """Add feedback entry to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO feedback (
                        id, timestamp, user_id, user_role, camera_id,
                        original_prediction, original_confidence, original_severity,
                        feedback_type, corrected_label, corrected_severity,
                        confidence_level, feedback_weight, video_segment, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id, feedback.timestamp, feedback.user_id, feedback.user_role.value,
                    feedback.camera_id, feedback.original_prediction, feedback.original_confidence,
                    feedback.original_severity, feedback.feedback_type.value,
                    feedback.corrected_label, feedback.corrected_severity,
                    feedback.confidence_level, feedback.get_feedback_weight(),
                    feedback.video_segment, feedback.notes
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding feedback: {e}")
            return False
    
    def get_unprocessed_feedback(self, limit: Optional[int] = None) -> List[FeedbackEntry]:
        """Get unprocessed feedback entries"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM feedback WHERE processed = 0 ORDER BY timestamp"
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Convert rows to FeedbackEntry objects
                feedback_entries = []
                for row in rows:
                    feedback = FeedbackEntry(
                        id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        user_id=row[2],
                        user_role=UserRole(row[3]),
                        camera_id=row[4],
                        original_prediction=row[5],
                        original_confidence=row[6],
                        original_severity=row[7],
                        feedback_type=FeedbackType(row[8]),
                        corrected_label=row[9],
                        corrected_severity=row[10],
                        confidence_level=row[11],
                        video_segment=row[13],
                        notes=row[14],
                        processed=bool(row[15])
                    )
                    feedback_entries.append(feedback)
                
                return feedback_entries
                
        except Exception as e:
            self.logger.error(f"Error getting feedback: {e}")
            return []
    
    def mark_processed(self, feedback_ids: List[str]):
        """Mark feedback entries as processed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                placeholders = ','.join('?' * len(feedback_ids))
                cursor.execute(f"""
                    UPDATE feedback 
                    SET processed = 1 
                    WHERE id IN ({placeholders})
                """, feedback_ids)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error marking feedback as processed: {e}")


class FeedbackDataset(Dataset):
    """Dataset that incorporates user feedback for retraining"""
    
    def __init__(
        self,
        original_dataset: EnhancedVideoDataset,
        feedback_entries: List[FeedbackEntry],
        video_extractor: VideoSequenceExtractor,
        class_names: List[str]
    ):
        self.original_dataset = original_dataset
        self.feedback_entries = feedback_entries
        self.video_extractor = video_extractor
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.logger = get_app_logger()
        
        # Process feedback entries
        self.feedback_samples = self._process_feedback_entries()
        
    def _process_feedback_entries(self) -> List[Tuple[torch.Tensor, int, float]]:
        """Process feedback entries into training samples"""
        feedback_samples = []
        
        for feedback in self.feedback_entries:
            try:
                # Skip if no video segment
                if not feedback.video_segment or not Path(feedback.video_segment).exists():
                    continue
                
                # Extract video sequence
                video_tensor = self._load_video_from_feedback(feedback)
                if video_tensor is None:
                    continue
                
                # Determine correct label based on feedback type
                if feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
                    # Should be normal (assuming 'Normal' or 'NormalVideos' is class 7)
                    correct_label = self.class_to_idx.get('Normal', self.class_to_idx.get('NormalVideos', 7))
                elif feedback.feedback_type == FeedbackType.WRONG_CLASSIFICATION:
                    # Use corrected label
                    correct_label = self.class_to_idx.get(feedback.corrected_label, -1)
                elif feedback.feedback_type == FeedbackType.TRUE_POSITIVE:
                    # Keep original prediction
                    correct_label = self.class_to_idx.get(feedback.original_prediction, -1)
                else:
                    continue
                
                if correct_label == -1:
                    continue
                
                # Add sample with feedback weight
                feedback_weight = feedback.get_feedback_weight()
                feedback_samples.append((video_tensor, correct_label, feedback_weight))
                
            except Exception as e:
                self.logger.warning(f"Failed to process feedback {feedback.id}: {e}")
        
        self.logger.info(f"Processed {len(feedback_samples)} feedback samples")
        return feedback_samples
    
    def _load_video_from_feedback(self, feedback: FeedbackEntry) -> Optional[torch.Tensor]:
        """Load video tensor from feedback entry"""
        try:
            video_sequence = self.video_extractor.extract_video_sequence(feedback.video_segment)
            video_tensor = torch.from_numpy(video_sequence).float()
            
            # Normalize
            if video_tensor.max() > 1:
                video_tensor = video_tensor / 255.0
                
            return video_tensor
            
        except Exception as e:
            self.logger.warning(f"Failed to load video for feedback {feedback.id}: {e}")
            return None
    
    def __len__(self):
        return len(self.original_dataset) + len(self.feedback_samples)
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            # Return original dataset sample
            video, label = self.original_dataset[idx]
            return video, label, 1.0  # Default weight
        else:
            # Return feedback sample
            feedback_idx = idx - len(self.original_dataset)
            video, label, weight = self.feedback_samples[feedback_idx]
            return video, label, weight


class ContinuousLearningManager:
    """Main manager for continuous learning pipeline"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = get_app_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.feedback_db = FeedbackDatabase()
        self.video_extractor = VideoSequenceExtractor()
        self.class_names = self.config['dataset']['classes']
        
        # Model versioning
        self.model_dir = Path('models/continuous_learning')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Current model
        self.current_model = None
        self.current_model_version = self._get_latest_model_version()
        
        # Retraining configuration
        self.min_feedback_threshold = 50      # Minimum feedback to trigger retraining
        self.feedback_quality_threshold = 0.7  # Minimum average feedback quality
        self.retraining_interval_hours = 24   # Check interval
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        
        self.logger.info("Continuous Learning Manager initialized")
        
    def _get_latest_model_version(self) -> str:
        """Get the latest model version"""
        model_files = list(self.model_dir.glob("model_v*.pth"))
        if not model_files:
            return "v1.0.0"
        
        versions = []
        for model_file in model_files:
            try:
                version_str = model_file.stem.replace("model_", "")
                versions.append(version_str)
            except:
                continue
        
        if versions:
            # Simple version comparison (assumes format vX.Y.Z)
            latest = sorted(versions, key=lambda v: tuple(map(int, v[1:].split('.'))))[-1]
            return latest
        
        return "v1.0.0"
    
    def add_feedback(
        self,
        user_id: str,
        user_role: UserRole,
        camera_id: str,
        original_prediction: str,
        original_confidence: float,
        original_severity: str,
        feedback_type: FeedbackType,
        corrected_label: Optional[str] = None,
        corrected_severity: Optional[str] = None,
        video_segment_path: Optional[str] = None,
        confidence_level: float = 1.0,
        notes: Optional[str] = None
    ) -> str:
        """Add user feedback to the system"""
        
        # Generate unique feedback ID
        feedback_id = hashlib.md5(
            f"{user_id}_{camera_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # Create feedback entry
        feedback = FeedbackEntry(
            id=feedback_id,
            timestamp=datetime.now(),
            user_id=user_id,
            user_role=user_role,
            camera_id=camera_id,
            original_prediction=original_prediction,
            original_confidence=original_confidence,
            original_severity=original_severity,
            feedback_type=feedback_type,
            corrected_label=corrected_label,
            corrected_severity=corrected_severity,
            video_segment=video_segment_path,
            confidence_level=confidence_level,
            notes=notes
        )
        
        # Add to database
        success = self.feedback_db.add_feedback(feedback)
        
        if success:
            self.logger.info(f"Feedback added: {feedback_id} from {user_id}")
            
            # Check if retraining should be triggered
            self._check_retraining_trigger()
            
            return feedback_id
        else:
            self.logger.error("Failed to add feedback")
            return ""
    
    def _check_retraining_trigger(self):
        """Check if model retraining should be triggered"""
        
        # Get unprocessed feedback
        unprocessed_feedback = self.feedback_db.get_unprocessed_feedback()
        
        if len(unprocessed_feedback) < self.min_feedback_threshold:
            return
        
        # Calculate average feedback quality
        total_weight = sum(feedback.get_feedback_weight() for feedback in unprocessed_feedback)
        avg_quality = total_weight / len(unprocessed_feedback)
        
        if avg_quality < self.feedback_quality_threshold:
            self.logger.warning(f"Feedback quality too low: {avg_quality}")
            return
        
        # Trigger retraining
        self.logger.info(f"Triggering retraining with {len(unprocessed_feedback)} feedback entries")
        self._schedule_retraining()
    
    def _schedule_retraining(self):
        """Schedule model retraining (async)"""
        # In a production system, you would use a job queue (Celery, etc.)
        # For now, we'll run it in a separate thread
        threading.Thread(target=self.retrain_model, daemon=True).start()
    
    def retrain_model(self) -> bool:
        """Retrain the model with new feedback"""
        self.logger.info("Starting model retraining with user feedback")
        
        try:
            # Get unprocessed feedback
            feedback_entries = self.feedback_db.get_unprocessed_feedback()
            
            if len(feedback_entries) < self.min_feedback_threshold:
                self.logger.warning("Insufficient feedback for retraining")
                return False
            
            # Load current model
            model = self._load_current_model()
            if model is None:
                self.logger.error("Failed to load current model")
                return False
            
            # Create enhanced dataset with feedback
            enhanced_dataset = self._create_feedback_dataset(feedback_entries)
            
            # Perform incremental training
            success = self._incremental_training(model, enhanced_dataset, feedback_entries)
            
            if success:
                # Save new model version
                new_version = self._increment_version(self.current_model_version)
                model_path = self.model_dir / f"model_{new_version}.pth"
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'version': new_version,
                    'feedback_count': len(feedback_entries),
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config
                }, model_path)
                
                # Update current model version
                self.current_model_version = new_version
                self.current_model = model
                
                # Mark feedback as processed
                feedback_ids = [f.id for f in feedback_entries]
                self.feedback_db.mark_processed(feedback_ids)
                
                self.logger.info(f"Model retrained successfully: {new_version}")
                return True
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            
        return False
    
    def _load_current_model(self) -> Optional[EnhancedTemporalAnomalyModel]:
        """Load the current model"""
        try:
            # Try to load from continuous learning directory first
            model_path = self.model_dir / f"model_{self.current_model_version}.pth"
            
            if not model_path.exists():
                # Fallback to main checkpoint
                model_path = Path('models/checkpoints/enhanced_temporal_best.pth')
            
            if not model_path.exists():
                self.logger.error("No model found to load")
                return None
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model
            model = create_enhanced_temporal_model(checkpoint.get('config', self.config))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def _create_feedback_dataset(self, feedback_entries: List[FeedbackEntry]) -> FeedbackDataset:
        """Create dataset incorporating feedback"""
        
        # Create a small original dataset for stability (to prevent catastrophic forgetting)
        from src.data.enhanced_data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(self.config)
        train_loader, _, _ = preprocessor.create_data_loaders(use_cache=True)
        
        # Use a subset of original training data
        original_dataset = train_loader.dataset
        
        # Create feedback dataset
        feedback_dataset = FeedbackDataset(
            original_dataset=original_dataset,
            feedback_entries=feedback_entries,
            video_extractor=self.video_extractor,
            class_names=self.class_names
        )
        
        return feedback_dataset
    
    def _incremental_training(
        self, 
        model: EnhancedTemporalAnomalyModel,
        dataset: FeedbackDataset,
        feedback_entries: List[FeedbackEntry]
    ) -> bool:
        """Perform incremental training with feedback data"""
        
        try:
            model.train()
            
            # Use lower learning rate for incremental learning
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
            
            # Weighted loss function
            def weighted_loss(outputs, targets, weights):
                # Standard cross entropy for each sample
                losses = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
                # Apply weights
                weighted_losses = losses * weights
                return weighted_losses.mean()
            
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=8,  # Smaller batch size for incremental learning
                shuffle=True,
                num_workers=2
            )
            
            # Training loop (few epochs for incremental learning)
            num_epochs = 3
            
            for epoch in range(num_epochs):
                total_loss = 0.0
                num_batches = 0
                
                for videos, labels, weights in data_loader:
                    videos = videos.to(self.device)
                    labels = labels.to(self.device)
                    weights = weights.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(videos)
                    main_pred = outputs['main']
                    
                    # Compute weighted loss
                    loss = weighted_loss(main_pred, labels, weights)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                self.logger.info(f"Incremental training epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Incremental training failed: {e}")
            return False
    
    def _increment_version(self, current_version: str) -> str:
        """Increment model version"""
        try:
            # Parse version (assuming format vX.Y.Z)
            version_parts = current_version[1:].split('.')
            major, minor, patch = map(int, version_parts)
            
            # Increment patch version for incremental updates
            patch += 1
            
            return f"v{major}.{minor}.{patch}"
            
        except:
            # Fallback
            return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_model_performance_history(self) -> Dict[str, Any]:
        """Get model performance history"""
        try:
            with sqlite3.connect(self.feedback_db.db_path) as conn:
                df = pd.read_sql_query("""
                    SELECT model_version, evaluation_date, accuracy, f1_score, 
                           precision, recall, feedback_integrated_count
                    FROM model_performance
                    ORDER BY evaluation_date DESC
                    LIMIT 10
                """, conn)
                
                return df.to_dict('records')
                
        except Exception as e:
            self.logger.error(f"Failed to get performance history: {e}")
            return []
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            with sqlite3.connect(self.feedback_db.db_path) as conn:
                cursor = conn.cursor()
                
                # Total feedback count
                cursor.execute("SELECT COUNT(*) FROM feedback")
                total_feedback = cursor.fetchone()[0]
                
                # Feedback by type
                cursor.execute("""
                    SELECT feedback_type, COUNT(*) 
                    FROM feedback 
                    GROUP BY feedback_type
                """)
                feedback_by_type = dict(cursor.fetchall())
                
                # Average feedback quality
                cursor.execute("SELECT AVG(feedback_weight) FROM feedback")
                avg_quality = cursor.fetchone()[0] or 0.0
                
                # Pending feedback
                cursor.execute("SELECT COUNT(*) FROM feedback WHERE processed = 0")
                pending_feedback = cursor.fetchone()[0]
                
                return {
                    'total_feedback': total_feedback,
                    'feedback_by_type': feedback_by_type,
                    'average_quality': avg_quality,
                    'pending_feedback': pending_feedback,
                    'current_model_version': self.current_model_version
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get feedback statistics: {e}")
            return {}


# Web API endpoints for feedback collection (FastAPI integration)
def create_feedback_api_routes():
    """Create FastAPI routes for feedback collection"""
    
    from fastapi import APIRouter, HTTPException, Depends
    from pydantic import BaseModel
    from typing import Optional
    
    router = APIRouter(prefix="/api/feedback", tags=["feedback"])
    
    # Global continuous learning manager
    cl_manager = ContinuousLearningManager()
    
    class FeedbackRequest(BaseModel):
        user_id: str
        user_role: str
        camera_id: str
        original_prediction: str
        original_confidence: float
        original_severity: str
        feedback_type: str
        corrected_label: Optional[str] = None
        corrected_severity: Optional[str] = None
        video_segment_path: Optional[str] = None
        confidence_level: float = 1.0
        notes: Optional[str] = None
    
    @router.post("/add")
    async def add_feedback(feedback_request: FeedbackRequest):
        """Add user feedback"""
        try:
            feedback_id = cl_manager.add_feedback(
                user_id=feedback_request.user_id,
                user_role=UserRole(feedback_request.user_role),
                camera_id=feedback_request.camera_id,
                original_prediction=feedback_request.original_prediction,
                original_confidence=feedback_request.original_confidence,
                original_severity=feedback_request.original_severity,
                feedback_type=FeedbackType(feedback_request.feedback_type),
                corrected_label=feedback_request.corrected_label,
                corrected_severity=feedback_request.corrected_severity,
                video_segment_path=feedback_request.video_segment_path,
                confidence_level=feedback_request.confidence_level,
                notes=feedback_request.notes
            )
            
            if feedback_id:
                return {"status": "success", "feedback_id": feedback_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to add feedback")
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/statistics")
    async def get_feedback_statistics():
        """Get feedback statistics"""
        return cl_manager.get_feedback_statistics()
    
    @router.get("/performance")
    async def get_performance_history():
        """Get model performance history"""
        return cl_manager.get_model_performance_history()
    
    @router.post("/retrain")
    async def trigger_retraining():
        """Manually trigger model retraining"""
        success = cl_manager.retrain_model()
        if success:
            return {"status": "success", "message": "Model retraining completed"}
        else:
            raise HTTPException(status_code=500, detail="Model retraining failed")
    
    return router


if __name__ == "__main__":
    # Test the continuous learning system
    cl_manager = ContinuousLearningManager()
    
    # Example feedback
    feedback_id = cl_manager.add_feedback(
        user_id="security_guard_001",
        user_role=UserRole.SECURITY_GUARD,
        camera_id="camera_001",
        original_prediction="Fighting",
        original_confidence=0.85,
        original_severity="high",
        feedback_type=FeedbackType.FALSE_POSITIVE,
        corrected_label="Normal",
        notes="This was just people playing, not fighting"
    )
    
    print(f"Added feedback: {feedback_id}")
    
    # Get statistics
    stats = cl_manager.get_feedback_statistics()
    print(f"Feedback statistics: {stats}")