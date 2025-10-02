"""
Feedback Collection System
=========================

Collects and processes user feedback to improve model performance through continuous learning.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiofiles
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    TRUE_POSITIVE = "true_positive"    # User confirms detection is correct
    FALSE_POSITIVE = "false_positive"  # User says detection is wrong
    FALSE_NEGATIVE = "false_negative"  # User reports missed detection
    SEVERITY_CORRECTION = "severity_correction"  # User corrects severity level
    CLASSIFICATION_CORRECTION = "classification_correction"  # User corrects event type

class ConfidenceLevel(Enum):
    """User confidence in their feedback"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERT = "expert"

@dataclass
class FeedbackEntry:
    """Structure for storing user feedback"""
    id: str
    timestamp: datetime
    user_id: str
    camera_id: str
    alert_id: Optional[str]
    feedback_type: FeedbackType
    confidence_level: ConfidenceLevel
    original_prediction: Dict[str, Any]
    user_correction: Dict[str, Any]
    image_path: Optional[str]
    video_path: Optional[str]
    metadata: Dict[str, Any]
    processing_status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['feedback_type'] = self.feedback_type.value
        data['confidence_level'] = self.confidence_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['feedback_type'] = FeedbackType(data['feedback_type'])
        data['confidence_level'] = ConfidenceLevel(data['confidence_level'])
        return cls(**data)

class FeedbackCollector:
    """Collects and manages user feedback for continuous learning"""
    
    def __init__(self, db_path: str = "data/feedback.db", 
                 storage_path: str = "data/feedback_storage/"):
        self.db_path = Path(db_path)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Feedback processing queue
        self.feedback_queue = asyncio.Queue()
        self.processing_task = None
        
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Feedback entries table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    alert_id TEXT,
                    feedback_type TEXT NOT NULL,
                    confidence_level TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,
                    user_correction TEXT NOT NULL,
                    image_path TEXT,
                    video_path TEXT,
                    metadata TEXT NOT NULL,
                    processing_status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # User statistics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback_stats (
                    user_id TEXT PRIMARY KEY,
                    total_feedback_count INTEGER DEFAULT 0,
                    accuracy_score REAL DEFAULT 0.0,
                    expertise_level TEXT DEFAULT 'beginner',
                    last_feedback_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Model performance tracking
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    feedback_incorporated INTEGER DEFAULT 0,
                    accuracy_improvement REAL DEFAULT 0.0,
                    false_positive_reduction REAL DEFAULT 0.0,
                    false_negative_reduction REAL DEFAULT 0.0,
                    evaluation_date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                conn.commit()
                logger.info("Feedback database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize feedback database: {e}")
            raise
    
    async def collect_feedback(self, 
                             user_id: str,
                             camera_id: str,
                             feedback_type: FeedbackType,
                             confidence_level: ConfidenceLevel,
                             original_prediction: Dict[str, Any],
                             user_correction: Dict[str, Any],
                             alert_id: Optional[str] = None,
                             image_data: Optional[bytes] = None,
                             video_data: Optional[bytes] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect user feedback asynchronously
        
        Args:
            user_id: ID of the user providing feedback
            camera_id: ID of the camera that generated the detection
            feedback_type: Type of feedback being provided
            confidence_level: User's confidence in their feedback
            original_prediction: Original model prediction
            user_correction: User's correction or confirmation
            alert_id: Optional alert ID if feedback is for an alert
            image_data: Optional image data for the detection
            video_data: Optional video data for the detection
            metadata: Additional metadata
            
        Returns:
            Feedback entry ID
        """
        try:
            # Generate unique feedback ID
            feedback_id = self._generate_feedback_id(user_id, camera_id)
            
            # Store media files if provided
            image_path = None
            video_path = None
            
            if image_data:
                image_path = await self._store_media_file(
                    feedback_id, "image.jpg", image_data
                )
            
            if video_data:
                video_path = await self._store_media_file(
                    feedback_id, "video.mp4", video_data
                )
            
            # Create feedback entry
            feedback_entry = FeedbackEntry(
                id=feedback_id,
                timestamp=datetime.now(),
                user_id=user_id,
                camera_id=camera_id,
                alert_id=alert_id,
                feedback_type=feedback_type,
                confidence_level=confidence_level,
                original_prediction=original_prediction,
                user_correction=user_correction,
                image_path=image_path,
                video_path=video_path,
                metadata=metadata or {}
            )
            
            # Store in database
            await self._store_feedback_entry(feedback_entry)
            
            # Add to processing queue
            await self.feedback_queue.put(feedback_entry)
            
            # Update user statistics
            await self._update_user_stats(user_id)
            
            logger.info(f"Feedback collected: {feedback_id} from user {user_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            raise
    
    def _generate_feedback_id(self, user_id: str, camera_id: str) -> str:
        """Generate unique feedback ID"""
        timestamp = datetime.now().isoformat()
        content = f"{user_id}_{camera_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _store_media_file(self, feedback_id: str, filename: str, data: bytes) -> str:
        """Store media file and return path"""
        try:
            # Create feedback-specific directory
            feedback_dir = self.storage_path / feedback_id
            feedback_dir.mkdir(exist_ok=True)
            
            file_path = feedback_dir / filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to store media file: {e}")
            return None
    
    async def _store_feedback_entry(self, feedback_entry: FeedbackEntry):
        """Store feedback entry in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                data = feedback_entry.to_dict()
                cursor.execute('''
                INSERT INTO feedback_entries (
                    id, timestamp, user_id, camera_id, alert_id,
                    feedback_type, confidence_level, original_prediction,
                    user_correction, image_path, video_path, metadata,
                    processing_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['id'], data['timestamp'], data['user_id'],
                    data['camera_id'], data['alert_id'], data['feedback_type'],
                    data['confidence_level'], json.dumps(data['original_prediction']),
                    json.dumps(data['user_correction']), data['image_path'],
                    data['video_path'], json.dumps(data['metadata']),
                    data['processing_status']
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store feedback entry: {e}")
            raise
    
    async def _update_user_stats(self, user_id: str):
        """Update user feedback statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current stats
                cursor.execute(
                    'SELECT total_feedback_count FROM user_feedback_stats WHERE user_id = ?',
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    # Update existing record
                    new_count = result[0] + 1
                    cursor.execute('''
                    UPDATE user_feedback_stats 
                    SET total_feedback_count = ?, 
                        last_feedback_date = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                    ''', (new_count, datetime.now().isoformat(), user_id))
                else:
                    # Create new record
                    cursor.execute('''
                    INSERT INTO user_feedback_stats (
                        user_id, total_feedback_count, last_feedback_date
                    ) VALUES (?, ?, ?)
                    ''', (user_id, 1, datetime.now().isoformat()))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update user stats: {e}")
    
    async def get_feedback_summary(self, 
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of collected feedback"""
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get feedback counts by type
                cursor.execute('''
                SELECT feedback_type, COUNT(*) as count
                FROM feedback_entries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY feedback_type
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                feedback_by_type = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get feedback by confidence level
                cursor.execute('''
                SELECT confidence_level, COUNT(*) as count
                FROM feedback_entries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY confidence_level
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                feedback_by_confidence = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get most active users
                cursor.execute('''
                SELECT user_id, COUNT(*) as count
                FROM feedback_entries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY user_id
                ORDER BY count DESC
                LIMIT 10
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                top_users = [{"user_id": row[0], "feedback_count": row[1]} 
                           for row in cursor.fetchall()]
                
                # Get camera feedback distribution
                cursor.execute('''
                SELECT camera_id, COUNT(*) as count
                FROM feedback_entries
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY camera_id
                ORDER BY count DESC
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                camera_feedback = [{"camera_id": row[0], "feedback_count": row[1]} 
                                 for row in cursor.fetchall()]
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    },
                    "total_feedback": sum(feedback_by_type.values()),
                    "feedback_by_type": feedback_by_type,
                    "feedback_by_confidence": feedback_by_confidence,
                    "top_users": top_users,
                    "camera_feedback": camera_feedback,
                    "processing_queue_size": self.feedback_queue.qsize()
                }
                
        except Exception as e:
            logger.error(f"Failed to get feedback summary: {e}")
            return {}
    
    async def get_training_data(self, 
                              feedback_types: Optional[List[FeedbackType]] = None,
                              min_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
                              limit: Optional[int] = None) -> List[FeedbackEntry]:
        """
        Get feedback data suitable for model training
        
        Args:
            feedback_types: Types of feedback to include
            min_confidence: Minimum confidence level required
            limit: Maximum number of entries to return
            
        Returns:
            List of feedback entries suitable for training
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                where_clauses = ["processing_status = 'pending' OR processing_status = 'processed'"]
                params = []
                
                if feedback_types:
                    type_placeholders = ','.join(['?' for _ in feedback_types])
                    where_clauses.append(f"feedback_type IN ({type_placeholders})")
                    params.extend([ft.value for ft in feedback_types])
                
                # Confidence level filtering
                confidence_order = {
                    ConfidenceLevel.LOW: 1,
                    ConfidenceLevel.MEDIUM: 2,
                    ConfidenceLevel.HIGH: 3,
                    ConfidenceLevel.EXPERT: 4
                }
                
                min_conf_value = confidence_order[min_confidence]
                confidence_filter = []
                for conf, value in confidence_order.items():
                    if value >= min_conf_value:
                        confidence_filter.append(conf.value)
                
                if confidence_filter:
                    conf_placeholders = ','.join(['?' for _ in confidence_filter])
                    where_clauses.append(f"confidence_level IN ({conf_placeholders})")
                    params.extend(confidence_filter)
                
                where_clause = ' AND '.join(where_clauses)
                
                query = f'''
                SELECT * FROM feedback_entries
                WHERE {where_clause}
                ORDER BY timestamp DESC
                '''
                
                if limit:
                    query += f' LIMIT {limit}'
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to FeedbackEntry objects
                columns = [desc[0] for desc in cursor.description]
                feedback_entries = []
                
                for row in rows:
                    data = dict(zip(columns, row))
                    # Parse JSON fields
                    data['original_prediction'] = json.loads(data['original_prediction'])
                    data['user_correction'] = json.loads(data['user_correction'])
                    data['metadata'] = json.loads(data['metadata'])
                    
                    feedback_entries.append(FeedbackEntry.from_dict(data))
                
                logger.info(f"Retrieved {len(feedback_entries)} training feedback entries")
                return feedback_entries
                
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []
    
    async def mark_feedback_processed(self, feedback_ids: List[str]):
        """Mark feedback entries as processed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                placeholders = ','.join(['?' for _ in feedback_ids])
                cursor.execute(f'''
                UPDATE feedback_entries
                SET processing_status = 'processed'
                WHERE id IN ({placeholders})
                ''', feedback_ids)
                
                conn.commit()
                logger.info(f"Marked {len(feedback_ids)} feedback entries as processed")
                
        except Exception as e:
            logger.error(f"Failed to mark feedback as processed: {e}")
    
    async def start_processing(self):
        """Start background feedback processing"""
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_feedback_queue())
            logger.info("Started feedback processing task")
    
    async def stop_processing(self):
        """Stop background feedback processing"""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped feedback processing task")
    
    async def _process_feedback_queue(self):
        """Process feedback entries from the queue"""
        while True:
            try:
                # Get feedback entry from queue
                feedback_entry = await self.feedback_queue.get()
                
                # Process the feedback (placeholder for actual processing logic)
                await self._process_single_feedback(feedback_entry)
                
                # Mark task as done
                self.feedback_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_single_feedback(self, feedback_entry: FeedbackEntry):
        """Process a single feedback entry"""
        try:
            # Validate feedback
            if self._validate_feedback(feedback_entry):
                # Update processing status
                await self._update_processing_status(feedback_entry.id, "validated")
                logger.info(f"Processed feedback: {feedback_entry.id}")
            else:
                await self._update_processing_status(feedback_entry.id, "invalid")
                logger.warning(f"Invalid feedback: {feedback_entry.id}")
                
        except Exception as e:
            logger.error(f"Failed to process feedback {feedback_entry.id}: {e}")
            await self._update_processing_status(feedback_entry.id, "error")
    
    def _validate_feedback(self, feedback_entry: FeedbackEntry) -> bool:
        """Validate feedback entry"""
        try:
            # Basic validation checks
            if not feedback_entry.user_id or not feedback_entry.camera_id:
                return False
            
            if not feedback_entry.original_prediction or not feedback_entry.user_correction:
                return False
            
            # Additional validation logic can be added here
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    async def _update_processing_status(self, feedback_id: str, status: str):
        """Update processing status of feedback entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE feedback_entries SET processing_status = ? WHERE id = ?',
                    (status, feedback_id)
                )
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update processing status: {e}")

# Export classes for use in other modules
__all__ = ['FeedbackCollector', 'FeedbackEntry', 'FeedbackType', 'ConfidenceLevel']