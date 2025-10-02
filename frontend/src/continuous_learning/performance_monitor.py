"""
Performance Monitoring System
============================

Comprehensive system for monitoring model performance and system health in production.
"""

import json
import logging
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import torch
import mlflow
from collections import defaultdict, deque
import threading
import time
import aiofiles
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    DISK_USAGE = "disk_usage"
    ERROR_RATE = "error_rate"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    FALSE_NEGATIVE_RATE = "false_negative_rate"

@dataclass
class MetricThreshold:
    """Threshold configuration for metric monitoring"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    direction: str  # 'above' or 'below'
    window_size: int = 100  # Number of samples to consider

@dataclass
class PerformanceMetric:
    """Performance metric measurement"""
    timestamp: datetime
    metric_type: MetricType
    value: float
    camera_id: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PerformanceAlert:
    """Performance alert"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    camera_id: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class PerformanceMonitor:
    """Monitors system and model performance in real-time"""
    
    def __init__(self, db_path: str = "data/performance.db",
                 config_path: str = "config/monitoring.yaml"):
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        
        # Initialize database
        self._init_database()
        
        # Load monitoring configuration
        self.thresholds = self._load_thresholds()
        
        # Metric storage (recent values in memory)
        self.metric_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_buffer = deque(maxlen=500)
        
        # Monitoring flags
        self.monitoring_active = False
        self.monitoring_tasks = []
        
        # Performance statistics
        self.stats_cache = {}
        self.last_stats_update = None
        
        # MLflow tracking
        self.mlflow_enabled = True
        try:
            mlflow.set_experiment("performance_monitoring")
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            self.mlflow_enabled = False
    
    def _init_database(self):
        """Initialize SQLite database for performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance metrics table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    camera_id TEXT,
                    model_version TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Performance alerts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    camera_id TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # System health snapshots
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_percent REAL,
                    gpu_percent REAL,
                    gpu_memory_percent REAL,
                    active_cameras INTEGER,
                    alerts_per_hour REAL,
                    avg_processing_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create indices for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type ON performance_metrics(metric_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON performance_alerts(severity)')
                
                conn.commit()
                logger.info("Performance monitoring database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {e}")
            raise
    
    def _load_thresholds(self) -> Dict[MetricType, MetricThreshold]:
        """Load monitoring thresholds from configuration"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    threshold_configs = config.get('thresholds', {})
            else:
                # Default thresholds
                threshold_configs = {
                    'accuracy': {'warning': 0.85, 'critical': 0.8, 'direction': 'below'},
                    'latency': {'warning': 1.0, 'critical': 2.0, 'direction': 'above'},
                    'memory_usage': {'warning': 80.0, 'critical': 90.0, 'direction': 'above'},
                    'cpu_usage': {'warning': 80.0, 'critical': 95.0, 'direction': 'above'},
                    'error_rate': {'warning': 0.05, 'critical': 0.1, 'direction': 'above'}
                }
            
            thresholds = {}
            for metric_name, config in threshold_configs.items():
                try:
                    metric_type = MetricType(metric_name)
                    thresholds[metric_type] = MetricThreshold(
                        metric_type=metric_type,
                        warning_threshold=config['warning'],
                        critical_threshold=config['critical'],
                        direction=config['direction'],
                        window_size=config.get('window_size', 100)
                    )
                except ValueError:
                    logger.warning(f"Unknown metric type: {metric_name}")
            
            logger.info(f"Loaded {len(thresholds)} monitoring thresholds")
            return thresholds
            
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    async def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, metric_type, value, camera_id, model_version, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.metric_type.value,
                    metric.value,
                    metric.camera_id,
                    metric.model_version,
                    json.dumps(metric.metadata)
                ))
                conn.commit()
            
            # Store in memory buffer
            self.metric_buffer[metric.metric_type].append(metric)
            
            # Check thresholds
            await self._check_thresholds(metric)
            
            # Log to MLflow if enabled
            if self.mlflow_enabled:
                try:
                    with mlflow.start_run(nested=True):
                        mlflow.log_metric(metric.metric_type.value, metric.value)
                except Exception as e:
                    logger.debug(f"MLflow logging failed: {e}")
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    async def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric violates thresholds and create alerts"""
        try:
            threshold = self.thresholds.get(metric.metric_type)
            if not threshold:
                return
            
            # Get recent values for the metric
            recent_values = [m.value for m in list(self.metric_buffer[metric.metric_type])[-threshold.window_size:]]
            if len(recent_values) < 5:  # Need minimum samples
                return
            
            avg_value = np.mean(recent_values)
            severity = None
            threshold_value = None
            
            if threshold.direction == 'above':
                if avg_value > threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif avg_value > threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold
            else:  # below
                if avg_value < threshold.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif avg_value < threshold.warning_threshold:
                    severity = AlertSeverity.HIGH
                    threshold_value = threshold.warning_threshold
            
            if severity:
                alert = PerformanceAlert(
                    id=f"{metric.metric_type.value}_{int(time.time())}",
                    timestamp=datetime.now(),
                    severity=severity,
                    metric_type=metric.metric_type,
                    message=f"{metric.metric_type.value} {threshold.direction} threshold: {avg_value:.3f} (threshold: {threshold_value})",
                    value=avg_value,
                    threshold=threshold_value,
                    camera_id=metric.camera_id
                )
                
                await self._create_alert(alert)
                
        except Exception as e:
            logger.error(f"Failed to check thresholds: {e}")
    
    async def _create_alert(self, alert: PerformanceAlert):
        """Create a performance alert"""
        try:
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO performance_alerts (
                    id, timestamp, severity, metric_type, message,
                    value, threshold_value, camera_id, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity.value,
                    alert.metric_type.value,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    alert.camera_id,
                    alert.resolved
                ))
                conn.commit()
            
            # Store in memory buffer
            self.alerts_buffer.append(alert)
            
            logger.warning(f"Performance alert: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_model_performance()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        logger.info("Started performance monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.monitoring_active = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.monitoring_tasks.clear()
        logger.info("Stopped performance monitoring")
    
    async def _monitor_system_health(self):
        """Monitor system health metrics"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                await self.record_metric(PerformanceMetric(
                    timestamp=datetime.now(),
                    metric_type=MetricType.CPU_USAGE,
                    value=cpu_percent
                ))
                
                # Memory usage
                memory = psutil.virtual_memory()
                await self.record_metric(PerformanceMetric(
                    timestamp=datetime.now(),
                    metric_type=MetricType.MEMORY_USAGE,
                    value=memory.percent
                ))
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                await self.record_metric(PerformanceMetric(
                    timestamp=datetime.now(),
                    metric_type=MetricType.DISK_USAGE,
                    value=disk_percent
                ))
                
                # GPU usage (if available)
                try:
                    if torch.cuda.is_available():
                        gpu_util = torch.cuda.utilization()
                        gpu_memory = torch.cuda.memory_usage()
                        
                        await self.record_metric(PerformanceMetric(
                            timestamp=datetime.now(),
                            metric_type=MetricType.GPU_USAGE,
                            value=gpu_util
                        ))
                except Exception as e:
                    logger.debug(f"GPU monitoring failed: {e}")
                
                # Store system health snapshot
                await self._store_system_health_snapshot()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_model_performance(self):
        """Monitor model performance metrics"""
        while self.monitoring_active:
            try:
                # This would integrate with the model inference pipeline
                # to collect performance metrics in real-time
                
                # Placeholder for actual model performance monitoring
                # In practice, this would collect metrics from the inference engine
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Model performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _store_system_health_snapshot(self):
        """Store a comprehensive system health snapshot"""
        try:
            timestamp = datetime.now()
            
            # Get current metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            gpu_percent = 0
            gpu_memory_percent = 0
            try:
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.utilization()
                    gpu_memory_percent = torch.cuda.memory_usage()
            except:
                pass
            
            # Get recent alert rate
            recent_alerts = [a for a in self.alerts_buffer 
                           if (timestamp - a.timestamp).total_seconds() < 3600]
            alerts_per_hour = len(recent_alerts)
            
            # Store snapshot
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO system_health (
                    timestamp, cpu_percent, memory_percent, disk_percent,
                    gpu_percent, gpu_memory_percent, active_cameras,
                    alerts_per_hour, avg_processing_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    cpu_percent,
                    memory_percent,
                    disk_percent,
                    gpu_percent,
                    gpu_memory_percent,
                    0,  # Would get from camera manager
                    alerts_per_hour,
                    0   # Would calculate from processing metrics
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store system health snapshot: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old performance data"""
        while self.monitoring_active:
            try:
                # Clean up data older than 30 days
                cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Clean old metrics
                    cursor.execute(
                        'DELETE FROM performance_metrics WHERE timestamp < ?',
                        (cutoff_date,)
                    )
                    
                    # Clean old resolved alerts
                    cursor.execute(
                        'DELETE FROM performance_alerts WHERE resolved = 1 AND timestamp < ?',
                        (cutoff_date,)
                    )
                    
                    # Clean old system health data
                    cursor.execute(
                        'DELETE FROM system_health WHERE timestamp < ?',
                        (cutoff_date,)
                    )
                    
                    conn.commit()
                
                logger.info("Cleaned up old performance data")
                await asyncio.sleep(24 * 3600)  # Clean once per day
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_performance_summary(self, 
                                    hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        try:
            start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get metric summaries
                cursor.execute('''
                SELECT metric_type, 
                       AVG(value) as avg_value,
                       MIN(value) as min_value,
                       MAX(value) as max_value,
                       COUNT(*) as count
                FROM performance_metrics
                WHERE timestamp > ?
                GROUP BY metric_type
                ''', (start_time,))
                
                metric_summaries = {}
                for row in cursor.fetchall():
                    metric_summaries[row[0]] = {
                        'average': row[1],
                        'minimum': row[2],
                        'maximum': row[3],
                        'sample_count': row[4]
                    }
                
                # Get alert counts
                cursor.execute('''
                SELECT severity, COUNT(*) as count
                FROM performance_alerts
                WHERE timestamp > ? AND resolved = 0
                GROUP BY severity
                ''', (start_time,))
                
                alert_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Get system health trend
                cursor.execute('''
                SELECT timestamp, cpu_percent, memory_percent, disk_percent
                FROM system_health
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
                ''', (start_time,))
                
                health_data = [
                    {
                        'timestamp': row[0],
                        'cpu_percent': row[1],
                        'memory_percent': row[2],
                        'disk_percent': row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                return {
                    'period_hours': hours,
                    'metric_summaries': metric_summaries,
                    'alert_counts': alert_counts,
                    'total_unresolved_alerts': sum(alert_counts.values()),
                    'system_health_trend': health_data,
                    'monitoring_active': self.monitoring_active,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM performance_alerts
                WHERE resolved = 0
                ORDER BY severity DESC, timestamp DESC
                ''')
                
                columns = [desc[0] for desc in cursor.description]
                alerts = []
                
                for row in cursor.fetchall():
                    alert_dict = dict(zip(columns, row))
                    alerts.append(alert_dict)
                
                return alerts
                
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE performance_alerts
                SET resolved = 1, resolved_at = ?
                WHERE id = ?
                ''', (datetime.now().isoformat(), alert_id))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Resolved alert: {alert_id}")
                    return True
                else:
                    logger.warning(f"Alert not found: {alert_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def generate_performance_report(self, 
                                        days: int = 7,
                                        output_path: Optional[str] = None) -> str:
        """Generate a comprehensive performance report"""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"reports/performance_report_{timestamp}.html"
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get data for the report
            start_time = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Get metric data
                metrics_df = pd.read_sql_query('''
                SELECT * FROM performance_metrics
                WHERE timestamp > ?
                ORDER BY timestamp
                ''', conn, params=(start_time,))
                
                # Get alert data
                alerts_df = pd.read_sql_query('''
                SELECT * FROM performance_alerts
                WHERE timestamp > ?
                ORDER BY timestamp
                ''', conn, params=(start_time,))
                
                # Get system health data
                health_df = pd.read_sql_query('''
                SELECT * FROM system_health
                WHERE timestamp > ?
                ORDER BY timestamp
                ''', conn, params=(start_time,))
            
            # Generate visualizations
            plots = self._generate_performance_plots(metrics_df, health_df, alerts_df)
            
            # Generate HTML report
            html_content = self._generate_html_report(
                metrics_df, alerts_df, health_df, plots, days
            )
            
            # Save report
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated performance report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            raise
    
    def _generate_performance_plots(self, 
                                  metrics_df: pd.DataFrame,
                                  health_df: pd.DataFrame,
                                  alerts_df: pd.DataFrame) -> Dict[str, str]:
        """Generate performance visualization plots"""
        plots = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Metric trends plot
            if not metrics_df.empty:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Performance Metrics Trends', fontsize=16)
                
                # Group by metric type and plot
                for i, metric_type in enumerate(['accuracy', 'latency', 'cpu_usage', 'memory_usage']):
                    ax = axes[i//2, i%2]
                    metric_data = metrics_df[metrics_df['metric_type'] == metric_type]
                    
                    if not metric_data.empty:
                        metric_data['timestamp'] = pd.to_datetime(metric_data['timestamp'])
                        ax.plot(metric_data['timestamp'], metric_data['value'])
                        ax.set_title(f'{metric_type.replace("_", " ").title()}')
                        ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plots['metrics_trends'] = 'data:image/png;base64,' + self._plot_to_base64(fig)
                plt.close(fig)
            
            # Alert severity distribution
            if not alerts_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                severity_counts = alerts_df['severity'].value_counts()
                ax.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
                ax.set_title('Alert Distribution by Severity')
                plots['alert_distribution'] = 'data:image/png;base64,' + self._plot_to_base64(fig)
                plt.close(fig)
            
            # System health over time
            if not health_df.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                health_df['timestamp'] = pd.to_datetime(health_df['timestamp'])
                
                ax.plot(health_df['timestamp'], health_df['cpu_percent'], label='CPU %')
                ax.plot(health_df['timestamp'], health_df['memory_percent'], label='Memory %')
                ax.plot(health_df['timestamp'], health_df['disk_percent'], label='Disk %')
                
                ax.set_title('System Health Over Time')
                ax.set_ylabel('Usage %')
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                
                plots['system_health'] = 'data:image/png;base64,' + self._plot_to_base64(fig)
                plt.close(fig)
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
        
        return plots
    
    def _plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        import io
        import base64
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        
        return base64.b64encode(plot_data).decode()
    
    def _generate_html_report(self, 
                            metrics_df: pd.DataFrame,
                            alerts_df: pd.DataFrame,
                            health_df: pd.DataFrame,
                            plots: Dict[str, str],
                            days: int) -> str:
        """Generate HTML report content"""
        
        # Calculate summary statistics
        total_metrics = len(metrics_df)
        total_alerts = len(alerts_df)
        unresolved_alerts = len(alerts_df[alerts_df['resolved'] == 0])
        
        avg_cpu = health_df['cpu_percent'].mean() if not health_df.empty else 0
        avg_memory = health_df['memory_percent'].mean() if not health_df.empty else 0
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Report - {days} Days</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .summary-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .alert {{ background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 5px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Monitoring Report</h1>
                <p>Report Period: Last {days} days</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <div class="summary-box">
                    <h3>{total_metrics}</h3>
                    <p>Total Metrics Collected</p>
                </div>
                <div class="summary-box">
                    <h3>{total_alerts}</h3>
                    <p>Total Alerts</p>
                </div>
                <div class="summary-box">
                    <h3>{unresolved_alerts}</h3>
                    <p>Unresolved Alerts</p>
                </div>
                <div class="summary-box">
                    <h3>{avg_cpu:.1f}%</h3>
                    <p>Average CPU Usage</p>
                </div>
                <div class="summary-box">
                    <h3>{avg_memory:.1f}%</h3>
                    <p>Average Memory Usage</p>
                </div>
            </div>
        """
        
        # Add plots
        for plot_name, plot_data in plots.items():
            html_template += f'''
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{plot_data}" alt="{plot_name}" style="max-width: 100%;">
            </div>
            '''
        
        # Add recent alerts table
        if not alerts_df.empty:
            html_template += '''
            <h3>Recent Alerts</h3>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>Severity</th>
                    <th>Metric</th>
                    <th>Message</th>
                    <th>Status</th>
                </tr>
            '''
            
            for _, alert in alerts_df.tail(20).iterrows():
                status = "Resolved" if alert['resolved'] else "Active"
                html_template += f'''
                <tr>
                    <td>{alert['timestamp']}</td>
                    <td>{alert['severity']}</td>
                    <td>{alert['metric_type']}</td>
                    <td>{alert['message']}</td>
                    <td>{status}</td>
                </tr>
                '''
            
            html_template += '</table>'
        
        html_template += '''
        </body>
        </html>
        '''
        
        return html_template

# Export classes
__all__ = ['PerformanceMonitor', 'PerformanceMetric', 'PerformanceAlert', 'MetricThreshold', 'MetricType', 'AlertSeverity']