"""
A/B Testing Framework
====================

Framework for conducting A/B tests on different model versions and configurations.
"""

import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import uuid
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Status of A/B test"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TestType(Enum):
    """Type of A/B test"""
    MODEL_COMPARISON = "model_comparison"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    ALGORITHM_COMPARISON = "algorithm_comparison"
    FEATURE_TESTING = "feature_testing"

@dataclass
class TestVariant:
    """A/B test variant configuration"""
    id: str
    name: str
    description: str
    model_version: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    traffic_allocation: float = 0.5
    is_control: bool = False

@dataclass
class TestMetric:
    """Metric to track in A/B test"""
    name: str
    description: str
    metric_type: str  # 'accuracy', 'precision', 'recall', 'f1', 'custom'
    higher_is_better: bool = True
    minimum_detectable_effect: float = 0.05

@dataclass
class ABTest:
    """A/B test configuration"""
    id: str
    name: str
    description: str
    test_type: TestType
    status: TestStatus
    variants: List[TestVariant]
    metrics: List[TestMetric]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_sample_size: int = 1000
    confidence_level: float = 0.95
    power: float = 0.8
    created_by: str = ""
    created_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TestResult:
    """Result of A/B test measurement"""
    test_id: str
    variant_id: str
    user_id: str
    camera_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ABTestManager:
    """Manages A/B tests for model experimentation"""
    
    def __init__(self, db_path: str = "data/ab_tests.db",
                 results_path: str = "data/ab_test_results/"):
        self.db_path = Path(db_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Active tests cache
        self.active_tests = {}
        self._load_active_tests()
        
        # User assignments cache
        self.user_assignments = {}
    
    def _init_database(self):
        """Initialize SQLite database for A/B testing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # A/B tests table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    test_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    variants TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    target_sample_size INTEGER,
                    confidence_level REAL,
                    power REAL,
                    created_by TEXT,
                    created_at TEXT,
                    metadata TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Test results table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (test_id) REFERENCES ab_tests (id)
                )
                ''')
                
                # User assignments table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    assigned_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(test_id, user_id),
                    FOREIGN KEY (test_id) REFERENCES ab_tests (id)
                )
                ''')
                
                conn.commit()
                logger.info("A/B testing database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize A/B testing database: {e}")
            raise
    
    def _load_active_tests(self):
        """Load active tests from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT * FROM ab_tests 
                WHERE status = 'running'
                ''')
                
                for row in cursor.fetchall():
                    test_data = self._row_to_test(row)
                    self.active_tests[test_data.id] = test_data
                
                logger.info(f"Loaded {len(self.active_tests)} active tests")
                
        except Exception as e:
            logger.error(f"Failed to load active tests: {e}")
    
    def _row_to_test(self, row) -> ABTest:
        """Convert database row to ABTest object"""
        return ABTest(
            id=row[0],
            name=row[1],
            description=row[2],
            test_type=TestType(row[3]),
            status=TestStatus(row[4]),
            variants=[TestVariant(**v) for v in json.loads(row[5])],
            metrics=[TestMetric(**m) for m in json.loads(row[6])],
            start_date=datetime.fromisoformat(row[7]) if row[7] else None,
            end_date=datetime.fromisoformat(row[8]) if row[8] else None,
            target_sample_size=row[9],
            confidence_level=row[10],
            power=row[11],
            created_by=row[12],
            created_at=datetime.fromisoformat(row[13]),
            metadata=json.loads(row[14]) if row[14] else {}
        )
    
    async def create_test(self, test_config: ABTest) -> str:
        """Create a new A/B test"""
        try:
            # Validate test configuration
            self._validate_test_config(test_config)
            
            # Generate test ID if not provided
            if not test_config.id:
                test_config.id = str(uuid.uuid4())
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO ab_tests (
                    id, name, description, test_type, status, variants,
                    metrics, start_date, end_date, target_sample_size,
                    confidence_level, power, created_by, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_config.id,
                    test_config.name,
                    test_config.description,
                    test_config.test_type.value,
                    test_config.status.value,
                    json.dumps([asdict(v) for v in test_config.variants]),
                    json.dumps([asdict(m) for m in test_config.metrics]),
                    test_config.start_date.isoformat() if test_config.start_date else None,
                    test_config.end_date.isoformat() if test_config.end_date else None,
                    test_config.target_sample_size,
                    test_config.confidence_level,
                    test_config.power,
                    test_config.created_by,
                    test_config.created_at.isoformat(),
                    json.dumps(test_config.metadata)
                ))
                conn.commit()
            
            logger.info(f"Created A/B test: {test_config.id}")
            return test_config.id
            
        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            raise
    
    def _validate_test_config(self, test_config: ABTest):
        """Validate A/B test configuration"""
        if len(test_config.variants) < 2:
            raise ValueError("Test must have at least 2 variants")
        
        total_allocation = sum(v.traffic_allocation for v in test_config.variants)
        if not (0.99 <= total_allocation <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        control_variants = [v for v in test_config.variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Test must have exactly one control variant")
        
        if not test_config.metrics:
            raise ValueError("Test must have at least one metric")
    
    async def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE ab_tests 
                SET status = 'running', start_date = ?
                WHERE id = ? AND status = 'draft'
                ''', (datetime.now().isoformat(), test_id))
                
                if cursor.rowcount == 0:
                    logger.warning(f"Could not start test {test_id} - may not exist or not in draft status")
                    return False
                
                conn.commit()
            
            # Load test into active tests cache
            test_data = await self.get_test(test_id)
            if test_data:
                self.active_tests[test_id] = test_data
                logger.info(f"Started A/B test: {test_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to start test {test_id}: {e}")
            return False
    
    async def stop_test(self, test_id: str, reason: str = "") -> bool:
        """Stop an A/B test"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE ab_tests 
                SET status = 'completed', end_date = ?
                WHERE id = ? AND status = 'running'
                ''', (datetime.now().isoformat(), test_id))
                
                if cursor.rowcount == 0:
                    logger.warning(f"Could not stop test {test_id} - may not exist or not running")
                    return False
                
                conn.commit()
            
            # Remove from active tests cache
            if test_id in self.active_tests:
                del self.active_tests[test_id]
            
            logger.info(f"Stopped A/B test: {test_id}. Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop test {test_id}: {e}")
            return False
    
    def assign_user_to_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign user to a test variant"""
        try:
            if test_id not in self.active_tests:
                return None
            
            # Check if user already assigned
            assignment_key = f"{test_id}_{user_id}"
            if assignment_key in self.user_assignments:
                return self.user_assignments[assignment_key]
            
            # Check database for existing assignment
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT variant_id FROM user_assignments
                WHERE test_id = ? AND user_id = ?
                ''', (test_id, user_id))
                
                result = cursor.fetchone()
                if result:
                    variant_id = result[0]
                    self.user_assignments[assignment_key] = variant_id
                    return variant_id
            
            # Assign to new variant
            test = self.active_tests[test_id]
            variant_id = self._determine_variant(test, user_id)
            
            # Store assignment
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO user_assignments (test_id, user_id, variant_id)
                VALUES (?, ?, ?)
                ''', (test_id, user_id, variant_id))
                conn.commit()
            
            self.user_assignments[assignment_key] = variant_id
            logger.debug(f"Assigned user {user_id} to variant {variant_id} in test {test_id}")
            return variant_id
            
        except Exception as e:
            logger.error(f"Failed to assign user to variant: {e}")
            return None
    
    def _determine_variant(self, test: ABTest, user_id: str) -> str:
        """Determine which variant to assign user to"""
        # Use consistent hashing for deterministic assignment
        hash_input = f"{test.id}_{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Assign based on traffic allocation
        cumulative_allocation = 0.0
        for variant in test.variants:
            cumulative_allocation += variant.traffic_allocation
            if random_value <= cumulative_allocation:
                return variant.id
        
        # Fallback to control variant
        control_variant = next(v for v in test.variants if v.is_control)
        return control_variant.id
    
    async def record_result(self, test_result: TestResult):
        """Record a test result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT INTO test_results (
                    test_id, variant_id, user_id, camera_id,
                    timestamp, metrics, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    test_result.test_id,
                    test_result.variant_id,
                    test_result.user_id,
                    test_result.camera_id,
                    test_result.timestamp.isoformat(),
                    json.dumps(test_result.metrics),
                    json.dumps(test_result.metadata)
                ))
                conn.commit()
            
            logger.debug(f"Recorded test result for test {test_result.test_id}")
            
        except Exception as e:
            logger.error(f"Failed to record test result: {e}")
    
    async def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM ab_tests WHERE id = ?', (test_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_test(row)
                return None
                
        except Exception as e:
            logger.error(f"Failed to get test {test_id}: {e}")
            return None
    
    async def list_tests(self, status: Optional[TestStatus] = None) -> List[ABTest]:
        """List all tests, optionally filtered by status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if status:
                    cursor.execute('SELECT * FROM ab_tests WHERE status = ?', (status.value,))
                else:
                    cursor.execute('SELECT * FROM ab_tests')
                
                tests = []
                for row in cursor.fetchall():
                    tests.append(self._row_to_test(row))
                
                return tests
                
        except Exception as e:
            logger.error(f"Failed to list tests: {e}")
            return []
    
    async def analyze_test_results(self, test_id: str) -> Dict[str, Any]:
        """Analyze results of an A/B test"""
        try:
            test = await self.get_test(test_id)
            if not test:
                raise ValueError(f"Test {test_id} not found")
            
            # Get all results for this test
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT variant_id, metrics, timestamp FROM test_results
                WHERE test_id = ?
                ORDER BY timestamp
                ''', (test_id,))
                
                results = cursor.fetchall()
            
            if not results:
                return {"error": "No results found for this test"}
            
            # Organize results by variant
            variant_results = {}
            for variant_id, metrics_json, timestamp in results:
                if variant_id not in variant_results:
                    variant_results[variant_id] = []
                
                metrics = json.loads(metrics_json)
                variant_results[variant_id].append({
                    'metrics': metrics,
                    'timestamp': timestamp
                })
            
            # Calculate statistics for each metric
            analysis = {
                'test_id': test_id,
                'test_name': test.name,
                'analysis_timestamp': datetime.now().isoformat(),
                'variants': {},
                'statistical_tests': {},
                'recommendations': []
            }
            
            # Find control variant
            control_variant = next(v for v in test.variants if v.is_control)
            control_variant_id = control_variant.id
            
            # Analyze each variant
            for variant_id, variant_data in variant_results.items():
                variant_info = next(v for v in test.variants if v.id == variant_id)
                
                analysis['variants'][variant_id] = {
                    'name': variant_info.name,
                    'is_control': variant_info.is_control,
                    'sample_size': len(variant_data),
                    'metrics': {}
                }
                
                # Calculate metric statistics
                for metric in test.metrics:
                    metric_values = [r['metrics'].get(metric.name, 0) for r in variant_data]
                    
                    analysis['variants'][variant_id]['metrics'][metric.name] = {
                        'mean': np.mean(metric_values),
                        'std': np.std(metric_values),
                        'median': np.median(metric_values),
                        'count': len([v for v in metric_values if v is not None])
                    }
            
            # Statistical significance testing
            if len(variant_results) >= 2 and control_variant_id in variant_results:
                control_data = variant_results[control_variant_id]
                
                for variant_id, variant_data in variant_results.items():
                    if variant_id == control_variant_id:
                        continue
                    
                    analysis['statistical_tests'][variant_id] = {}
                    
                    for metric in test.metrics:
                        control_values = [r['metrics'].get(metric.name, 0) for r in control_data]
                        variant_values = [r['metrics'].get(metric.name, 0) for r in variant_data]
                        
                        # Perform t-test
                        try:
                            t_stat, p_value = stats.ttest_ind(variant_values, control_values)
                            
                            analysis['statistical_tests'][variant_id][metric.name] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < (1 - test.confidence_level),
                                'effect_size': (np.mean(variant_values) - np.mean(control_values)) / np.std(control_values) if np.std(control_values) > 0 else 0
                            }
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {metric.name}: {e}")
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis, test)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze test results: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, analysis: Dict[str, Any], test: ABTest) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        try:
            control_variant = next(v for v in test.variants if v.is_control)
            control_id = control_variant.id
            
            # Check sample size
            for variant_id, variant_data in analysis['variants'].items():
                sample_size = variant_data['sample_size']
                if sample_size < test.target_sample_size:
                    recommendations.append(
                        f"Variant {variant_data['name']} has insufficient sample size "
                        f"({sample_size} < {test.target_sample_size}). Continue test."
                    )
            
            # Check for winners
            for variant_id, tests in analysis.get('statistical_tests', {}).items():
                variant_name = analysis['variants'][variant_id]['name']
                
                significant_improvements = []
                for metric_name, test_result in tests.items():
                    if test_result['significant'] and test_result['effect_size'] > 0:
                        significant_improvements.append(metric_name)
                
                if significant_improvements:
                    recommendations.append(
                        f"Variant {variant_name} shows significant improvement in: "
                        f"{', '.join(significant_improvements)}"
                    )
            
            # Overall recommendation
            if not any('significant improvement' in r for r in recommendations):
                recommendations.append(
                    "No significant improvements detected. Consider keeping control variant."
                )
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Error generating recommendations")
        
        return recommendations
    
    async def export_test_results(self, test_id: str, format: str = 'csv') -> str:
        """Export test results to file"""
        try:
            # Get test results
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query('''
                SELECT tr.*, at.name as test_name
                FROM test_results tr
                JOIN ab_tests at ON tr.test_id = at.id
                WHERE tr.test_id = ?
                ORDER BY tr.timestamp
                ''', conn, params=(test_id,))
            
            if df.empty:
                raise ValueError("No results found for this test")
            
            # Expand metrics column
            metrics_df = pd.json_normalize(df['metrics'].apply(json.loads))
            result_df = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)
            
            # Export to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ab_test_{test_id}_{timestamp}.{format}"
            filepath = self.results_path / filename
            
            if format == 'csv':
                result_df.to_csv(filepath, index=False)
            elif format == 'json':
                result_df.to_json(filepath, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported test results to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to export test results: {e}")
            raise
    
    async def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all A/B tests"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Test counts by status
                cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM ab_tests
                GROUP BY status
                ''')
                status_counts = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Recent test activity
                cursor.execute('''
                SELECT id, name, status, created_at
                FROM ab_tests
                ORDER BY created_at DESC
                LIMIT 10
                ''')
                recent_tests = [
                    {
                        'id': row[0],
                        'name': row[1],
                        'status': row[2],
                        'created_at': row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Results summary
                cursor.execute('''
                SELECT COUNT(*) as total_results,
                       COUNT(DISTINCT test_id) as tests_with_results,
                       COUNT(DISTINCT user_id) as unique_users
                FROM test_results
                ''')
                result_stats = cursor.fetchone()
                
                return {
                    'status_counts': status_counts,
                    'recent_tests': recent_tests,
                    'active_tests_count': len(self.active_tests),
                    'total_results': result_stats[0] if result_stats else 0,
                    'tests_with_results': result_stats[1] if result_stats else 0,
                    'unique_users': result_stats[2] if result_stats else 0,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get test summary: {e}")
            return {}

# Export classes
__all__ = ['ABTestManager', 'ABTest', 'TestVariant', 'TestMetric', 'TestResult', 'TestStatus', 'TestType']