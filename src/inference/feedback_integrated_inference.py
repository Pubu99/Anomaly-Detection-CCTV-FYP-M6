"""
Real-time Feedback Integration
=============================

Integrates the enhanced continuous learning system with the real-time inference engine
to collect feedback and improve the model automatically.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import cv2
import numpy as np

from src.inference.enhanced_real_time_inference import EnhancedRealTimeInference, AlertInfo
from src.continuous_learning.enhanced_continuous_learning import (
    ContinuousLearningManager, FeedbackType, UserRole
)
from src.utils.logging_config import get_app_logger


class FeedbackIntegratedInference(EnhancedRealTimeInference):
    """
    Enhanced real-time inference with integrated feedback collection
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
        # Initialize continuous learning manager
        self.cl_manager = ContinuousLearningManager(config_path)
        self.logger = get_app_logger()
        
        # Feedback collection settings
        self.save_video_segments = True
        self.video_segment_duration = 5.0  # seconds
        self.segments_dir = Path('alerts/video_segments')
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Alert tracking for feedback
        self.active_alerts = {}  # alert_id -> alert_info
        self.alert_video_buffers = {}  # camera_id -> video_buffer
        
        self.logger.info("Feedback-integrated inference engine initialized")
    
    def process_alert(self, alert: AlertInfo):
        """
        Enhanced alert processing with video segment saving for feedback
        """
        # Call parent method for standard alert processing
        super().process_alert(alert)
        
        # Save video segment for potential feedback
        if self.save_video_segments and alert.frame is not None:
            video_segment_path = self._save_video_segment(alert)
            alert.video_segment_path = video_segment_path
        
        # Store alert for feedback tracking
        alert_id = f"{alert.camera_id}_{int(alert.timestamp)}"
        self.active_alerts[alert_id] = alert
        
        # Log alert for feedback system
        self.logger.info(f"Alert ready for feedback: {alert_id}")
    
    def _save_video_segment(self, alert: AlertInfo) -> Optional[str]:
        """
        Save video segment around the alert for feedback purposes
        """
        try:
            # Generate unique filename
            timestamp_str = datetime.fromtimestamp(alert.timestamp).strftime('%Y%m%d_%H%M%S')
            filename = f"{alert.camera_id}_{timestamp_str}_{alert.anomaly_type}.mp4"
            video_path = self.segments_dir / filename
            
            # Get video buffer for this camera
            video_buffer = self.alert_video_buffers.get(alert.camera_id, [])
            
            if not video_buffer:
                # Save single frame as image if no video buffer
                image_path = self.segments_dir / filename.replace('.mp4', '.jpg')
                cv2.imwrite(str(image_path), alert.frame)
                return str(image_path)
            
            # Save video segment
            height, width = alert.frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (width, height))
            
            for frame in video_buffer:
                out.write(frame)
            
            out.release()
            
            self.logger.info(f"Video segment saved: {video_path}")
            return str(video_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save video segment: {e}")
            return None
    
    def add_user_feedback(
        self,
        alert_id: str,
        user_id: str,
        user_role: str,
        feedback_type: str,
        corrected_label: Optional[str] = None,
        corrected_severity: Optional[str] = None,
        confidence_level: float = 1.0,
        notes: Optional[str] = None
    ) -> bool:
        """
        Add user feedback for a specific alert
        """
        try:
            # Get alert information
            alert = self.active_alerts.get(alert_id)
            if not alert:
                self.logger.warning(f"Alert not found for feedback: {alert_id}")
                return False
            
            # Add feedback to continuous learning system
            feedback_id = self.cl_manager.add_feedback(
                user_id=user_id,
                user_role=UserRole(user_role),
                camera_id=alert.camera_id,
                original_prediction=alert.anomaly_type,
                original_confidence=alert.confidence,
                original_severity=alert.severity,
                feedback_type=FeedbackType(feedback_type),
                corrected_label=corrected_label,
                corrected_severity=corrected_severity,
                video_segment_path=getattr(alert, 'video_segment_path', None),
                confidence_level=confidence_level,
                notes=notes
            )
            
            if feedback_id:
                self.logger.info(f"Feedback added for alert {alert_id}: {feedback_id}")
                
                # Remove alert from active alerts (feedback processed)
                self.active_alerts.pop(alert_id, None)
                
                return True
            else:
                self.logger.error(f"Failed to add feedback for alert {alert_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding feedback for alert {alert_id}: {e}")
            return False
    
    def get_pending_alerts_for_feedback(self) -> List[Dict[str, Any]]:
        """
        Get alerts that are pending user feedback
        """
        pending_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            alert_info = {
                'alert_id': alert_id,
                'timestamp': alert.timestamp,
                'camera_id': alert.camera_id,
                'anomaly_type': alert.anomaly_type,
                'confidence': alert.confidence,
                'severity': alert.severity,
                'description': alert.description,
                'has_video_segment': hasattr(alert, 'video_segment_path') and alert.video_segment_path
            }
            pending_alerts.append(alert_info)
        
        return pending_alerts
    
    def get_feedback_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for feedback dashboard
        """
        try:
            # Get statistics from continuous learning manager
            cl_stats = self.cl_manager.get_feedback_statistics()
            
            # Get performance history
            performance_history = self.cl_manager.get_model_performance_history()
            
            # Get pending alerts
            pending_alerts = self.get_pending_alerts_for_feedback()
            
            # System status
            system_status = self.get_system_status()
            
            dashboard_data = {
                'continuous_learning_stats': cl_stats,
                'performance_history': performance_history,
                'pending_alerts': pending_alerts,
                'system_status': system_status,
                'active_alerts_count': len(self.active_alerts),
                'current_time': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}


# Web interface components (for integration with React frontend)
def create_feedback_web_components():
    """
    Create web components for feedback collection
    """
    
    # React component template for feedback form
    feedback_form_component = """
    import React, { useState, useEffect } from 'react';
    import { Button, Card, Form, Select, Input, Rate, message } from 'antd';
    
    const FeedbackForm = ({ alert, onSubmit, onCancel }) => {
      const [form] = Form.useForm();
      const [loading, setLoading] = useState(false);
    
      const feedbackTypes = [
        { value: 'true_positive', label: 'Correct Detection' },
        { value: 'false_positive', label: 'False Alarm (Not an Anomaly)' },
        { value: 'wrong_classification', label: 'Wrong Type of Anomaly' },
        { value: 'severity_correction', label: 'Wrong Severity Level' }
      ];
    
      const anomalyTypes = [
        'Normal', 'Abuse', 'Assault', 'Burglary', 'Fighting', 
        'Robbery', 'Shooting', 'Vandalism', 'Explosion'
      ];
    
      const severityLevels = ['low', 'medium', 'high', 'critical'];
    
      const handleSubmit = async (values) => {
        setLoading(true);
        try {
          await onSubmit({
            alert_id: alert.alert_id,
            ...values
          });
          message.success('Feedback submitted successfully');
          form.resetFields();
          onCancel();
        } catch (error) {
          message.error('Failed to submit feedback');
        } finally {
          setLoading(false);
        }
      };
    
      return (
        <Card title={`Provide Feedback for Alert: ${alert.anomaly_type}`}>
          <div style={{ marginBottom: 16 }}>
            <strong>Original Detection:</strong> {alert.anomaly_type} 
            (Confidence: {(alert.confidence * 100).toFixed(1)}%, 
             Severity: {alert.severity})
          </div>
          
          <Form form={form} onFinish={handleSubmit} layout="vertical">
            <Form.Item
              name="feedback_type"
              label="Feedback Type"
              rules={[{ required: true, message: 'Please select feedback type' }]}
            >
              <Select options={feedbackTypes} placeholder="Select feedback type" />
            </Form.Item>
    
            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) =>
                prevValues.feedback_type !== currentValues.feedback_type
              }
            >
              {({ getFieldValue }) => {
                const feedbackType = getFieldValue('feedback_type');
                return feedbackType === 'wrong_classification' ? (
                  <Form.Item
                    name="corrected_label"
                    label="Correct Anomaly Type"
                    rules={[{ required: true, message: 'Please select correct type' }]}
                  >
                    <Select options={anomalyTypes.map(type => ({ value: type, label: type }))} />
                  </Form.Item>
                ) : null;
              }}
            </Form.Item>
    
            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) =>
                prevValues.feedback_type !== currentValues.feedback_type
              }
            >
              {({ getFieldValue }) => {
                const feedbackType = getFieldValue('feedback_type');
                return feedbackType === 'severity_correction' ? (
                  <Form.Item
                    name="corrected_severity"
                    label="Correct Severity Level"
                    rules={[{ required: true, message: 'Please select correct severity' }]}
                  >
                    <Select options={severityLevels.map(level => ({ value: level, label: level }))} />
                  </Form.Item>
                ) : null;
              }}
            </Form.Item>
    
            <Form.Item
              name="confidence_level"
              label="Your Confidence in This Feedback"
              initialValue={1.0}
            >
              <Rate count={5} allowHalf />
            </Form.Item>
    
            <Form.Item name="notes" label="Additional Notes (Optional)">
              <Input.TextArea rows={3} placeholder="Any additional comments..." />
            </Form.Item>
    
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading} style={{ marginRight: 8 }}>
                Submit Feedback
              </Button>
              <Button onClick={onCancel}>
                Cancel
              </Button>
            </Form.Item>
          </Form>
        </Card>
      );
    };
    
    export default FeedbackForm;
    """
    
    # Dashboard component for monitoring feedback and continuous learning
    feedback_dashboard_component = """
    import React, { useState, useEffect } from 'react';
    import { Card, Row, Col, Statistic, Table, Progress, Button, message } from 'antd';
    import { Line } from '@ant-design/plots';
    
    const FeedbackDashboard = () => {
      const [dashboardData, setDashboardData] = useState(null);
      const [loading, setLoading] = useState(true);
      const [retraining, setRetraining] = useState(false);
    
      useEffect(() => {
        fetchDashboardData();
        const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
        return () => clearInterval(interval);
      }, []);
    
      const fetchDashboardData = async () => {
        try {
          const response = await fetch('/api/feedback/dashboard');
          const data = await response.json();
          setDashboardData(data);
        } catch (error) {
          message.error('Failed to fetch dashboard data');
        } finally {
          setLoading(false);
        }
      };
    
      const triggerRetraining = async () => {
        setRetraining(true);
        try {
          await fetch('/api/feedback/retrain', { method: 'POST' });
          message.success('Model retraining triggered successfully');
          fetchDashboardData();
        } catch (error) {
          message.error('Failed to trigger retraining');
        } finally {
          setRetraining(false);
        }
      };
    
      if (loading) return <div>Loading...</div>;
      if (!dashboardData) return <div>No data available</div>;
    
      const { continuous_learning_stats, performance_history, pending_alerts } = dashboardData;
    
      const performanceData = performance_history.map(record => ({
        date: record.evaluation_date,
        accuracy: record.accuracy,
        f1_score: record.f1_score
      }));
    
      return (
        <div>
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Total Feedback"
                  value={continuous_learning_stats.total_feedback}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Pending Feedback"
                  value={continuous_learning_stats.pending_feedback}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Feedback Quality"
                  value={continuous_learning_stats.average_quality}
                  precision={2}
                  suffix="/ 1.0"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Model Version"
                  value={continuous_learning_stats.current_model_version}
                />
              </Card>
            </Col>
          </Row>
    
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col span={12}>
              <Card title="Performance History">
                <Line
                  data={performanceData}
                  xField="date"
                  yField="accuracy"
                  seriesField="metric"
                  height={300}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card 
                title="Model Retraining" 
                extra={
                  <Button 
                    type="primary" 
                    onClick={triggerRetraining} 
                    loading={retraining}
                  >
                    Trigger Retraining
                  </Button>
                }
              >
                <div>
                  <p>Pending Feedback: {continuous_learning_stats.pending_feedback}</p>
                  <p>Quality Threshold: 0.7</p>
                  <Progress 
                    percent={continuous_learning_stats.average_quality * 100} 
                    status={continuous_learning_stats.average_quality > 0.7 ? "success" : "normal"}
                  />
                </div>
              </Card>
            </Col>
          </Row>
    
          <Row style={{ marginTop: 16 }}>
            <Col span={24}>
              <Card title="Pending Alerts for Feedback">
                <Table
                  dataSource={pending_alerts}
                  columns={[
                    { title: 'Alert ID', dataIndex: 'alert_id', key: 'alert_id' },
                    { title: 'Camera', dataIndex: 'camera_id', key: 'camera_id' },
                    { title: 'Type', dataIndex: 'anomaly_type', key: 'anomaly_type' },
                    { title: 'Confidence', dataIndex: 'confidence', key: 'confidence', 
                      render: (value) => `${(value * 100).toFixed(1)}%` },
                    { title: 'Severity', dataIndex: 'severity', key: 'severity' },
                    { title: 'Timestamp', dataIndex: 'timestamp', key: 'timestamp',
                      render: (value) => new Date(value * 1000).toLocaleString() }
                  ]}
                  pagination={false}
                  scroll={{ y: 300 }}
                />
              </Card>
            </Col>
          </Row>
        </div>
      );
    };
    
    export default FeedbackDashboard;
    """
    
    return {
        'feedback_form': feedback_form_component,
        'feedback_dashboard': feedback_dashboard_component
    }


# API integration for the backend
def integrate_with_backend():
    """
    Integration code for the FastAPI backend
    """
    
    api_routes = """
    # Add to backend/main.py
    
    from src.continuous_learning.enhanced_continuous_learning import create_feedback_api_routes
    from src.inference.feedback_integrated_inference import FeedbackIntegratedInference
    
    # Initialize feedback-integrated inference engine
    feedback_inference = FeedbackIntegratedInference()
    
    # Add feedback API routes
    app.include_router(create_feedback_api_routes())
    
    @app.get("/api/feedback/dashboard")
    async def get_feedback_dashboard():
        '''Get comprehensive feedback dashboard data'''
        return feedback_inference.get_feedback_dashboard_data()
    
    @app.post("/api/alerts/{alert_id}/feedback")
    async def submit_alert_feedback(
        alert_id: str,
        feedback_data: dict,
        current_user: dict = Depends(get_current_user)
    ):
        '''Submit feedback for a specific alert'''
        success = feedback_inference.add_user_feedback(
            alert_id=alert_id,
            user_id=current_user['user_id'],
            user_role=current_user['role'],
            feedback_type=feedback_data['feedback_type'],
            corrected_label=feedback_data.get('corrected_label'),
            corrected_severity=feedback_data.get('corrected_severity'),
            confidence_level=feedback_data.get('confidence_level', 1.0),
            notes=feedback_data.get('notes')
        )
        
        if success:
            return {"status": "success", "message": "Feedback submitted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to submit feedback")
    """
    
    return api_routes


if __name__ == "__main__":
    # Test feedback integrated inference
    feedback_inference = FeedbackIntegratedInference()
    
    print("Feedback-integrated inference engine initialized")
    print("Dashboard data:", feedback_inference.get_feedback_dashboard_data())