/**
 * Dashboard Component
 * ==================
 *
 * Main dashboard with system overview, real-time stats, camera grid, and recent alerts.
 */

import React, { useEffect, useState } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  IconButton,
  Chip,
  Alert,
  LinearProgress,
  Stack,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  useTheme,
} from "@mui/material";
import {
  PlayArrow,
  Stop,
  Refresh,
  Warning,
  CheckCircle,
  Error,
  Info,
  Videocam,
  Timeline,
  Security,
  Speed,
} from "@mui/icons-material";
import { useSelector, useDispatch } from "react-redux";
import { format } from "date-fns";

import { RootState } from "../../store/store";
import {
  fetchStatsStart,
  fetchStatsSuccess,
  fetchStatsFailure,
  setInferenceRunning,
  fetchCamerasSuccess,
  fetchAlertsSuccess,
} from "../../store/store";

// Import API functions (these would be in a separate api module)
// import { getSystemStats, getCameras, getRecentAlerts, startInference, stopInference } from '../../api/api';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: "primary" | "secondary" | "error" | "warning" | "info" | "success";
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  icon,
  color,
  trend,
}) => {
  const theme = useTheme();

  return (
    <Card sx={{ height: "100%" }}>
      <CardContent>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <Box>
            <Typography color="textSecondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography
              variant="h4"
              component="div"
              sx={{ fontWeight: "bold" }}
            >
              {value}
            </Typography>
            {trend && (
              <Box sx={{ display: "flex", alignItems: "center", mt: 1 }}>
                <Typography
                  variant="body2"
                  color={trend.isPositive ? "success.main" : "error.main"}
                >
                  {trend.isPositive ? "+" : ""}
                  {trend.value}%
                </Typography>
                <Typography
                  variant="body2"
                  color="textSecondary"
                  sx={{ ml: 1 }}
                >
                  vs last hour
                </Typography>
              </Box>
            )}
          </Box>
          <Box
            sx={{
              backgroundColor: theme.palette[color].main + "20",
              borderRadius: 2,
              p: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {React.cloneElement(icon as React.ReactElement, {
              sx: { fontSize: 32, color: theme.palette[color].main },
            })}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const dispatch = useDispatch();

  const { stats, loading, connectionStatus, inferenceRunning } = useSelector(
    (state: RootState) => state.system
  );
  const { cameras } = useSelector((state: RootState) => state.cameras);
  const { alerts } = useSelector((state: RootState) => state.alerts);

  const [refreshing, setRefreshing] = useState(false);

  // Mock data - in real app, this would come from API
  useEffect(() => {
    const mockStats = {
      total_cameras: 12,
      active_cameras: 10,
      total_alerts_24h: 45,
      critical_alerts_24h: 3,
      system_uptime: 99.8,
      average_fps: 28.5,
      model_accuracy: 94.2,
      error_rate_24h: 0.5,
      false_positive_rate: 2.1,
    };

    const mockCameras = Array.from({ length: 12 }, (_, i) => ({
      id: i + 1,
      camera_id: `CAM-${String(i + 1).padStart(3, "0")}`,
      name: `Camera ${i + 1}`,
      location: `Location ${i + 1}`,
      status: Math.random() > 0.2 ? "active" : "inactive",
      position_x: Math.random() * 100,
      position_y: Math.random() * 100,
      weight: Math.random(),
      resolution_width: 1920,
      resolution_height: 1080,
      fps: 30,
      enabled: true,
      created_at: new Date().toISOString(),
      total_frames_processed: Math.floor(Math.random() * 100000),
      error_count: Math.floor(Math.random() * 10),
    }));

    const mockAlerts = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      camera_id: `CAM-${String(Math.floor(Math.random() * 12) + 1).padStart(
        3,
        "0"
      )}`,
      anomaly_class: [
        "fighting",
        "robbery",
        "vandalism",
        "suspicious_activity",
      ][Math.floor(Math.random() * 4)],
      confidence: 0.7 + Math.random() * 0.3,
      severity: ["low", "medium", "high", "critical"][
        Math.floor(Math.random() * 4)
      ],
      status: ["active", "acknowledged", "resolved"][
        Math.floor(Math.random() * 3)
      ],
      emergency_contact_sent: Math.random() > 0.5,
      created_at: new Date(Date.now() - Math.random() * 86400000).toISOString(),
    }));

    dispatch(fetchStatsSuccess(mockStats));
    dispatch(fetchCamerasSuccess(mockCameras));
    dispatch(
      fetchAlertsSuccess({ alerts: mockAlerts, totalCount: mockAlerts.length })
    );
  }, [dispatch]);

  const handleRefresh = async () => {
    setRefreshing(true);
    // In real app, fetch fresh data from API
    setTimeout(() => setRefreshing(false), 1000);
  };

  const handleInferenceToggle = async () => {
    // In real app, call API to start/stop inference
    dispatch(setInferenceRunning(!inferenceRunning));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "success";
      case "acknowledged":
        return "warning";
      case "resolved":
        return "info";
      default:
        return "default";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "low":
        return "info";
      case "medium":
        return "warning";
      case "high":
        return "error";
      case "critical":
        return "error";
      default:
        return "default";
    }
  };

  const recentAlerts = alerts.slice(0, 5);
  const activeCameras = cameras.filter((c) => c.status === "active");

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header Actions */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 3,
        }}
      >
        <Typography variant="h4" component="h1" sx={{ fontWeight: "bold" }}>
          System Dashboard
        </Typography>
        <Stack direction="row" spacing={2}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={inferenceRunning ? <Stop /> : <PlayArrow />}
            onClick={handleInferenceToggle}
            color={inferenceRunning ? "error" : "success"}
          >
            {inferenceRunning ? "Stop Detection" : "Start Detection"}
          </Button>
        </Stack>
      </Box>

      {/* Connection Status Alert */}
      {connectionStatus !== "connected" && (
        <Alert
          severity={connectionStatus === "connecting" ? "warning" : "error"}
          sx={{ mb: 3 }}
        >
          {connectionStatus === "connecting"
            ? "Connecting to detection system..."
            : "Connection to detection system lost. Some features may not be available."}
        </Alert>
      )}

      {/* Loading Bar */}
      {(loading || refreshing) && <LinearProgress sx={{ mb: 3 }} />}

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Cameras"
            value={stats?.total_cameras || 0}
            icon={<Videocam />}
            color="primary"
            trend={{ value: 8.2, isPositive: true }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Active Cameras"
            value={stats?.active_cameras || 0}
            icon={<CheckCircle />}
            color="success"
            trend={{ value: 2.1, isPositive: true }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Alerts (24h)"
            value={stats?.total_alerts_24h || 0}
            icon={<Warning />}
            color="warning"
            trend={{ value: -12.3, isPositive: false }}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Model Accuracy"
            value={`${stats?.model_accuracy?.toFixed(1) || 0}%`}
            icon={<Timeline />}
            color="info"
            trend={{ value: 1.4, isPositive: true }}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* System Performance */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  mb: 2,
                }}
              >
                <Typography variant="h6" component="h2">
                  System Performance
                </Typography>
                <Chip
                  icon={<Speed />}
                  label={`${stats?.average_fps?.toFixed(1) || 0} FPS`}
                  color="primary"
                  variant="outlined"
                />
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        System Uptime
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                        {stats?.system_uptime?.toFixed(1) || 0}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={stats?.system_uptime || 0}
                      color="success"
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Error Rate (24h)
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                        {stats?.error_rate_24h?.toFixed(1) || 0}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={stats?.error_rate_24h || 0}
                      color="error"
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        False Positive Rate
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                        {stats?.false_positive_rate?.toFixed(1) || 0}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={stats?.false_positive_rate || 0}
                      color="warning"
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={6}>
                  <Box sx={{ mb: 2 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        mb: 1,
                      }}
                    >
                      <Typography variant="body2" color="textSecondary">
                        Model Accuracy
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                        {stats?.model_accuracy?.toFixed(1) || 0}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={stats?.model_accuracy || 0}
                      color="primary"
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
                Recent Alerts
              </Typography>

              <Stack spacing={2}>
                {recentAlerts.map((alert) => (
                  <Paper
                    key={alert.id}
                    sx={{
                      p: 2,
                      border: `1px solid ${theme.palette.divider}`,
                      borderLeft: `4px solid ${
                        theme.palette[
                          getSeverityColor(
                            alert.severity
                          ) as keyof typeof theme.palette
                        ].main
                      }`,
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        mb: 1,
                      }}
                    >
                      <Typography
                        variant="subtitle2"
                        sx={{ fontWeight: "bold" }}
                      >
                        {alert.anomaly_class.replace("_", " ").toUpperCase()}
                      </Typography>
                      <Chip
                        label={alert.severity}
                        size="small"
                        color={getSeverityColor(alert.severity) as any}
                        variant="outlined"
                      />
                    </Box>

                    <Typography
                      variant="body2"
                      color="textSecondary"
                      sx={{ mb: 1 }}
                    >
                      Camera: {alert.camera_id}
                    </Typography>

                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography variant="caption" color="textSecondary">
                        {format(new Date(alert.created_at), "MMM dd, HH:mm")}
                      </Typography>
                      <Chip
                        label={alert.status}
                        size="small"
                        color={getStatusColor(alert.status) as any}
                        variant="filled"
                      />
                    </Box>
                  </Paper>
                ))}

                {recentAlerts.length === 0 && (
                  <Box sx={{ textAlign: "center", py: 4 }}>
                    <Security
                      sx={{ fontSize: 48, color: "text.secondary", mb: 2 }}
                    />
                    <Typography color="textSecondary">
                      No recent alerts
                    </Typography>
                  </Box>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
