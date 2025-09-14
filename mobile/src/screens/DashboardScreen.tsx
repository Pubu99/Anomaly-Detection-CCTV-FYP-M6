/**
 * Dashboard Screen
 * ===============
 *
 * Main dashboard screen showing system overview, active alerts, and camera status.
 */

import React, { useEffect, useRef } from "react";
import {
  View,
  ScrollView,
  RefreshControl,
  Alert,
  StyleSheet,
  Dimensions,
} from "react-native";
import {
  Card,
  Title,
  Paragraph,
  Avatar,
  Badge,
  FAB,
  Portal,
  Modal,
  Button,
  Divider,
  List,
  Surface,
  Text,
  IconButton,
  ProgressBar,
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";
import { LineChart, PieChart } from "react-native-chart-kit";

// Hooks
import { useAppSelector, useAppDispatch } from "../hooks/redux";

// Services
import { websocketService } from "../services/websocketService";
import { notificationService } from "../services/notificationService";

// Types
interface SystemMetrics {
  totalCameras: number;
  activeCameras: number;
  alertsToday: number;
  highRiskAlerts: number;
  averageConfidence: number;
  systemHealth: number;
}

const { width: screenWidth } = Dimensions.get("window");

const DashboardScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isDarkMode } = useAppSelector((state) => state.ui);
  const { alerts } = useAppSelector((state) => state.alerts);
  const { cameras } = useAppSelector((state) => state.cameras);
  const { user } = useAppSelector((state) => state.auth);

  const [refreshing, setRefreshing] = React.useState(false);
  const [metrics, setMetrics] = React.useState<SystemMetrics>({
    totalCameras: 0,
    activeCameras: 0,
    alertsToday: 0,
    highRiskAlerts: 0,
    averageConfidence: 0,
    systemHealth: 0,
  });
  const [showModal, setShowModal] = React.useState(false);
  const [alertTrends, setAlertTrends] = React.useState<number[]>([]);

  const wsRef = useRef<any>(null);

  useEffect(() => {
    initializeWebSocket();
    loadDashboardData();
    requestNotificationPermissions();

    return () => {
      if (wsRef.current) {
        websocketService.disconnect();
      }
    };
  }, []);

  const initializeWebSocket = async () => {
    try {
      wsRef.current = await websocketService.connect("ws://localhost:8000/ws");

      websocketService.onMessage((data) => {
        console.log("WebSocket message:", data);
        // Handle real-time updates
        if (data.type === "alert") {
          showAlertNotification(data.alert);
        }
      });
    } catch (error) {
      console.error("WebSocket connection failed:", error);
    }
  };

  const requestNotificationPermissions = async () => {
    try {
      await notificationService.requestPermission();
    } catch (error) {
      console.error("Notification permission failed:", error);
    }
  };

  const showAlertNotification = (alert: any) => {
    notificationService.showLocalNotification({
      title: "Security Alert",
      body: `${alert.type} detected on ${alert.camera_name}`,
      data: { alertId: alert.id },
    });
  };

  const loadDashboardData = async () => {
    try {
      // Calculate metrics from current state
      const totalCameras = cameras.length;
      const activeCameras = cameras.filter(
        (cam) => cam.status === "active"
      ).length;
      const todayAlerts = alerts.filter((alert) => {
        const today = new Date().toDateString();
        return new Date(alert.timestamp).toDateString() === today;
      });
      const highRiskAlerts = alerts.filter(
        (alert) => alert.severity === "high"
      ).length;

      const confidenceSum = alerts.reduce(
        (sum, alert) => sum + alert.confidence,
        0
      );
      const averageConfidence =
        alerts.length > 0 ? confidenceSum / alerts.length : 0;

      const systemHealth = (activeCameras / Math.max(totalCameras, 1)) * 100;

      setMetrics({
        totalCameras,
        activeCameras,
        alertsToday: todayAlerts.length,
        highRiskAlerts,
        averageConfidence,
        systemHealth,
      });

      // Generate mock trend data
      setAlertTrends([5, 8, 12, 6, 9, 15, 11]);
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  const getSystemHealthColor = (health: number) => {
    if (health >= 80) return "#4caf50";
    if (health >= 60) return "#ff9800";
    return "#f44336";
  };

  const chartConfig = {
    backgroundGradientFrom: isDarkMode ? "#1e1e1e" : "#ffffff",
    backgroundGradientTo: isDarkMode ? "#1e1e1e" : "#ffffff",
    color: (opacity = 1) =>
      isDarkMode
        ? `rgba(144, 202, 249, ${opacity})`
        : `rgba(25, 118, 210, ${opacity})`,
    strokeWidth: 2,
    barPercentage: 0.5,
    useShadowColorFromDataset: false,
  };

  const pieData = [
    {
      name: "Low Risk",
      population: alerts.filter((a) => a.severity === "low").length,
      color: "#4caf50",
      legendFontColor: isDarkMode ? "#ffffff" : "#000000",
      legendFontSize: 12,
    },
    {
      name: "Medium Risk",
      population: alerts.filter((a) => a.severity === "medium").length,
      color: "#ff9800",
      legendFontColor: isDarkMode ? "#ffffff" : "#000000",
      legendFontSize: 12,
    },
    {
      name: "High Risk",
      population: alerts.filter((a) => a.severity === "high").length,
      color: "#f44336",
      legendFontColor: isDarkMode ? "#ffffff" : "#000000",
      legendFontSize: 12,
    },
  ];

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Welcome Section */}
        <Card style={styles.welcomeCard}>
          <Card.Content>
            <View style={styles.welcomeHeader}>
              <View>
                <Title>Welcome back, {user?.username || "User"}</Title>
                <Paragraph>System Status: All systems operational</Paragraph>
              </View>
              <Avatar.Icon size={48} icon="security" />
            </View>
          </Card.Content>
        </Card>

        {/* Metrics Cards */}
        <View style={styles.metricsGrid}>
          <Card style={styles.metricCard}>
            <Card.Content style={styles.metricContent}>
              <View style={styles.metricHeader}>
                <Icon name="videocam" size={24} color="#1976d2" />
                <Text style={styles.metricValue}>{metrics.totalCameras}</Text>
              </View>
              <Text style={styles.metricLabel}>Total Cameras</Text>
              <Text style={styles.metricSubtext}>
                {metrics.activeCameras} active
              </Text>
            </Card.Content>
          </Card>

          <Card style={styles.metricCard}>
            <Card.Content style={styles.metricContent}>
              <View style={styles.metricHeader}>
                <Icon name="warning" size={24} color="#f44336" />
                <Text style={styles.metricValue}>{metrics.alertsToday}</Text>
              </View>
              <Text style={styles.metricLabel}>Alerts Today</Text>
              <Text style={styles.metricSubtext}>
                {metrics.highRiskAlerts} high risk
              </Text>
            </Card.Content>
          </Card>
        </View>

        {/* System Health */}
        <Card style={styles.card}>
          <Card.Content>
            <Title>System Health</Title>
            <View style={styles.healthContainer}>
              <Text style={styles.healthValue}>
                {metrics.systemHealth.toFixed(1)}%
              </Text>
              <ProgressBar
                progress={metrics.systemHealth / 100}
                color={getSystemHealthColor(metrics.systemHealth)}
                style={styles.healthBar}
              />
            </View>
            <Paragraph>
              Average Detection Confidence:{" "}
              {(metrics.averageConfidence * 100).toFixed(1)}%
            </Paragraph>
          </Card.Content>
        </Card>

        {/* Alert Trends Chart */}
        <Card style={styles.card}>
          <Card.Content>
            <Title>Alert Trends (Last 7 Days)</Title>
            <LineChart
              data={{
                labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                datasets: [
                  {
                    data: alertTrends,
                  },
                ],
              }}
              width={screenWidth - 60}
              height={220}
              chartConfig={chartConfig}
              bezier
              style={styles.chart}
            />
          </Card.Content>
        </Card>

        {/* Alert Distribution */}
        <Card style={styles.card}>
          <Card.Content>
            <Title>Alert Distribution by Risk Level</Title>
            <PieChart
              data={pieData}
              width={screenWidth - 60}
              height={220}
              chartConfig={chartConfig}
              accessor="population"
              backgroundColor="transparent"
              paddingLeft="15"
              style={styles.chart}
            />
          </Card.Content>
        </Card>

        {/* Recent Alerts */}
        <Card style={styles.card}>
          <Card.Content>
            <Title>Recent Alerts</Title>
            {alerts.slice(0, 5).map((alert, index) => (
              <View key={alert.id}>
                <List.Item
                  title={alert.event_type}
                  description={`Camera: ${alert.camera_name} • ${new Date(
                    alert.timestamp
                  ).toLocaleTimeString()}`}
                  left={() => (
                    <Avatar.Icon
                      size={40}
                      icon="warning"
                      style={{
                        backgroundColor:
                          alert.severity === "high"
                            ? "#f44336"
                            : alert.severity === "medium"
                            ? "#ff9800"
                            : "#4caf50",
                      }}
                    />
                  )}
                  right={() => (
                    <Badge style={{ alignSelf: "center" }}>
                      {(alert.confidence * 100).toFixed(0)}%
                    </Badge>
                  )}
                />
                {index < 4 && <Divider />}
              </View>
            ))}
          </Card.Content>
        </Card>

        <View style={{ height: 100 }} />
      </ScrollView>

      {/* Emergency FAB */}
      <FAB
        style={styles.fab}
        icon="phone"
        label="Emergency"
        onPress={() => setShowModal(true)}
        color="#ffffff"
        customSize={56}
      />

      {/* Emergency Modal */}
      <Portal>
        <Modal
          visible={showModal}
          onDismiss={() => setShowModal(false)}
          contentContainerStyle={styles.modal}
        >
          <Title>Emergency Contacts</Title>
          <Paragraph>Call emergency services immediately</Paragraph>
          <View style={styles.emergencyButtons}>
            <Button
              mode="contained"
              icon="phone"
              onPress={() => {
                // Handle emergency call
                Alert.alert("Emergency", "Calling emergency services...");
                setShowModal(false);
              }}
              style={styles.emergencyButton}
            >
              Call 911
            </Button>
            <Button
              mode="outlined"
              onPress={() => setShowModal(false)}
              style={styles.emergencyButton}
            >
              Cancel
            </Button>
          </View>
        </Modal>
      </Portal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  scrollView: {
    flex: 1,
  },
  welcomeCard: {
    margin: 16,
    marginBottom: 8,
  },
  welcomeHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  metricsGrid: {
    flexDirection: "row",
    paddingHorizontal: 16,
    gap: 8,
  },
  metricCard: {
    flex: 1,
    marginVertical: 4,
  },
  metricContent: {
    alignItems: "center",
  },
  metricHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 8,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: "bold",
    marginLeft: 8,
  },
  metricLabel: {
    fontSize: 12,
    opacity: 0.7,
  },
  metricSubtext: {
    fontSize: 10,
    opacity: 0.5,
    marginTop: 4,
  },
  card: {
    margin: 16,
    marginVertical: 8,
  },
  healthContainer: {
    marginVertical: 16,
  },
  healthValue: {
    fontSize: 32,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 8,
  },
  healthBar: {
    height: 8,
    borderRadius: 4,
    marginBottom: 16,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  fab: {
    position: "absolute",
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: "#f44336",
  },
  modal: {
    backgroundColor: "white",
    padding: 20,
    margin: 20,
    borderRadius: 8,
  },
  emergencyButtons: {
    flexDirection: "row",
    gap: 8,
    marginTop: 16,
  },
  emergencyButton: {
    flex: 1,
  },
});

export default DashboardScreen;
