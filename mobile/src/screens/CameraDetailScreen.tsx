/**
 * Camera Detail Screen
 * ===================
 *
 * Detailed view of a specific camera with live feed and controls.
 */

import React, { useState, useEffect } from "react";
import { View, ScrollView, StyleSheet, Alert, Dimensions } from "react-native";
import {
  Card,
  Title,
  Paragraph,
  Button,
  Chip,
  Surface,
  Text,
  Divider,
  List,
  IconButton,
  FAB,
  Switch,
  Slider,
  ProgressBar,
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";

const { width: screenWidth } = Dimensions.get("window");

const CameraDetailScreen: React.FC = () => {
  // Mock camera data - in real app, this would come from navigation params or API
  const [camera, setCamera] = useState({
    id: "1",
    name: "Main Entrance",
    location: "Building A - Lobby",
    status: "active",
    ip_address: "192.168.1.100",
    stream_url: "rtsp://192.168.1.100:554/stream1",
    last_seen: "2024-01-15T14:30:00Z",
    alerts_count: 5,
    confidence_threshold: 0.8,
    recording: true,
    motion_detection: true,
    night_vision: false,
    resolution: "1920x1080",
    fps: 30,
    storage_usage: 0.65,
  });

  const [isStreaming, setIsStreaming] = useState(false);
  const [streamQuality, setStreamQuality] = useState("high");

  useEffect(() => {
    // Simulate stream connection
    setIsStreaming(true);
  }, []);

  const handleToggleRecording = () => {
    Alert.alert(
      "Toggle Recording",
      `${camera.recording ? "Stop" : "Start"} recording for this camera?`,
      [
        { text: "Cancel", style: "cancel" },
        {
          text: camera.recording ? "Stop Recording" : "Start Recording",
          onPress: () => {
            setCamera((prev) => ({ ...prev, recording: !prev.recording }));
            console.log("Recording toggled");
          },
        },
      ]
    );
  };

  const handleToggleMotionDetection = () => {
    setCamera((prev) => ({
      ...prev,
      motion_detection: !prev.motion_detection,
    }));
  };

  const handleToggleNightVision = () => {
    setCamera((prev) => ({ ...prev, night_vision: !prev.night_vision }));
  };

  const handleConfidenceThresholdChange = (value: number) => {
    setCamera((prev) => ({ ...prev, confidence_threshold: value }));
  };

  const handleTakeSnapshot = () => {
    Alert.alert("Snapshot", "Snapshot captured successfully");
  };

  const handleViewAlerts = () => {
    console.log("Navigate to camera alerts");
    Alert.alert("Camera Alerts", "Showing alerts for this camera...");
  };

  const handleManageStorage = () => {
    Alert.alert("Storage Management", "Storage management coming soon!");
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active":
        return "#4caf50";
      case "inactive":
        return "#757575";
      case "maintenance":
        return "#ff9800";
      case "error":
        return "#f44336";
      default:
        return "#757575";
    }
  };

  const formatLastSeen = (lastSeen: string) => {
    const date = new Date(lastSeen);
    const now = new Date();
    const diffInMinutes = (now.getTime() - date.getTime()) / (1000 * 60);

    if (diffInMinutes < 1) {
      return "Just now";
    } else if (diffInMinutes < 60) {
      return `${Math.floor(diffInMinutes)}m ago`;
    } else {
      return date.toLocaleString();
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
      >
        {/* Camera Header */}
        <Card style={styles.headerCard}>
          <Card.Content>
            <View style={styles.header}>
              <View style={styles.headerInfo}>
                <Title style={styles.cameraTitle}>{camera.name}</Title>
                <Text style={styles.locationInfo}>📍 {camera.location}</Text>
                <Text style={styles.lastSeen}>
                  Last seen: {formatLastSeen(camera.last_seen)}
                </Text>
              </View>
              <Chip
                style={[
                  styles.statusChip,
                  { backgroundColor: getStatusColor(camera.status) },
                ]}
                textStyle={{ color: "#ffffff" }}
              >
                {camera.status.toUpperCase()}
              </Chip>
            </View>
          </Card.Content>
        </Card>

        {/* Live Stream */}
        <Card style={styles.streamCard}>
          <Card.Content>
            <View style={styles.streamHeader}>
              <Title>Live Stream</Title>
              <View style={styles.streamControls}>
                <Chip
                  icon={isStreaming ? "wifi" : "wifi-off"}
                  style={[
                    styles.streamStatus,
                    {
                      backgroundColor: isStreaming ? "#4caf50" : "#f44336",
                    },
                  ]}
                  textStyle={{ color: "#ffffff" }}
                >
                  {isStreaming ? "LIVE" : "OFFLINE"}
                </Chip>
              </View>
            </View>

            {/* Stream Placeholder */}
            <View style={styles.streamContainer}>
              {isStreaming ? (
                <View style={styles.liveStream}>
                  <Icon name="videocam" size={64} color="#ffffff" />
                  <Text style={styles.streamText}>Live Video Feed</Text>
                  <Text style={styles.streamInfo}>
                    {camera.resolution} • {camera.fps} FPS
                  </Text>
                </View>
              ) : (
                <View style={styles.offlineStream}>
                  <Icon name="videocam-off" size={64} color="#757575" />
                  <Text style={styles.streamText}>Camera Offline</Text>
                </View>
              )}
            </View>

            {/* Stream Actions */}
            <View style={styles.streamActions}>
              <Button
                mode="outlined"
                icon="camera"
                onPress={handleTakeSnapshot}
                style={styles.streamButton}
              >
                Snapshot
              </Button>
              <Button
                mode="outlined"
                icon={camera.recording ? "stop" : "fiber-manual-record"}
                onPress={handleToggleRecording}
                style={styles.streamButton}
              >
                {camera.recording ? "Stop" : "Record"}
              </Button>
              <Button
                mode="outlined"
                icon="fullscreen"
                onPress={() => Alert.alert("Fullscreen", "Fullscreen mode")}
                style={styles.streamButton}
              >
                Fullscreen
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Camera Settings */}
        <Card style={styles.settingsCard}>
          <Card.Content>
            <Title>Camera Settings</Title>

            <List.Item
              title="Motion Detection"
              description="Detect movement in camera view"
              left={() => <List.Icon icon="directions-run" />}
              right={() => (
                <Switch
                  value={camera.motion_detection}
                  onValueChange={handleToggleMotionDetection}
                />
              )}
            />

            <List.Item
              title="Night Vision"
              description="Enable infrared night vision"
              left={() => <List.Icon icon="brightness-3" />}
              right={() => (
                <Switch
                  value={camera.night_vision}
                  onValueChange={handleToggleNightVision}
                />
              )}
            />

            <Divider style={styles.divider} />

            <View style={styles.sliderContainer}>
              <Text style={styles.sliderLabel}>
                Detection Confidence:{" "}
                {(camera.confidence_threshold * 100).toFixed(0)}%
              </Text>
              <Slider
                style={styles.slider}
                minimumValue={0.1}
                maximumValue={1.0}
                value={camera.confidence_threshold}
                onValueChange={handleConfidenceThresholdChange}
                step={0.05}
              />
              <Text style={styles.sliderDescription}>
                Higher values reduce false positives but may miss some events
              </Text>
            </View>
          </Card.Content>
        </Card>

        {/* Camera Stats */}
        <Card style={styles.statsCard}>
          <Card.Content>
            <Title>Statistics</Title>

            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Total Alerts:</Text>
              <Text style={styles.statValue}>{camera.alerts_count}</Text>
            </View>

            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Resolution:</Text>
              <Text style={styles.statValue}>{camera.resolution}</Text>
            </View>

            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Frame Rate:</Text>
              <Text style={styles.statValue}>{camera.fps} FPS</Text>
            </View>

            <View style={styles.statRow}>
              <Text style={styles.statLabel}>IP Address:</Text>
              <Text style={styles.statValue}>{camera.ip_address}</Text>
            </View>

            <Divider style={styles.divider} />

            <Text style={styles.storageLabel}>
              Storage Usage: {(camera.storage_usage * 100).toFixed(0)}%
            </Text>
            <ProgressBar
              progress={camera.storage_usage}
              color={camera.storage_usage > 0.8 ? "#f44336" : "#4caf50"}
              style={styles.storageBar}
            />

            <Button
              mode="outlined"
              icon="storage"
              onPress={handleManageStorage}
              style={styles.storageButton}
            >
              Manage Storage
            </Button>
          </Card.Content>
        </Card>

        {/* Quick Actions */}
        <Card style={styles.actionsCard}>
          <Card.Content>
            <Title>Quick Actions</Title>

            <List.Item
              title="View Alerts"
              description={`${camera.alerts_count} alerts from this camera`}
              left={() => <List.Icon icon="warning" />}
              right={() => (
                <IconButton icon="arrow-forward" onPress={handleViewAlerts} />
              )}
              onPress={handleViewAlerts}
            />

            <List.Item
              title="Camera Configuration"
              description="Advanced camera settings"
              left={() => <List.Icon icon="settings" />}
              right={() => (
                <IconButton
                  icon="arrow-forward"
                  onPress={() => Alert.alert("Config", "Configuration panel")}
                />
              )}
              onPress={() =>
                Alert.alert("Configuration", "Camera configuration panel")
              }
            />

            <List.Item
              title="Download Recordings"
              description="Access recorded footage"
              left={() => <List.Icon icon="download" />}
              right={() => (
                <IconButton
                  icon="arrow-forward"
                  onPress={() => Alert.alert("Download", "Download recordings")}
                />
              )}
              onPress={() =>
                Alert.alert("Recordings", "Download recordings panel")
              }
            />
          </Card.Content>
        </Card>

        <View style={{ height: 100 }} />
      </ScrollView>

      {/* Emergency FAB */}
      <FAB
        style={styles.fab}
        icon="phone"
        onPress={() =>
          Alert.alert("Emergency", "Calling emergency services...")
        }
        color="#ffffff"
      />
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
  headerCard: {
    margin: 16,
    marginBottom: 8,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
  },
  headerInfo: {
    flex: 1,
    marginRight: 16,
  },
  cameraTitle: {
    fontSize: 20,
    marginBottom: 4,
  },
  locationInfo: {
    fontSize: 14,
    opacity: 0.7,
    marginBottom: 2,
  },
  lastSeen: {
    fontSize: 12,
    color: "#1976d2",
  },
  statusChip: {},
  streamCard: {
    margin: 16,
    marginVertical: 4,
  },
  streamHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
  },
  streamControls: {
    flexDirection: "row",
    alignItems: "center",
  },
  streamStatus: {
    height: 24,
  },
  streamContainer: {
    height: 200,
    borderRadius: 8,
    overflow: "hidden",
    marginBottom: 16,
  },
  liveStream: {
    flex: 1,
    backgroundColor: "#1976d2",
    justifyContent: "center",
    alignItems: "center",
  },
  offlineStream: {
    flex: 1,
    backgroundColor: "#757575",
    justifyContent: "center",
    alignItems: "center",
  },
  streamText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "bold",
    marginTop: 8,
  },
  streamInfo: {
    color: "#ffffff",
    fontSize: 12,
    marginTop: 4,
    opacity: 0.8,
  },
  streamActions: {
    flexDirection: "row",
    gap: 8,
  },
  streamButton: {
    flex: 1,
  },
  settingsCard: {
    margin: 16,
    marginVertical: 4,
  },
  divider: {
    marginVertical: 16,
  },
  sliderContainer: {
    marginVertical: 8,
  },
  sliderLabel: {
    fontSize: 16,
    fontWeight: "500",
    marginBottom: 8,
  },
  slider: {
    height: 40,
  },
  sliderDescription: {
    fontSize: 12,
    opacity: 0.7,
    marginTop: 4,
  },
  statsCard: {
    margin: 16,
    marginVertical: 4,
  },
  statRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 4,
  },
  statLabel: {
    fontWeight: "500",
  },
  statValue: {
    color: "#1976d2",
  },
  storageLabel: {
    fontSize: 16,
    fontWeight: "500",
    marginBottom: 8,
  },
  storageBar: {
    height: 8,
    borderRadius: 4,
    marginBottom: 16,
  },
  storageButton: {
    alignSelf: "flex-start",
  },
  actionsCard: {
    margin: 16,
    marginVertical: 4,
  },
  fab: {
    position: "absolute",
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: "#f44336",
  },
});

export default CameraDetailScreen;
