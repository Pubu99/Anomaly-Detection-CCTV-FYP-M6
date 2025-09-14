/**
 * Alert Detail Screen
 * ==================
 *
 * Detailed view of a specific security alert.
 */

import React, { useState } from "react";
import {
  View,
  ScrollView,
  StyleSheet,
  Alert,
  Image,
  Dimensions,
} from "react-native";
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
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";

const { width: screenWidth } = Dimensions.get("window");

const AlertDetailScreen: React.FC = () => {
  // Mock alert data - in real app, this would come from navigation params or API
  const [alert] = useState({
    id: "1",
    event_type: "Suspicious Activity",
    camera_name: "Main Entrance",
    camera_location: "Building A - Lobby",
    timestamp: "2024-01-15T14:30:00Z",
    confidence: 0.92,
    severity: "high",
    status: "new",
    description:
      "Person loitering near entrance for extended period. Unusual behavior pattern detected.",
    thumbnail_url: null,
    video_url: null,
    coordinates: { x: 150, y: 200, width: 100, height: 150 },
    metadata: {
      duration: 45,
      object_count: 2,
      motion_intensity: "medium",
      lighting_conditions: "good",
    },
  });

  const handleAcknowledge = () => {
    Alert.alert("Acknowledge Alert", "Mark this alert as acknowledged?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Acknowledge",
        onPress: () => {
          console.log("Alert acknowledged");
          Alert.alert("Success", "Alert acknowledged successfully");
        },
      },
    ]);
  };

  const handleResolve = () => {
    Alert.alert("Resolve Alert", "Mark this alert as resolved?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Resolve",
        onPress: () => {
          console.log("Alert resolved");
          Alert.alert("Success", "Alert resolved successfully");
        },
      },
    ]);
  };

  const handleViewCamera = () => {
    console.log("Navigate to camera view");
    Alert.alert("Camera View", "Opening camera live feed...");
  };

  const handleDownload = () => {
    Alert.alert("Download", "Evidence download started");
  };

  const handleShare = () => {
    Alert.alert("Share", "Sharing alert details...");
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high":
        return "#f44336";
      case "medium":
        return "#ff9800";
      case "low":
        return "#4caf50";
      default:
        return "#757575";
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "new":
        return "#2196f3";
      case "acknowledged":
        return "#ff9800";
      case "resolved":
        return "#4caf50";
      default:
        return "#757575";
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
      >
        {/* Alert Header */}
        <Card style={styles.headerCard}>
          <Card.Content>
            <View style={styles.header}>
              <View style={styles.headerInfo}>
                <Title style={styles.alertTitle}>{alert.event_type}</Title>
                <Text style={styles.cameraInfo}>
                  📹 {alert.camera_name} • 📍 {alert.camera_location}
                </Text>
                <Text style={styles.timestamp}>
                  ⏰ {formatTimestamp(alert.timestamp)}
                </Text>
              </View>
              <View style={styles.chips}>
                <Chip
                  style={[
                    styles.severityChip,
                    { backgroundColor: getSeverityColor(alert.severity) },
                  ]}
                  textStyle={{ color: "#ffffff" }}
                >
                  {alert.severity.toUpperCase()}
                </Chip>
                <Chip
                  style={[
                    styles.statusChip,
                    { backgroundColor: getStatusColor(alert.status) },
                  ]}
                  textStyle={{ color: "#ffffff" }}
                >
                  {alert.status.toUpperCase()}
                </Chip>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Alert Image/Video */}
        <Card style={styles.mediaCard}>
          <Card.Content>
            <Title>Evidence</Title>
            {alert.thumbnail_url ? (
              <Image
                source={{ uri: alert.thumbnail_url }}
                style={styles.alertImage}
                resizeMode="cover"
              />
            ) : (
              <View style={styles.placeholderImage}>
                <Icon name="image" size={64} color="#757575" />
                <Text style={styles.placeholderText}>No image available</Text>
              </View>
            )}

            <View style={styles.mediaActions}>
              <Button
                mode="outlined"
                icon="play-arrow"
                onPress={() => Alert.alert("Video", "Playing video...")}
                style={styles.mediaButton}
              >
                Play Video
              </Button>
              <Button
                mode="outlined"
                icon="download"
                onPress={handleDownload}
                style={styles.mediaButton}
              >
                Download
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Alert Details */}
        <Card style={styles.detailsCard}>
          <Card.Content>
            <Title>Alert Details</Title>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Confidence:</Text>
              <Text style={styles.detailValue}>
                {(alert.confidence * 100).toFixed(1)}%
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Duration:</Text>
              <Text style={styles.detailValue}>
                {alert.metadata.duration} seconds
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Objects Detected:</Text>
              <Text style={styles.detailValue}>
                {alert.metadata.object_count}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Motion Intensity:</Text>
              <Text style={styles.detailValue}>
                {alert.metadata.motion_intensity}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Lighting:</Text>
              <Text style={styles.detailValue}>
                {alert.metadata.lighting_conditions}
              </Text>
            </View>

            <Divider style={styles.divider} />

            <Title>Description</Title>
            <Paragraph style={styles.description}>
              {alert.description}
            </Paragraph>
          </Card.Content>
        </Card>

        {/* Location Info */}
        <Card style={styles.locationCard}>
          <Card.Content>
            <Title>Location Information</Title>
            <List.Item
              title="Camera Location"
              description={alert.camera_location}
              left={() => <List.Icon icon="location-on" />}
            />
            <List.Item
              title="Detection Area"
              description={`X: ${alert.coordinates.x}, Y: ${alert.coordinates.y}`}
              left={() => <List.Icon icon="crop-free" />}
            />
            <List.Item
              title="View Camera"
              description="Open live camera feed"
              left={() => <List.Icon icon="videocam" />}
              right={() => (
                <IconButton icon="arrow-forward" onPress={handleViewCamera} />
              )}
              onPress={handleViewCamera}
            />
          </Card.Content>
        </Card>

        {/* Actions */}
        <Card style={styles.actionsCard}>
          <Card.Content>
            <Title>Actions</Title>
            <View style={styles.actionButtons}>
              {alert.status === "new" && (
                <Button
                  mode="contained"
                  icon="check"
                  onPress={handleAcknowledge}
                  style={styles.actionButton}
                >
                  Acknowledge
                </Button>
              )}
              {alert.status !== "resolved" && (
                <Button
                  mode="contained"
                  icon="check-circle"
                  onPress={handleResolve}
                  style={[styles.actionButton, { backgroundColor: "#4caf50" }]}
                >
                  Resolve
                </Button>
              )}
              <Button
                mode="outlined"
                icon="share"
                onPress={handleShare}
                style={styles.actionButton}
              >
                Share
              </Button>
            </View>
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
  alertTitle: {
    fontSize: 20,
    marginBottom: 4,
  },
  cameraInfo: {
    fontSize: 14,
    opacity: 0.7,
    marginBottom: 2,
  },
  timestamp: {
    fontSize: 12,
    color: "#1976d2",
  },
  chips: {
    alignItems: "flex-end",
    gap: 4,
  },
  severityChip: {
    marginBottom: 4,
  },
  statusChip: {},
  mediaCard: {
    margin: 16,
    marginVertical: 4,
  },
  alertImage: {
    width: "100%",
    height: 200,
    borderRadius: 8,
    marginVertical: 16,
  },
  placeholderImage: {
    height: 200,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f0f0f0",
    borderRadius: 8,
    marginVertical: 16,
  },
  placeholderText: {
    marginTop: 8,
    opacity: 0.7,
  },
  mediaActions: {
    flexDirection: "row",
    gap: 8,
    marginTop: 8,
  },
  mediaButton: {
    flex: 1,
  },
  detailsCard: {
    margin: 16,
    marginVertical: 4,
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 4,
  },
  detailLabel: {
    fontWeight: "500",
  },
  detailValue: {
    color: "#1976d2",
  },
  divider: {
    marginVertical: 16,
  },
  description: {
    marginTop: 8,
    lineHeight: 20,
  },
  locationCard: {
    margin: 16,
    marginVertical: 4,
  },
  actionsCard: {
    margin: 16,
    marginVertical: 4,
  },
  actionButtons: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: 8,
  },
  actionButton: {
    flex: 1,
    minWidth: 120,
  },
  fab: {
    position: "absolute",
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: "#f44336",
  },
});

export default AlertDetailScreen;
