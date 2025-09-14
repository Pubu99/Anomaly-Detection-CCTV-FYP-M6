/**
 * Cameras Screen
 * =============
 *
 * Screen for viewing and managing security cameras.
 */

import React, { useState, useEffect } from "react";
import {
  View,
  FlatList,
  RefreshControl,
  StyleSheet,
  Alert,
  Dimensions,
} from "react-native";
import {
  Card,
  Title,
  Paragraph,
  Avatar,
  Badge,
  Chip,
  FAB,
  Searchbar,
  Button,
  Text,
  Surface,
  IconButton,
  Menu,
  Divider,
  ProgressBar,
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";

// Hooks
import { useAppSelector, useAppDispatch } from "../hooks/redux";

// Types
interface Camera {
  id: string;
  name: string;
  location: string;
  status: "active" | "inactive" | "maintenance" | "error";
  ip_address: string;
  stream_url?: string;
  last_seen?: string;
  alerts_count: number;
  confidence_threshold: number;
  recording: boolean;
}

const { width: screenWidth } = Dimensions.get("window");
const cardWidth = (screenWidth - 48) / 2; // 2 columns with margins

const CamerasScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { cameras } = useAppSelector((state) => state.cameras);
  const { isDarkMode } = useAppSelector((state) => state.ui);

  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterMenuVisible, setFilterMenuVisible] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  const filteredCameras = cameras.filter((camera: Camera) => {
    const matchesSearch =
      camera.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      camera.location.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesFilter =
      selectedFilter === "all" || camera.status === selectedFilter;

    return matchesSearch && matchesFilter;
  });

  const onRefresh = async () => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => setRefreshing(false), 1000);
  };

  const handleCameraPress = (camera: Camera) => {
    // Navigate to camera detail screen
    console.log("Navigate to camera detail:", camera.id);
  };

  const handleToggleRecording = (cameraId: string) => {
    Alert.alert("Toggle Recording", "Start/stop recording for this camera?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Toggle",
        onPress: () => {
          console.log("Toggle recording for camera:", cameraId);
          // dispatch(toggleCameraRecording(cameraId));
        },
      },
    ]);
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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "active":
        return "check-circle";
      case "inactive":
        return "pause-circle-outline";
      case "maintenance":
        return "build";
      case "error":
        return "error";
      default:
        return "help";
    }
  };

  const formatLastSeen = (lastSeen?: string) => {
    if (!lastSeen) return "Never";

    const date = new Date(lastSeen);
    const now = new Date();
    const diffInMinutes = (now.getTime() - date.getTime()) / (1000 * 60);

    if (diffInMinutes < 1) {
      return "Just now";
    } else if (diffInMinutes < 60) {
      return `${Math.floor(diffInMinutes)}m ago`;
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const renderGridCamera = ({ item }: { item: Camera }) => (
    <Card
      style={[styles.gridCard, { width: cardWidth }]}
      onPress={() => handleCameraPress(item)}
    >
      <Card.Content style={styles.gridCardContent}>
        {/* Camera Header */}
        <View style={styles.gridHeader}>
          <Avatar.Icon
            size={32}
            icon="videocam"
            style={{
              backgroundColor: getStatusColor(item.status),
            }}
          />
          <IconButton
            icon={item.recording ? "stop" : "play-arrow"}
            size={16}
            onPress={() => handleToggleRecording(item.id)}
          />
        </View>

        {/* Camera Info */}
        <Text style={styles.gridCameraName} numberOfLines={1}>
          {item.name}
        </Text>
        <Text style={styles.gridCameraLocation} numberOfLines={1}>
          📍 {item.location}
        </Text>

        {/* Status and Stats */}
        <View style={styles.gridStats}>
          <Chip
            style={[
              styles.statusChip,
              { backgroundColor: getStatusColor(item.status) },
            ]}
            textStyle={{ color: "#ffffff", fontSize: 8 }}
          >
            {item.status.toUpperCase()}
          </Chip>
          {item.alerts_count > 0 && (
            <Badge size={16} style={styles.alertsBadge}>
              {item.alerts_count}
            </Badge>
          )}
        </View>

        {/* Confidence Bar */}
        <View style={styles.confidenceContainer}>
          <Text style={styles.confidenceLabel}>
            Confidence: {(item.confidence_threshold * 100).toFixed(0)}%
          </Text>
          <ProgressBar
            progress={item.confidence_threshold}
            color="#1976d2"
            style={styles.confidenceBar}
          />
        </View>
      </Card.Content>
    </Card>
  );

  const renderListCamera = ({ item }: { item: Camera }) => (
    <Card style={styles.listCard} onPress={() => handleCameraPress(item)}>
      <Card.Content>
        <View style={styles.listHeader}>
          <View style={styles.listCameraInfo}>
            <View style={styles.listTitleRow}>
              <Text style={styles.listCameraName}>{item.name}</Text>
              <Chip
                style={[
                  styles.statusChip,
                  { backgroundColor: getStatusColor(item.status) },
                ]}
                textStyle={{ color: "#ffffff", fontSize: 10 }}
              >
                {item.status.toUpperCase()}
              </Chip>
            </View>
            <Text style={styles.listCameraDetails}>
              📍 {item.location} • 🌐 {item.ip_address}
            </Text>
            <Text style={styles.listCameraLastSeen}>
              Last seen: {formatLastSeen(item.last_seen)}
            </Text>
          </View>
          <Avatar.Icon
            size={40}
            icon="videocam"
            style={{
              backgroundColor: getStatusColor(item.status),
            }}
          />
        </View>

        <View style={styles.listActions}>
          <View style={styles.listStats}>
            <Text style={styles.statText}>Alerts: {item.alerts_count}</Text>
            <Text style={styles.statText}>
              Confidence: {(item.confidence_threshold * 100).toFixed(0)}%
            </Text>
          </View>

          <View style={styles.listActionButtons}>
            <Button
              mode="outlined"
              compact
              onPress={() => handleToggleRecording(item.id)}
              icon={item.recording ? "stop" : "play-arrow"}
              style={styles.actionButton}
            >
              {item.recording ? "Stop" : "Record"}
            </Button>
          </View>
        </View>
      </Card.Content>
    </Card>
  );

  const EmptyState = () => (
    <View style={styles.emptyState}>
      <Icon name="videocam-off" size={64} color="#757575" />
      <Title style={styles.emptyTitle}>No Cameras</Title>
      <Paragraph style={styles.emptyText}>
        No cameras found. Add cameras to start monitoring.
      </Paragraph>
    </View>
  );

  const CameraStats = () => {
    const activeCount = cameras.filter(
      (c: Camera) => c.status === "active"
    ).length;
    const recordingCount = cameras.filter((c: Camera) => c.recording).length;
    const totalAlerts = cameras.reduce(
      (sum: number, c: Camera) => sum + c.alerts_count,
      0
    );

    return (
      <Surface style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{activeCount}</Text>
          <Text style={styles.statLabel}>Active</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{recordingCount}</Text>
          <Text style={styles.statLabel}>Recording</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{totalAlerts}</Text>
          <Text style={styles.statLabel}>Total Alerts</Text>
        </View>
      </Surface>
    );
  };

  return (
    <View style={styles.container}>
      {/* Search and Controls */}
      <Surface style={styles.controlsContainer}>
        <Searchbar
          placeholder="Search cameras..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchbar}
        />
        <View style={styles.controlButtons}>
          <IconButton
            icon={viewMode === "grid" ? "view-list" : "view-module"}
            onPress={() => setViewMode(viewMode === "grid" ? "list" : "grid")}
          />
          <Menu
            visible={filterMenuVisible}
            onDismiss={() => setFilterMenuVisible(false)}
            anchor={
              <IconButton
                icon="filter-list"
                onPress={() => setFilterMenuVisible(true)}
              />
            }
          >
            <Menu.Item
              onPress={() => {
                setSelectedFilter("all");
                setFilterMenuVisible(false);
              }}
              title="All Cameras"
            />
            <Divider />
            <Menu.Item
              onPress={() => {
                setSelectedFilter("active");
                setFilterMenuVisible(false);
              }}
              title="Active"
            />
            <Menu.Item
              onPress={() => {
                setSelectedFilter("inactive");
                setFilterMenuVisible(false);
              }}
              title="Inactive"
            />
            <Menu.Item
              onPress={() => {
                setSelectedFilter("maintenance");
                setFilterMenuVisible(false);
              }}
              title="Maintenance"
            />
            <Menu.Item
              onPress={() => {
                setSelectedFilter("error");
                setFilterMenuVisible(false);
              }}
              title="Error"
            />
          </Menu>
        </View>
      </Surface>

      {/* Camera Stats */}
      <CameraStats />

      {/* Cameras List */}
      <FlatList
        data={filteredCameras}
        renderItem={viewMode === "grid" ? renderGridCamera : renderListCamera}
        keyExtractor={(item) => item.id}
        style={styles.camerasList}
        numColumns={viewMode === "grid" ? 2 : 1}
        key={viewMode} // Force re-render when view mode changes
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={EmptyState}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={
          viewMode === "grid" ? styles.gridContainer : undefined
        }
      />

      {/* Add Camera FAB */}
      <FAB
        style={styles.fab}
        icon="add"
        onPress={() => Alert.alert("Add Camera", "Camera setup coming soon!")}
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
  controlsContainer: {
    flexDirection: "row",
    alignItems: "center",
    margin: 16,
    marginBottom: 8,
    borderRadius: 8,
    elevation: 2,
  },
  searchbar: {
    flex: 1,
    elevation: 0,
  },
  controlButtons: {
    flexDirection: "row",
  },
  statsContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    margin: 16,
    marginTop: 8,
    padding: 16,
    borderRadius: 8,
    elevation: 2,
  },
  statItem: {
    alignItems: "center",
  },
  statValue: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1976d2",
  },
  statLabel: {
    fontSize: 12,
    opacity: 0.7,
    marginTop: 4,
  },
  camerasList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  gridContainer: {
    justifyContent: "space-between",
  },
  // Grid styles
  gridCard: {
    marginBottom: 8,
  },
  gridCardContent: {
    padding: 12,
  },
  gridHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  gridCameraName: {
    fontSize: 14,
    fontWeight: "bold",
    marginBottom: 2,
  },
  gridCameraLocation: {
    fontSize: 11,
    opacity: 0.7,
    marginBottom: 8,
  },
  gridStats: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  statusChip: {
    height: 20,
  },
  alertsBadge: {
    backgroundColor: "#f44336",
  },
  confidenceContainer: {
    marginTop: 4,
  },
  confidenceLabel: {
    fontSize: 10,
    opacity: 0.7,
    marginBottom: 4,
  },
  confidenceBar: {
    height: 4,
    borderRadius: 2,
  },
  // List styles
  listCard: {
    marginBottom: 8,
  },
  listHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 12,
  },
  listCameraInfo: {
    flex: 1,
    marginRight: 12,
  },
  listTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  listCameraName: {
    fontSize: 16,
    fontWeight: "bold",
    flex: 1,
  },
  listCameraDetails: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 2,
  },
  listCameraLastSeen: {
    fontSize: 11,
    color: "#1976d2",
  },
  listActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  listStats: {
    flexDirection: "row",
    gap: 16,
  },
  statText: {
    fontSize: 12,
    opacity: 0.7,
  },
  listActionButtons: {
    flexDirection: "row",
    gap: 8,
  },
  actionButton: {
    minWidth: 70,
  },
  emptyState: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingVertical: 64,
  },
  emptyTitle: {
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    textAlign: "center",
    opacity: 0.7,
    paddingHorizontal: 32,
  },
  fab: {
    position: "absolute",
    margin: 16,
    right: 0,
    bottom: 0,
    backgroundColor: "#1976d2",
  },
});

export default CamerasScreen;
