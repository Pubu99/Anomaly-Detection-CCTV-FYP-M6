/**
 * Alerts Screen
 * ============
 *
 * Screen for viewing and managing security alerts.
 */

import React, { useState, useEffect } from "react";
import {
  View,
  FlatList,
  RefreshControl,
  StyleSheet,
  Alert,
} from "react-native";
import {
  Card,
  Title,
  Paragraph,
  List,
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
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";

// Hooks
import { useAppSelector, useAppDispatch } from "../hooks/redux";

// Types
interface Alert {
  id: string;
  event_type: string;
  camera_name: string;
  timestamp: string;
  confidence: number;
  severity: "low" | "medium" | "high";
  status: "new" | "acknowledged" | "resolved";
  description?: string;
  location?: string;
}

const AlertsScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { alerts } = useAppSelector((state) => state.alerts);
  const { isDarkMode } = useAppSelector((state) => state.ui);

  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterMenuVisible, setFilterMenuVisible] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState("all");
  const [sortBy, setSortBy] = useState("timestamp");

  const filteredAlerts = alerts
    .filter((alert: Alert) => {
      // Search filter
      const matchesSearch =
        alert.event_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        alert.camera_name.toLowerCase().includes(searchQuery.toLowerCase());

      // Status filter
      const matchesFilter =
        selectedFilter === "all" ||
        (selectedFilter === "unresolved" && alert.status !== "resolved") ||
        alert.severity === selectedFilter ||
        alert.status === selectedFilter;

      return matchesSearch && matchesFilter;
    })
    .sort((a: Alert, b: Alert) => {
      if (sortBy === "timestamp") {
        return (
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
      } else if (sortBy === "severity") {
        const severityOrder = { high: 3, medium: 2, low: 1 };
        return severityOrder[b.severity] - severityOrder[a.severity];
      }
      return 0;
    });

  const onRefresh = async () => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => setRefreshing(false), 1000);
  };

  const handleAlertPress = (alert: Alert) => {
    // Navigate to alert detail screen
    console.log("Navigate to alert detail:", alert.id);
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    Alert.alert("Acknowledge Alert", "Mark this alert as acknowledged?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Acknowledge",
        onPress: () => {
          console.log("Acknowledge alert:", alertId);
          // dispatch(acknowledgeAlert(alertId));
        },
      },
    ]);
  };

  const handleResolveAlert = (alertId: string) => {
    Alert.alert("Resolve Alert", "Mark this alert as resolved?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Resolve",
        onPress: () => {
          console.log("Resolve alert:", alertId);
          // dispatch(resolveAlert(alertId));
        },
      },
    ]);
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
    const now = new Date();
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60);

    if (diffInHours < 1) {
      return `${Math.floor(diffInHours * 60)}m ago`;
    } else if (diffInHours < 24) {
      return `${Math.floor(diffInHours)}h ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const renderAlert = ({ item }: { item: Alert }) => (
    <Card style={styles.alertCard} onPress={() => handleAlertPress(item)}>
      <Card.Content>
        <View style={styles.alertHeader}>
          <View style={styles.alertInfo}>
            <View style={styles.alertTitleRow}>
              <Text style={styles.alertTitle}>{item.event_type}</Text>
              <Chip
                style={[
                  styles.severityChip,
                  { backgroundColor: getSeverityColor(item.severity) },
                ]}
                textStyle={{ color: "#ffffff", fontSize: 10 }}
              >
                {item.severity.toUpperCase()}
              </Chip>
            </View>
            <Text style={styles.alertSubtitle}>
              📹 {item.camera_name} • ⏰ {formatTimestamp(item.timestamp)}
            </Text>
            <Text style={styles.alertConfidence}>
              Confidence: {(item.confidence * 100).toFixed(0)}%
            </Text>
          </View>
          <Avatar.Icon
            size={40}
            icon="warning"
            style={{
              backgroundColor: getSeverityColor(item.severity),
            }}
          />
        </View>

        <View style={styles.alertActions}>
          <Chip
            style={[
              styles.statusChip,
              { backgroundColor: getStatusColor(item.status) },
            ]}
            textStyle={{ color: "#ffffff", fontSize: 10 }}
          >
            {item.status.toUpperCase()}
          </Chip>

          <View style={styles.actionButtons}>
            {item.status === "new" && (
              <Button
                mode="outlined"
                compact
                onPress={() => handleAcknowledgeAlert(item.id)}
                style={styles.actionButton}
              >
                Acknowledge
              </Button>
            )}
            {item.status !== "resolved" && (
              <Button
                mode="contained"
                compact
                onPress={() => handleResolveAlert(item.id)}
                style={styles.actionButton}
              >
                Resolve
              </Button>
            )}
          </View>
        </View>
      </Card.Content>
    </Card>
  );

  const EmptyState = () => (
    <View style={styles.emptyState}>
      <Icon name="check-circle" size={64} color="#4caf50" />
      <Title style={styles.emptyTitle}>No Alerts</Title>
      <Paragraph style={styles.emptyText}>
        All systems are running normally. No security alerts at this time.
      </Paragraph>
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Search and Filter */}
      <Surface style={styles.searchContainer}>
        <Searchbar
          placeholder="Search alerts..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchbar}
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
            title="All Alerts"
          />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("unresolved");
              setFilterMenuVisible(false);
            }}
            title="Unresolved"
          />
          <Divider />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("high");
              setFilterMenuVisible(false);
            }}
            title="High Severity"
          />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("medium");
              setFilterMenuVisible(false);
            }}
            title="Medium Severity"
          />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("low");
              setFilterMenuVisible(false);
            }}
            title="Low Severity"
          />
          <Divider />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("new");
              setFilterMenuVisible(false);
            }}
            title="New Alerts"
          />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("acknowledged");
              setFilterMenuVisible(false);
            }}
            title="Acknowledged"
          />
          <Menu.Item
            onPress={() => {
              setSelectedFilter("resolved");
              setFilterMenuVisible(false);
            }}
            title="Resolved"
          />
        </Menu>
      </Surface>

      {/* Alert Stats */}
      <Surface style={styles.statsContainer}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>
            {alerts.filter((a: Alert) => a.status === "new").length}
          </Text>
          <Text style={styles.statLabel}>New</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>
            {alerts.filter((a: Alert) => a.severity === "high").length}
          </Text>
          <Text style={styles.statLabel}>High Risk</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{filteredAlerts.length}</Text>
          <Text style={styles.statLabel}>Filtered</Text>
        </View>
      </Surface>

      {/* Alerts List */}
      <FlatList
        data={filteredAlerts}
        renderItem={renderAlert}
        keyExtractor={(item) => item.id}
        style={styles.alertsList}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={EmptyState}
        showsVerticalScrollIndicator={false}
      />

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
  searchContainer: {
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
  alertsList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  alertCard: {
    marginBottom: 8,
  },
  alertHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 12,
  },
  alertInfo: {
    flex: 1,
    marginRight: 12,
  },
  alertTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  alertTitle: {
    fontSize: 16,
    fontWeight: "bold",
    flex: 1,
  },
  alertSubtitle: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 4,
  },
  alertConfidence: {
    fontSize: 12,
    fontWeight: "500",
    color: "#1976d2",
  },
  severityChip: {
    marginLeft: 8,
    height: 24,
  },
  alertActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  statusChip: {
    height: 24,
  },
  actionButtons: {
    flexDirection: "row",
    gap: 8,
  },
  actionButton: {
    minWidth: 80,
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
    backgroundColor: "#f44336",
  },
});

export default AlertsScreen;
