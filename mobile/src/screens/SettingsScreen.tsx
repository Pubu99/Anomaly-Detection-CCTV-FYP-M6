/**
 * Settings Screen
 * ==============
 *
 * App settings and preferences screen.
 */

import React, { useState } from "react";
import { View, ScrollView, StyleSheet, Alert, Linking } from "react-native";
import {
  List,
  Switch,
  Button,
  Divider,
  Title,
  Paragraph,
  Card,
  Avatar,
  Text,
  Surface,
  Dialog,
  Portal,
  TextInput,
} from "react-native-paper";
import Icon from "react-native-vector-icons/MaterialIcons";

// Hooks
import { useAppSelector, useAppDispatch } from "../hooks/redux";

const SettingsScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { user } = useAppSelector((state) => state.auth);
  const { isDarkMode, notifications } = useAppSelector((state) => state.ui);

  const [showProfileDialog, setShowProfileDialog] = useState(false);
  const [showAboutDialog, setShowAboutDialog] = useState(false);
  const [profileData, setProfileData] = useState({
    username: user?.username || "",
    email: user?.email || "",
    phone: user?.phone || "",
  });

  const handleToggleDarkMode = () => {
    // dispatch(toggleDarkMode());
    console.log("Toggle dark mode");
  };

  const handleToggleNotifications = (type: string) => {
    // dispatch(toggleNotification(type));
    console.log(`Toggle ${type} notifications`);
  };

  const handleLogout = () => {
    Alert.alert("Logout", "Are you sure you want to logout?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Logout",
        style: "destructive",
        onPress: () => {
          // dispatch(logout());
          console.log("Logout");
        },
      },
    ]);
  };

  const handleUpdateProfile = () => {
    console.log("Update profile:", profileData);
    setShowProfileDialog(false);
    Alert.alert("Success", "Profile updated successfully");
  };

  const handleContactSupport = () => {
    Alert.alert("Contact Support", "Choose how to contact support:", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Email",
        onPress: () => Linking.openURL("mailto:support@anomalydetection.com"),
      },
      {
        text: "Phone",
        onPress: () => Linking.openURL("tel:+1234567890"),
      },
    ]);
  };

  const handleExportData = () => {
    Alert.alert("Export Data", "Export your data and settings?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Export",
        onPress: () => {
          console.log("Export data");
          Alert.alert("Success", "Data exported successfully");
        },
      },
    ]);
  };

  const handleClearCache = () => {
    Alert.alert("Clear Cache", "This will clear all cached data. Continue?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: () => {
          console.log("Clear cache");
          Alert.alert("Success", "Cache cleared successfully");
        },
      },
    ]);
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
      >
        {/* Profile Section */}
        <Card style={styles.profileCard}>
          <Card.Content>
            <View style={styles.profileHeader}>
              <Avatar.Text
                size={64}
                label={user?.username?.charAt(0).toUpperCase() || "U"}
                style={styles.profileAvatar}
              />
              <View style={styles.profileInfo}>
                <Title>{user?.username || "User"}</Title>
                <Paragraph>{user?.email || "user@example.com"}</Paragraph>
                <Button
                  mode="outlined"
                  compact
                  onPress={() => setShowProfileDialog(true)}
                  style={styles.editButton}
                >
                  Edit Profile
                </Button>
              </View>
            </View>
          </Card.Content>
        </Card>

        {/* Appearance Settings */}
        <Surface style={styles.section}>
          <List.Subheader>Appearance</List.Subheader>
          <List.Item
            title="Dark Mode"
            description="Switch between light and dark themes"
            left={() => <List.Icon icon="brightness-6" />}
            right={() => (
              <Switch value={isDarkMode} onValueChange={handleToggleDarkMode} />
            )}
          />
        </Surface>

        {/* Notification Settings */}
        <Surface style={styles.section}>
          <List.Subheader>Notifications</List.Subheader>
          <List.Item
            title="Security Alerts"
            description="Get notified about security events"
            left={() => <List.Icon icon="security" />}
            right={() => (
              <Switch
                value={notifications?.security !== false}
                onValueChange={() => handleToggleNotifications("security")}
              />
            )}
          />
          <List.Item
            title="System Updates"
            description="Notifications about system status"
            left={() => <List.Icon icon="system-update" />}
            right={() => (
              <Switch
                value={notifications?.system !== false}
                onValueChange={() => handleToggleNotifications("system")}
              />
            )}
          />
          <List.Item
            title="Push Notifications"
            description="Receive push notifications"
            left={() => <List.Icon icon="notifications" />}
            right={() => (
              <Switch
                value={notifications?.push !== false}
                onValueChange={() => handleToggleNotifications("push")}
              />
            )}
          />
        </Surface>

        {/* Data & Privacy */}
        <Surface style={styles.section}>
          <List.Subheader>Data & Privacy</List.Subheader>
          <List.Item
            title="Export Data"
            description="Download your data and settings"
            left={() => <List.Icon icon="download" />}
            onPress={handleExportData}
          />
          <List.Item
            title="Clear Cache"
            description="Clear app cache and temporary files"
            left={() => <List.Icon icon="clear-all" />}
            onPress={handleClearCache}
          />
        </Surface>

        {/* Support & Feedback */}
        <Surface style={styles.section}>
          <List.Subheader>Support & Feedback</List.Subheader>
          <List.Item
            title="Contact Support"
            description="Get help from our support team"
            left={() => <List.Icon icon="support-agent" />}
            onPress={handleContactSupport}
          />
          <List.Item
            title="Send Feedback"
            description="Help us improve the app"
            left={() => <List.Icon icon="feedback" />}
            onPress={() =>
              Alert.alert("Feedback", "Feedback feature coming soon!")
            }
          />
          <List.Item
            title="Rate App"
            description="Rate us on the app store"
            left={() => <List.Icon icon="star" />}
            onPress={() =>
              Alert.alert("Rate App", "Thank you for your support!")
            }
          />
        </Surface>

        {/* About */}
        <Surface style={styles.section}>
          <List.Subheader>About</List.Subheader>
          <List.Item
            title="App Version"
            description="1.0.0"
            left={() => <List.Icon icon="info" />}
          />
          <List.Item
            title="Privacy Policy"
            description="Read our privacy policy"
            left={() => <List.Icon icon="privacy-tip" />}
            onPress={() =>
              Alert.alert("Privacy Policy", "Privacy policy coming soon!")
            }
          />
          <List.Item
            title="Terms of Service"
            description="Read our terms of service"
            left={() => <List.Icon icon="description" />}
            onPress={() =>
              Alert.alert("Terms of Service", "Terms of service coming soon!")
            }
          />
          <List.Item
            title="About"
            description="Learn more about this app"
            left={() => <List.Icon icon="help" />}
            onPress={() => setShowAboutDialog(true)}
          />
        </Surface>

        {/* Logout Button */}
        <View style={styles.logoutContainer}>
          <Button
            mode="contained"
            onPress={handleLogout}
            style={styles.logoutButton}
            buttonColor="#f44336"
            icon="logout"
          >
            Logout
          </Button>
        </View>

        <View style={{ height: 50 }} />
      </ScrollView>

      {/* Profile Dialog */}
      <Portal>
        <Dialog
          visible={showProfileDialog}
          onDismiss={() => setShowProfileDialog(false)}
        >
          <Dialog.Title>Edit Profile</Dialog.Title>
          <Dialog.Content>
            <TextInput
              label="Username"
              value={profileData.username}
              onChangeText={(text) =>
                setProfileData({ ...profileData, username: text })
              }
              style={styles.input}
              mode="outlined"
            />
            <TextInput
              label="Email"
              value={profileData.email}
              onChangeText={(text) =>
                setProfileData({ ...profileData, email: text })
              }
              style={styles.input}
              mode="outlined"
              keyboardType="email-address"
            />
            <TextInput
              label="Phone"
              value={profileData.phone}
              onChangeText={(text) =>
                setProfileData({ ...profileData, phone: text })
              }
              style={styles.input}
              mode="outlined"
              keyboardType="phone-pad"
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowProfileDialog(false)}>Cancel</Button>
            <Button onPress={handleUpdateProfile}>Save</Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>

      {/* About Dialog */}
      <Portal>
        <Dialog
          visible={showAboutDialog}
          onDismiss={() => setShowAboutDialog(false)}
        >
          <Dialog.Title>About Anomaly Detection</Dialog.Title>
          <Dialog.Content>
            <Paragraph>
              Advanced AI-powered security monitoring system for detecting
              anomalous events in surveillance camera feeds.
            </Paragraph>
            <Paragraph style={{ marginTop: 16 }}>
              Built with cutting-edge machine learning technology to provide
              real-time threat detection and analysis.
            </Paragraph>
            <Paragraph style={{ marginTop: 16 }}>
              Version: 1.0.0{"\n"}
              Developed by: Security AI Team{"\n"}© 2024 All rights reserved
            </Paragraph>
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setShowAboutDialog(false)}>Close</Button>
          </Dialog.Actions>
        </Dialog>
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
  profileCard: {
    margin: 16,
    marginBottom: 8,
  },
  profileHeader: {
    flexDirection: "row",
    alignItems: "center",
  },
  profileAvatar: {
    backgroundColor: "#1976d2",
    marginRight: 16,
  },
  profileInfo: {
    flex: 1,
  },
  editButton: {
    marginTop: 8,
    alignSelf: "flex-start",
  },
  section: {
    margin: 16,
    marginVertical: 4,
    borderRadius: 8,
    elevation: 1,
  },
  logoutContainer: {
    margin: 16,
    marginTop: 24,
  },
  logoutButton: {
    paddingVertical: 4,
  },
  input: {
    marginBottom: 12,
  },
});

export default SettingsScreen;
