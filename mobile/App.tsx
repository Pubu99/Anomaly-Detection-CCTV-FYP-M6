/**
 * React Native Mobile App for Multi-Camera Anomaly Detection
 * ==========================================================
 *
 * Cross-platform mobile application for real-time monitoring,
 * instant notifications, camera viewing, and alert management.
 */

import React, { useEffect } from "react";
import {
  StatusBar,
  Platform,
  Alert,
  AppState,
  AppStateStatus,
} from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { Provider as PaperProvider } from "react-native-paper";
import { Provider as ReduxProvider } from "react-redux";
import { PersistGate } from "redux-persist/integration/react";
import PushNotification from "react-native-push-notification";
import { request, PERMISSIONS, RESULTS } from "react-native-permissions";

// Store and theme
import { store, persistor } from "./src/store/store";
import { lightTheme, darkTheme } from "./src/theme/theme";

// Navigation
import AppNavigator from "./src/navigation/AppNavigator";

// Components
import LoadingScreen from "./src/components/LoadingScreen";
import NotificationHandler from "./src/components/NotificationHandler";

// Hooks and services
import { useAppSelector } from "./src/hooks/redux";
import { NotificationService } from "./src/services/NotificationService";
import { WebSocketService } from "./src/services/WebSocketService";

const App: React.FC = () => {
  return (
    <ReduxProvider store={store}>
      <PersistGate loading={<LoadingScreen />} persistor={persistor}>
        <AppContent />
      </PersistGate>
    </ReduxProvider>
  );
};

const AppContent: React.FC = () => {
  const { isDarkMode } = useAppSelector((state) => state.ui);
  const { isAuthenticated } = useAppSelector((state) => state.auth);

  const theme = isDarkMode ? darkTheme : lightTheme;

  useEffect(() => {
    initializeApp();
    setupNotifications();
    handleAppStateChange();
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      WebSocketService.connect();
    } else {
      WebSocketService.disconnect();
    }
  }, [isAuthenticated]);

  const initializeApp = async () => {
    try {
      // Request permissions
      await requestPermissions();

      // Initialize services
      await NotificationService.initialize();

      console.log("App initialized successfully");
    } catch (error) {
      console.error("App initialization failed:", error);
    }
  };

  const requestPermissions = async () => {
    try {
      // Camera permission
      const cameraPermission = await request(
        Platform.OS === "ios"
          ? PERMISSIONS.IOS.CAMERA
          : PERMISSIONS.ANDROID.CAMERA
      );

      // Notification permission
      const notificationPermission = await request(
        Platform.OS === "ios"
          ? PERMISSIONS.IOS.NOTIFICATIONS
          : PERMISSIONS.ANDROID.POST_NOTIFICATIONS
      );

      if (cameraPermission !== RESULTS.GRANTED) {
        Alert.alert(
          "Camera Permission",
          "Camera access is needed to view live feeds and capture images.",
          [{ text: "OK" }]
        );
      }

      if (notificationPermission !== RESULTS.GRANTED) {
        Alert.alert(
          "Notification Permission",
          "Notifications are needed to receive real-time alerts.",
          [{ text: "OK" }]
        );
      }
    } catch (error) {
      console.error("Permission request failed:", error);
    }
  };

  const setupNotifications = () => {
    // Configure push notifications
    PushNotification.configure({
      onNotification: function (notification) {
        console.log("Notification received:", notification);
        // Handle notification tap
        if (notification.userInteraction) {
          // Navigate to alerts screen
          // This would be handled by the NotificationHandler component
        }
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: Platform.OS === "ios",
    });

    // Create notification channels for Android
    if (Platform.OS === "android") {
      PushNotification.createChannel(
        {
          channelId: "critical-alerts",
          channelName: "Critical Alerts",
          channelDescription: "Critical security alerts",
          importance: 4,
          vibrate: true,
          sound: "alert_sound.mp3",
        },
        (created) => console.log(`Critical alerts channel created: ${created}`)
      );

      PushNotification.createChannel(
        {
          channelId: "general-alerts",
          channelName: "General Alerts",
          channelDescription: "General system alerts",
          importance: 3,
          vibrate: true,
        },
        (created) => console.log(`General alerts channel created: ${created}`)
      );
    }
  };

  const handleAppStateChange = () => {
    const handleChange = (nextAppState: AppStateStatus) => {
      if (nextAppState === "active") {
        // App came to foreground
        console.log("App active");
        if (isAuthenticated) {
          WebSocketService.reconnect();
        }
      } else if (nextAppState === "background") {
        // App went to background
        console.log("App backgrounded");
      }
    };

    const subscription = AppState.addEventListener("change", handleChange);
    return () => subscription?.remove();
  };

  return (
    <PaperProvider theme={theme}>
      <StatusBar
        barStyle={isDarkMode ? "light-content" : "dark-content"}
        backgroundColor={theme.colors.surface}
        translucent={false}
      />

      <NavigationContainer theme={theme}>
        <AppNavigator />
      </NavigationContainer>

      <NotificationHandler />
    </PaperProvider>
  );
};

export default App;
