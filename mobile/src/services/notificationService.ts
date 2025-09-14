/**
 * Notification Service
 * ===================
 *
 * Handles push notifications and local notifications for the mobile app.
 */

import { Alert, Platform } from "react-native";

interface NotificationData {
  title: string;
  body: string;
  data?: any;
  sound?: boolean;
  vibration?: boolean;
}

interface NotificationPermissionStatus {
  granted: boolean;
  denied: boolean;
  provisional?: boolean;
}

class NotificationService {
  private isInitialized: boolean = false;
  private permissionStatus: NotificationPermissionStatus = {
    granted: false,
    denied: false,
  };

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Initialize notification services
      console.log("Initializing notification service...");

      // For React Native, you would typically use:
      // - @react-native-firebase/messaging for push notifications
      // - @react-native-async-storage/async-storage for local storage
      // - react-native-push-notification for local notifications

      this.isInitialized = true;
      console.log("Notification service initialized");
    } catch (error) {
      console.error("Failed to initialize notification service:", error);
    }
  }

  async requestPermission(): Promise<NotificationPermissionStatus> {
    try {
      if (Platform.OS === "ios") {
        // For iOS, request notification permissions
        const permission = await this.requestIOSPermission();
        this.permissionStatus = permission;
      } else {
        // For Android, permissions are typically granted by default
        this.permissionStatus = { granted: true, denied: false };
      }

      return this.permissionStatus;
    } catch (error) {
      console.error("Failed to request notification permission:", error);
      return { granted: false, denied: true };
    }
  }

  private async requestIOSPermission(): Promise<NotificationPermissionStatus> {
    return new Promise((resolve) => {
      // Simulate iOS permission request
      Alert.alert(
        "Enable Notifications",
        "Allow notifications to receive security alerts?",
        [
          {
            text: "Don't Allow",
            onPress: () => resolve({ granted: false, denied: true }),
            style: "cancel",
          },
          {
            text: "Allow",
            onPress: () => resolve({ granted: true, denied: false }),
          },
        ]
      );
    });
  }

  async showLocalNotification(data: NotificationData): Promise<void> {
    try {
      if (!this.permissionStatus.granted) {
        console.warn("Notification permission not granted");
        return;
      }

      // In a real app, you would use a notification library like:
      // PushNotification.localNotification({
      //   title: data.title,
      //   message: data.body,
      //   playSound: data.sound !== false,
      //   vibrate: data.vibration !== false,
      //   userInfo: data.data,
      // });

      // For now, simulate with Alert
      console.log("Local notification:", data);
      Alert.alert(data.title, data.body);
    } catch (error) {
      console.error("Failed to show local notification:", error);
    }
  }

  async scheduleNotification(
    data: NotificationData,
    delay: number
  ): Promise<void> {
    try {
      if (!this.permissionStatus.granted) {
        console.warn("Notification permission not granted");
        return;
      }

      // Schedule notification after delay
      setTimeout(() => {
        this.showLocalNotification(data);
      }, delay);

      console.log(`Notification scheduled for ${delay}ms`);
    } catch (error) {
      console.error("Failed to schedule notification:", error);
    }
  }

  async cancelAllNotifications(): Promise<void> {
    try {
      // In a real app:
      // PushNotification.cancelAllLocalNotifications();
      console.log("All notifications cancelled");
    } catch (error) {
      console.error("Failed to cancel notifications:", error);
    }
  }

  async registerForPushNotifications(): Promise<string | null> {
    try {
      if (!this.permissionStatus.granted) {
        await this.requestPermission();
      }

      if (!this.permissionStatus.granted) {
        throw new Error("Push notification permission denied");
      }

      // In a real app with Firebase:
      // const token = await messaging().getToken();
      // return token;

      // Simulate token generation
      const mockToken = `mock_token_${Date.now()}`;
      console.log("Push notification token:", mockToken);
      return mockToken;
    } catch (error) {
      console.error("Failed to register for push notifications:", error);
      return null;
    }
  }

  async handleBackgroundMessage(message: any): Promise<void> {
    try {
      console.log("Background message received:", message);

      // Handle background notification
      if (message.data) {
        await this.showLocalNotification({
          title: message.notification?.title || "Security Alert",
          body: message.notification?.body || "New security event detected",
          data: message.data,
        });
      }
    } catch (error) {
      console.error("Failed to handle background message:", error);
    }
  }

  async setBadgeCount(count: number): Promise<void> {
    try {
      // In a real app:
      // PushNotification.setApplicationIconBadgeNumber(count);
      console.log(`Badge count set to: ${count}`);
    } catch (error) {
      console.error("Failed to set badge count:", error);
    }
  }

  getPermissionStatus(): NotificationPermissionStatus {
    return this.permissionStatus;
  }

  isPermissionGranted(): boolean {
    return this.permissionStatus.granted;
  }

  // Notification categories for different alert types
  createSecurityAlertNotification(
    alertType: string,
    cameraName: string,
    severity: string
  ): NotificationData {
    const severityEmoji =
      severity === "high" ? "🚨" : severity === "medium" ? "⚠️" : "ℹ️";

    return {
      title: `${severityEmoji} Security Alert`,
      body: `${alertType} detected on ${cameraName}`,
      data: {
        type: "security_alert",
        alertType,
        cameraName,
        severity,
        timestamp: new Date().toISOString(),
      },
      sound: severity === "high",
      vibration: true,
    };
  }

  createSystemNotification(
    message: string,
    type: "info" | "warning" | "error" = "info"
  ): NotificationData {
    const typeEmoji =
      type === "error" ? "❌" : type === "warning" ? "⚠️" : "ℹ️";

    return {
      title: `${typeEmoji} System Notification`,
      body: message,
      data: {
        type: "system_notification",
        notificationType: type,
        timestamp: new Date().toISOString(),
      },
      sound: type === "error",
      vibration: type !== "info",
    };
  }
}

export const notificationService = new NotificationService();
