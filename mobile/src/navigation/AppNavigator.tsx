/**
 * App Navigator
 * =============
 *
 * Main navigation structure for the mobile app with tab navigation and authentication flow.
 */

import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createStackNavigator } from "@react-navigation/stack";
import Icon from "react-native-vector-icons/MaterialIcons";

// Screens
import LoginScreen from "../screens/LoginScreen";
import DashboardScreen from "../screens/DashboardScreen";
import AlertsScreen from "../screens/AlertsScreen";
import CamerasScreen from "../screens/CamerasScreen";
import SettingsScreen from "../screens/SettingsScreen";
import AlertDetailScreen from "../screens/AlertDetailScreen";
import CameraDetailScreen from "../screens/CameraDetailScreen";

// Hooks
import { useAppSelector } from "../hooks/redux";

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Auth Stack
const AuthStack = () => (
  <Stack.Navigator screenOptions={{ headerShown: false }}>
    <Stack.Screen name="Login" component={LoginScreen} />
  </Stack.Navigator>
);

// Main Tab Navigator
const MainTabs = () => {
  const { isDarkMode } = useAppSelector((state) => state.ui);
  const { unreadCount } = useAppSelector((state) => state.alerts);

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case "Dashboard":
              iconName = "dashboard";
              break;
            case "Alerts":
              iconName = "warning";
              break;
            case "Cameras":
              iconName = "videocam";
              break;
            case "Settings":
              iconName = "settings";
              break;
            default:
              iconName = "help";
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: isDarkMode ? "#90caf9" : "#1976d2",
        tabBarInactiveTintColor: isDarkMode ? "#666666" : "#999999",
        tabBarStyle: {
          backgroundColor: isDarkMode ? "#1e1e1e" : "#ffffff",
          borderTopColor: isDarkMode ? "#333333" : "#e0e0e0",
        },
        headerStyle: {
          backgroundColor: isDarkMode ? "#1e1e1e" : "#ffffff",
        },
        headerTintColor: isDarkMode ? "#ffffff" : "#000000",
      })}
    >
      <Tab.Screen
        name="Dashboard"
        component={DashboardScreen}
        options={{
          title: "Dashboard",
          headerTitle: "Anomaly Detection",
        }}
      />
      <Tab.Screen
        name="Alerts"
        component={AlertsScreen}
        options={{
          title: "Alerts",
          tabBarBadge: unreadCount > 0 ? unreadCount : undefined,
        }}
      />
      <Tab.Screen
        name="Cameras"
        component={CamerasScreen}
        options={{
          title: "Cameras",
        }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          title: "Settings",
        }}
      />
    </Tab.Navigator>
  );
};

// Main Stack Navigator
const MainStack = () => (
  <Stack.Navigator>
    <Stack.Screen
      name="MainTabs"
      component={MainTabs}
      options={{ headerShown: false }}
    />
    <Stack.Screen
      name="AlertDetail"
      component={AlertDetailScreen}
      options={{
        title: "Alert Details",
        headerBackTitle: "Back",
      }}
    />
    <Stack.Screen
      name="CameraDetail"
      component={CameraDetailScreen}
      options={{
        title: "Camera Details",
        headerBackTitle: "Back",
      }}
    />
  </Stack.Navigator>
);

// App Navigator
const AppNavigator: React.FC = () => {
  const { isAuthenticated } = useAppSelector((state) => state.auth);

  return isAuthenticated ? <MainStack /> : <AuthStack />;
};

export default AppNavigator;
