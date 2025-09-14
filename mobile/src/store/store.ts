/**
 * Redux Store Configuration for Mobile App
 * ========================================
 *
 * Simplified store for mobile with focus on alerts, notifications, and offline support.
 */

import { configureStore, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { persistStore, persistReducer } from "redux-persist";
import AsyncStorage from "@react-native-async-storage/async-storage";

// Types
export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

export interface Alert {
  id: number;
  camera_id: string;
  anomaly_class: string;
  confidence: number;
  severity: "low" | "medium" | "high" | "critical";
  status: "active" | "acknowledged" | "resolved";
  created_at: string;
  emergency_type?: "police" | "medical" | "fire" | "security";
}

export interface Camera {
  id: number;
  camera_id: string;
  name: string;
  location: string;
  status: "active" | "inactive" | "error" | "maintenance";
  enabled: boolean;
}

// Auth Slice
interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

const initialAuthState: AuthState = {
  isAuthenticated: false,
  user: null,
  token: null,
  loading: false,
  error: null,
};

export const authSlice = createSlice({
  name: "auth",
  initialState: initialAuthState,
  reducers: {
    loginStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    loginSuccess: (
      state,
      action: PayloadAction<{ user: User; token: string }>
    ) => {
      state.loading = false;
      state.isAuthenticated = true;
      state.user = action.payload.user;
      state.token = action.payload.token;
      state.error = null;
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
      state.error = action.payload;
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
      state.error = null;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
});

// UI Slice
interface UIState {
  isDarkMode: boolean;
  isConnected: boolean;
  notificationsEnabled: boolean;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
  selectedCamera: string | null;
  refreshing: boolean;
}

const initialUIState: UIState = {
  isDarkMode: false,
  isConnected: false,
  notificationsEnabled: true,
  soundEnabled: true,
  vibrationEnabled: true,
  selectedCamera: null,
  refreshing: false,
};

export const uiSlice = createSlice({
  name: "ui",
  initialState: initialUIState,
  reducers: {
    toggleDarkMode: (state) => {
      state.isDarkMode = !state.isDarkMode;
    },
    setConnectionStatus: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    toggleNotifications: (state) => {
      state.notificationsEnabled = !state.notificationsEnabled;
    },
    toggleSound: (state) => {
      state.soundEnabled = !state.soundEnabled;
    },
    toggleVibration: (state) => {
      state.vibrationEnabled = !state.vibrationEnabled;
    },
    setSelectedCamera: (state, action: PayloadAction<string | null>) => {
      state.selectedCamera = action.payload;
    },
    setRefreshing: (state, action: PayloadAction<boolean>) => {
      state.refreshing = action.payload;
    },
  },
});

// Alerts Slice
interface AlertsState {
  alerts: Alert[];
  unreadCount: number;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialAlertsState: AlertsState = {
  alerts: [],
  unreadCount: 0,
  loading: false,
  error: null,
  lastUpdated: null,
};

export const alertsSlice = createSlice({
  name: "alerts",
  initialState: initialAlertsState,
  reducers: {
    fetchAlertsStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchAlertsSuccess: (state, action: PayloadAction<Alert[]>) => {
      state.loading = false;
      state.alerts = action.payload;
      state.lastUpdated = new Date().toISOString();
      state.error = null;
    },
    fetchAlertsFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    addAlert: (state, action: PayloadAction<Alert>) => {
      state.alerts.unshift(action.payload);
      if (action.payload.status === "active") {
        state.unreadCount += 1;
      }
    },
    acknowledgeAlert: (state, action: PayloadAction<number>) => {
      const alert = state.alerts.find((a) => a.id === action.payload);
      if (alert && alert.status === "active") {
        alert.status = "acknowledged";
        state.unreadCount = Math.max(0, state.unreadCount - 1);
      }
    },
    markAllAsRead: (state) => {
      state.unreadCount = 0;
    },
    clearAlerts: (state) => {
      state.alerts = [];
      state.unreadCount = 0;
    },
  },
});

// Cameras Slice
interface CamerasState {
  cameras: Camera[];
  loading: boolean;
  error: string | null;
}

const initialCamerasState: CamerasState = {
  cameras: [],
  loading: false,
  error: null,
};

export const camerasSlice = createSlice({
  name: "cameras",
  initialState: initialCamerasState,
  reducers: {
    fetchCamerasStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchCamerasSuccess: (state, action: PayloadAction<Camera[]>) => {
      state.loading = false;
      state.cameras = action.payload;
      state.error = null;
    },
    fetchCamerasFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    updateCameraStatus: (
      state,
      action: PayloadAction<{ id: number; status: Camera["status"] }>
    ) => {
      const camera = state.cameras.find((c) => c.id === action.payload.id);
      if (camera) {
        camera.status = action.payload.status;
      }
    },
  },
});

// Notifications Slice
interface NotificationState {
  notifications: Array<{
    id: string;
    title: string;
    message: string;
    type: "info" | "warning" | "error" | "success";
    timestamp: string;
    read: boolean;
  }>;
  pushToken: string | null;
}

const initialNotificationState: NotificationState = {
  notifications: [],
  pushToken: null,
};

export const notificationSlice = createSlice({
  name: "notifications",
  initialState: initialNotificationState,
  reducers: {
    addNotification: (
      state,
      action: PayloadAction<{
        title: string;
        message: string;
        type: "info" | "warning" | "error" | "success";
      }>
    ) => {
      const notification = {
        id: Date.now().toString(),
        ...action.payload,
        timestamp: new Date().toISOString(),
        read: false,
      };
      state.notifications.unshift(notification);
    },
    markNotificationRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(
        (n) => n.id === action.payload
      );
      if (notification) {
        notification.read = true;
      }
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setPushToken: (state, action: PayloadAction<string>) => {
      state.pushToken = action.payload;
    },
  },
});

// Root reducer
const rootReducer = {
  auth: persistReducer(
    {
      key: "auth",
      storage: AsyncStorage,
      whitelist: ["isAuthenticated", "user", "token"],
    },
    authSlice.reducer
  ),
  ui: persistReducer(
    {
      key: "ui",
      storage: AsyncStorage,
      whitelist: [
        "isDarkMode",
        "notificationsEnabled",
        "soundEnabled",
        "vibrationEnabled",
      ],
    },
    uiSlice.reducer
  ),
  alerts: persistReducer(
    {
      key: "alerts",
      storage: AsyncStorage,
      whitelist: ["alerts", "unreadCount"],
    },
    alertsSlice.reducer
  ),
  cameras: camerasSlice.reducer,
  notifications: notificationSlice.reducer,
};

// Configure store
export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ["persist/PERSIST", "persist/REHYDRATE"],
      },
    }),
});

export const persistor = persistStore(store);

// Types
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Action exports
export const { loginStart, loginSuccess, loginFailure, logout, clearError } =
  authSlice.actions;

export const {
  toggleDarkMode,
  setConnectionStatus,
  toggleNotifications,
  toggleSound,
  toggleVibration,
  setSelectedCamera,
  setRefreshing,
} = uiSlice.actions;

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  addAlert,
  acknowledgeAlert,
  markAllAsRead,
  clearAlerts,
} = alertsSlice.actions;

export const {
  fetchCamerasStart,
  fetchCamerasSuccess,
  fetchCamerasFailure,
  updateCameraStatus,
} = camerasSlice.actions;

export const {
  addNotification,
  markNotificationRead,
  clearNotifications,
  setPushToken,
} = notificationSlice.actions;
