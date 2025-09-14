/**
 * Redux Store Configuration for Anomaly Detection Dashboard
 * ========================================================
 *
 * Centralized state management with Redux Toolkit, including:
 * - Authentication state
 * - UI preferences
 * - Camera management
 * - Alert management
 * - Real-time data updates
 */

import { configureStore, createSlice, PayloadAction } from "@reduxjs/toolkit";
import { persistStore, persistReducer } from "redux-persist";
import storage from "redux-persist/lib/storage";

// Types
export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

export interface Camera {
  id: number;
  camera_id: string;
  name: string;
  location: string;
  status: "active" | "inactive" | "error" | "maintenance";
  position_x: number;
  position_y: number;
  weight: number;
  resolution_width: number;
  resolution_height: number;
  fps: number;
  enabled: boolean;
  created_at: string;
  last_seen?: string;
  total_frames_processed: number;
  error_count: number;
}

export interface Alert {
  id: number;
  camera_id: string;
  anomaly_class: string;
  confidence: number;
  severity: "low" | "medium" | "high" | "critical";
  status: "active" | "acknowledged" | "resolved" | "false_positive";
  bbox?: number[];
  objects_detected?: string[];
  metadata?: Record<string, any>;
  frame_path?: string;
  video_clip_path?: string;
  emergency_type?: "police" | "medical" | "fire" | "security";
  emergency_contact_sent: boolean;
  created_at: string;
  acknowledged_at?: string;
  acknowledged_by?: string;
  resolved_at?: string;
  resolved_by?: string;
}

export interface SystemStats {
  total_cameras: number;
  active_cameras: number;
  total_alerts_24h: number;
  critical_alerts_24h: number;
  system_uptime: number;
  average_fps: number;
  model_accuracy: number;
  last_retrain?: string;
  error_rate_24h: number;
  false_positive_rate: number;
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
  sidebarOpen: boolean;
  selectedCamera: string | null;
  viewMode: "grid" | "list" | "map";
  alertFilters: {
    severity: string[];
    status: string[];
    cameras: string[];
  };
  notifications: Array<{
    id: string;
    type: "success" | "error" | "warning" | "info";
    message: string;
    timestamp: string;
  }>;
}

const initialUIState: UIState = {
  isDarkMode: false,
  sidebarOpen: true,
  selectedCamera: null,
  viewMode: "grid",
  alertFilters: {
    severity: [],
    status: [],
    cameras: [],
  },
  notifications: [],
};

export const uiSlice = createSlice({
  name: "ui",
  initialState: initialUIState,
  reducers: {
    toggleDarkMode: (state) => {
      state.isDarkMode = !state.isDarkMode;
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    setSelectedCamera: (state, action: PayloadAction<string | null>) => {
      state.selectedCamera = action.payload;
    },
    setViewMode: (state, action: PayloadAction<"grid" | "list" | "map">) => {
      state.viewMode = action.payload;
    },
    setAlertFilters: (
      state,
      action: PayloadAction<Partial<UIState["alertFilters"]>>
    ) => {
      state.alertFilters = { ...state.alertFilters, ...action.payload };
    },
    addNotification: (
      state,
      action: PayloadAction<
        Omit<UIState["notifications"][0], "id" | "timestamp">
      >
    ) => {
      const notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
      };
      state.notifications.push(notification);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(
        (n) => n.id !== action.payload
      );
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
  },
});

// Cameras Slice
interface CamerasState {
  cameras: Camera[];
  loading: boolean;
  error: string | null;
  selectedCamera: Camera | null;
}

const initialCamerasState: CamerasState = {
  cameras: [],
  loading: false,
  error: null,
  selectedCamera: null,
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
    updateCamera: (state, action: PayloadAction<Camera>) => {
      const index = state.cameras.findIndex((c) => c.id === action.payload.id);
      if (index !== -1) {
        state.cameras[index] = action.payload;
      }
    },
    addCamera: (state, action: PayloadAction<Camera>) => {
      state.cameras.push(action.payload);
    },
    removeCamera: (state, action: PayloadAction<number>) => {
      state.cameras = state.cameras.filter((c) => c.id !== action.payload);
    },
    selectCamera: (state, action: PayloadAction<Camera | null>) => {
      state.selectedCamera = action.payload;
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

// Alerts Slice
interface AlertsState {
  alerts: Alert[];
  loading: boolean;
  error: string | null;
  totalCount: number;
  currentPage: number;
  pageSize: number;
  realTimeEnabled: boolean;
}

const initialAlertsState: AlertsState = {
  alerts: [],
  loading: false,
  error: null,
  totalCount: 0,
  currentPage: 1,
  pageSize: 50,
  realTimeEnabled: true,
};

export const alertsSlice = createSlice({
  name: "alerts",
  initialState: initialAlertsState,
  reducers: {
    fetchAlertsStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchAlertsSuccess: (
      state,
      action: PayloadAction<{ alerts: Alert[]; totalCount: number }>
    ) => {
      state.loading = false;
      state.alerts = action.payload.alerts;
      state.totalCount = action.payload.totalCount;
      state.error = null;
    },
    fetchAlertsFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    addAlert: (state, action: PayloadAction<Alert>) => {
      state.alerts.unshift(action.payload);
      state.totalCount += 1;
    },
    updateAlert: (state, action: PayloadAction<Alert>) => {
      const index = state.alerts.findIndex((a) => a.id === action.payload.id);
      if (index !== -1) {
        state.alerts[index] = action.payload;
      }
    },
    acknowledgeAlert: (
      state,
      action: PayloadAction<{ id: number; acknowledgedBy: string }>
    ) => {
      const alert = state.alerts.find((a) => a.id === action.payload.id);
      if (alert) {
        alert.status = "acknowledged";
        alert.acknowledged_at = new Date().toISOString();
        alert.acknowledged_by = action.payload.acknowledgedBy;
      }
    },
    resolveAlert: (
      state,
      action: PayloadAction<{ id: number; resolvedBy: string }>
    ) => {
      const alert = state.alerts.find((a) => a.id === action.payload.id);
      if (alert) {
        alert.status = "resolved";
        alert.resolved_at = new Date().toISOString();
        alert.resolved_by = action.payload.resolvedBy;
      }
    },
    setCurrentPage: (state, action: PayloadAction<number>) => {
      state.currentPage = action.payload;
    },
    setPageSize: (state, action: PayloadAction<number>) => {
      state.pageSize = action.payload;
    },
    toggleRealTime: (state) => {
      state.realTimeEnabled = !state.realTimeEnabled;
    },
  },
});

// System Slice
interface SystemState {
  stats: SystemStats | null;
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
  connectionStatus: "connected" | "connecting" | "disconnected";
  inferenceRunning: boolean;
}

const initialSystemState: SystemState = {
  stats: null,
  loading: false,
  error: null,
  lastUpdated: null,
  connectionStatus: "disconnected",
  inferenceRunning: false,
};

export const systemSlice = createSlice({
  name: "system",
  initialState: initialSystemState,
  reducers: {
    fetchStatsStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchStatsSuccess: (state, action: PayloadAction<SystemStats>) => {
      state.loading = false;
      state.stats = action.payload;
      state.lastUpdated = new Date().toISOString();
      state.error = null;
    },
    fetchStatsFailure: (state, action: PayloadAction<string>) => {
      state.loading = false;
      state.error = action.payload;
    },
    setConnectionStatus: (
      state,
      action: PayloadAction<SystemState["connectionStatus"]>
    ) => {
      state.connectionStatus = action.payload;
    },
    setInferenceRunning: (state, action: PayloadAction<boolean>) => {
      state.inferenceRunning = action.payload;
    },
  },
});

// Root reducer
const rootReducer = {
  auth: persistReducer(
    {
      key: "auth",
      storage,
      whitelist: ["isAuthenticated", "user", "token"],
    },
    authSlice.reducer
  ),
  ui: persistReducer(
    {
      key: "ui",
      storage,
      whitelist: ["isDarkMode", "viewMode", "alertFilters"],
    },
    uiSlice.reducer
  ),
  cameras: camerasSlice.reducer,
  alerts: alertsSlice.reducer,
  system: systemSlice.reducer,
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
  toggleSidebar,
  setSidebarOpen,
  setSelectedCamera,
  setViewMode,
  setAlertFilters,
  addNotification,
  removeNotification,
  clearNotifications,
} = uiSlice.actions;

export const {
  fetchCamerasStart,
  fetchCamerasSuccess,
  fetchCamerasFailure,
  updateCamera,
  addCamera,
  removeCamera,
  selectCamera,
  updateCameraStatus,
} = camerasSlice.actions;

export const {
  fetchAlertsStart,
  fetchAlertsSuccess,
  fetchAlertsFailure,
  addAlert,
  updateAlert,
  acknowledgeAlert,
  resolveAlert,
  setCurrentPage,
  setPageSize,
  toggleRealTime,
} = alertsSlice.actions;

export const {
  fetchStatsStart,
  fetchStatsSuccess,
  fetchStatsFailure,
  setConnectionStatus,
  setInferenceRunning,
} = systemSlice.actions;
