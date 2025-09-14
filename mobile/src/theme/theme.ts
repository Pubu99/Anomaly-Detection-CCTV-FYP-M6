/**
 * React Native Paper Theme Configuration
 * =====================================
 *
 * Custom theme for mobile app with Material Design 3 support.
 */

import { MD3LightTheme, MD3DarkTheme } from "react-native-paper";

// Common theme properties
const commonTheme = {
  fonts: {
    regular: {
      fontFamily: "System",
      fontWeight: "400" as const,
    },
    medium: {
      fontFamily: "System",
      fontWeight: "500" as const,
    },
    bold: {
      fontFamily: "System",
      fontWeight: "700" as const,
    },
  },
  roundness: 8,
};

// Light theme
export const lightTheme = {
  ...MD3LightTheme,
  ...commonTheme,
  colors: {
    ...MD3LightTheme.colors,
    primary: "#1976d2",
    primaryContainer: "#e3f2fd",
    secondary: "#dc004e",
    secondaryContainer: "#fce4ec",
    tertiary: "#9c27b0",
    tertiaryContainer: "#f3e5f5",
    surface: "#ffffff",
    surfaceVariant: "#f5f5f5",
    background: "#fafafa",
    error: "#d32f2f",
    errorContainer: "#ffebee",
    onPrimary: "#ffffff",
    onSecondary: "#ffffff",
    onTertiary: "#ffffff",
    onSurface: "#1a1a1a",
    onSurfaceVariant: "#666666",
    onBackground: "#1a1a1a",
    outline: "#cccccc",
    outlineVariant: "#e0e0e0",
    shadow: "#000000",
    scrim: "#000000",

    // Custom colors for alerts
    critical: "#d32f2f",
    high: "#f57c00",
    medium: "#ffa000",
    low: "#388e3c",

    // Camera status colors
    cameraActive: "#4caf50",
    cameraInactive: "#757575",
    cameraError: "#f44336",
    cameraMaintenance: "#ff9800",
  },
};

// Dark theme
export const darkTheme = {
  ...MD3DarkTheme,
  ...commonTheme,
  colors: {
    ...MD3DarkTheme.colors,
    primary: "#90caf9",
    primaryContainer: "#1565c0",
    secondary: "#f48fb1",
    secondaryContainer: "#ad1457",
    tertiary: "#ce93d8",
    tertiaryContainer: "#7b1fa2",
    surface: "#1e1e1e",
    surfaceVariant: "#2d2d2d",
    background: "#121212",
    error: "#f44336",
    errorContainer: "#5d1a1a",
    onPrimary: "#000000",
    onSecondary: "#000000",
    onTertiary: "#000000",
    onSurface: "#ffffff",
    onSurfaceVariant: "#cccccc",
    onBackground: "#ffffff",
    outline: "#666666",
    outlineVariant: "#444444",
    shadow: "#000000",
    scrim: "#000000",

    // Custom colors for alerts
    critical: "#ff5252",
    high: "#ff9800",
    medium: "#ffb74d",
    low: "#81c784",

    // Camera status colors
    cameraActive: "#66bb6a",
    cameraInactive: "#9e9e9e",
    cameraError: "#f44336",
    cameraMaintenance: "#ffb74d",
  },
};

// Alert severity colors
export const alertColors = {
  critical: "#d32f2f",
  high: "#f57c00",
  medium: "#ffa000",
  low: "#388e3c",
};

// Camera status colors
export const cameraStatusColors = {
  active: "#4caf50",
  inactive: "#757575",
  error: "#f44336",
  maintenance: "#ff9800",
};

export default { lightTheme, darkTheme };
