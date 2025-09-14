/**
 * Main Layout Component
 * ====================
 *
 * Responsive layout with sidebar navigation, header, and main content area.
 */

import React from "react";
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Avatar,
  Menu,
  MenuItem,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  AccountCircle,
  Dashboard as DashboardIcon,
  Videocam as VideocamIcon,
  Warning as WarningIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Brightness4,
  Brightness7,
  ExitToApp,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";

import { RootState } from "../../store/store";
import {
  toggleSidebar,
  setSidebarOpen,
  toggleDarkMode,
  logout,
  setConnectionStatus,
} from "../../store/store";

const drawerWidth = 280;

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useDispatch();

  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const { sidebarOpen, isDarkMode } = useSelector(
    (state: RootState) => state.ui
  );
  const { user } = useSelector((state: RootState) => state.auth);
  const { connectionStatus } = useSelector((state: RootState) => state.system);
  const { alerts } = useSelector((state: RootState) => state.alerts);

  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  // Calculate unread notifications
  const unreadAlerts = alerts.filter(
    (alert) =>
      alert.status === "active" && ["high", "critical"].includes(alert.severity)
  ).length;

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    dispatch(logout());
    handleProfileMenuClose();
    navigate("/login");
  };

  const handleSidebarToggle = () => {
    if (isMobile) {
      dispatch(setSidebarOpen(!sidebarOpen));
    } else {
      dispatch(toggleSidebar());
    }
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      dispatch(setSidebarOpen(false));
    }
  };

  // Navigation items
  const navigationItems = [
    { text: "Dashboard", icon: <DashboardIcon />, path: "/dashboard" },
    { text: "Cameras", icon: <VideocamIcon />, path: "/cameras" },
    { text: "Alerts", icon: <WarningIcon />, path: "/alerts" },
    { text: "Analytics", icon: <AnalyticsIcon />, path: "/analytics" },
    { text: "Settings", icon: <SettingsIcon />, path: "/settings" },
  ];

  // Connection status indicator
  const getConnectionColor = () => {
    switch (connectionStatus) {
      case "connected":
        return "#4caf50";
      case "connecting":
        return "#ff9800";
      case "disconnected":
        return "#f44336";
      default:
        return "#9e9e9e";
    }
  };

  const drawer = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Box sx={{ p: 2, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{ fontWeight: "bold" }}
        >
          Anomaly Detection
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Multi-Camera System
        </Typography>
      </Box>

      <List sx={{ flexGrow: 1, pt: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => handleNavigation(item.path)}
              sx={{
                minHeight: 48,
                px: 2.5,
                mx: 1,
                mb: 0.5,
                borderRadius: 1,
                "&.Mui-selected": {
                  backgroundColor: theme.palette.primary.main + "20",
                  "&:hover": {
                    backgroundColor: theme.palette.primary.main + "30",
                  },
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: 2,
                  color:
                    location.pathname === item.path
                      ? theme.palette.primary.main
                      : "inherit",
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                  fontSize: "0.9rem",
                  fontWeight: location.pathname === item.path ? 600 : 400,
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />

      <Box sx={{ p: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: getConnectionColor(),
              mr: 1,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {connectionStatus === "connected"
              ? "System Online"
              : connectionStatus === "connecting"
              ? "Connecting..."
              : "System Offline"}
          </Typography>
        </Box>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: "flex" }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { md: sidebarOpen ? `calc(100% - ${drawerWidth}px)` : "100%" },
          ml: { md: sidebarOpen ? `${drawerWidth}px` : 0 },
          zIndex: theme.zIndex.drawer + 1,
          transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle drawer"
            onClick={handleSidebarToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>

          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {navigationItems.find((item) => item.path === location.pathname)
              ?.text || "Dashboard"}
          </Typography>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {/* Theme toggle */}
            <IconButton
              color="inherit"
              onClick={() => dispatch(toggleDarkMode())}
              title={`Switch to ${isDarkMode ? "light" : "dark"} mode`}
            >
              {isDarkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>

            {/* Notifications */}
            <IconButton color="inherit" onClick={() => navigate("/alerts")}>
              <Badge badgeContent={unreadAlerts} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            {/* Profile menu */}
            <IconButton
              size="large"
              edge="end"
              aria-label="account menu"
              aria-controls="profile-menu"
              aria-haspopup="true"
              onClick={handleProfileMenuOpen}
              color="inherit"
            >
              <Avatar sx={{ width: 32, height: 32 }}>
                {user?.username?.charAt(0).toUpperCase() || "U"}
              </Avatar>
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Profile Menu */}
      <Menu
        id="profile-menu"
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        onClick={handleProfileMenuClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: "visible",
            filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.32))",
            mt: 1.5,
            "& .MuiAvatar-root": {
              width: 32,
              height: 32,
              ml: -0.5,
              mr: 1,
            },
          },
        }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        <MenuItem onClick={handleProfileMenuClose}>
          <Avatar /> {user?.username}
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => navigate("/settings")}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <ExitToApp fontSize="small" />
          </ListItemIcon>
          Logout
        </MenuItem>
      </Menu>

      {/* Sidebar */}
      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={sidebarOpen}
        onClose={() => dispatch(setSidebarOpen(false))}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: drawerWidth,
            boxSizing: "border-box",
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
      >
        {drawer}
      </Drawer>

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: "64px", // Height of AppBar
          width: { md: sidebarOpen ? `calc(100% - ${drawerWidth}px)` : "100%" },
          transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout;
