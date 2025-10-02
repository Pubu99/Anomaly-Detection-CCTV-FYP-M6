import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import { CssBaseline, Box } from "@mui/material";
import { Provider } from "react-redux";
import { QueryClient, QueryClientProvider } from "react-query";
import { SnackbarProvider } from "notistack";

// Store and theme
import { store } from "./store/store";
import { darkTheme, lightTheme } from "./theme/theme";

// Components
import Layout from "./components/Layout/Layout";
import Dashboard from "./pages/Dashboard/Dashboard";

// For now, let's create simple placeholder components for the other pages
const Cameras = () => <div>Cameras Page</div>;
const Alerts = () => <div>Alerts Page</div>;
const Analytics = () => <div>Analytics Page</div>;
const Settings = () => <div>Settings Page</div>;
const Login = () => <div>Login Page</div>;

const queryClient = new QueryClient();

const AppContent = () => {
  const [isDarkMode, setIsDarkMode] = React.useState(false);
  const theme = React.useMemo(
    () => createTheme(isDarkMode ? darkTheme : lightTheme),
    [isDarkMode]
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3}>
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/cameras" element={<Cameras />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Navigate to="/dashboard" />} />
            </Routes>
          </Layout>
        </Router>
      </SnackbarProvider>
    </ThemeProvider>
  );
};

function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <AppContent />
      </QueryClientProvider>
    </Provider>
  );
}

export default App;
