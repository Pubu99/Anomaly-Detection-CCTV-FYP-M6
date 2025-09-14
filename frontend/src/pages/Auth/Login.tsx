/**
 * Login Component
 * ==============
 *
 * Authentication form for the anomaly detection dashboard.
 */

import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  Container,
  InputAdornment,
  IconButton,
} from "@mui/material";
import {
  Visibility,
  VisibilityOff,
  Security,
  Email,
  Lock,
} from "@mui/icons-material";
import { useSelector, useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";

import { RootState } from "../../store/store";
import { loginStart, loginSuccess, loginFailure } from "../../store/store";

const Login: React.FC = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const { loading, error } = useSelector((state: RootState) => state.auth);

  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);

  const handleChange =
    (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
      setFormData((prev) => ({
        ...prev,
        [field]: event.target.value,
      }));
    };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    if (!formData.username || !formData.password) {
      dispatch(loginFailure("Please enter both username and password"));
      return;
    }

    dispatch(loginStart());

    try {
      // Mock authentication - in real app, call API
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Simulate login validation
      if (formData.username === "admin" && formData.password === "password") {
        const mockUser = {
          id: "1",
          username: formData.username,
          email: "admin@anomalydetection.com",
          role: "administrator",
        };

        const mockToken = "mock-jwt-token-" + Date.now();

        dispatch(loginSuccess({ user: mockUser, token: mockToken }));
        navigate("/dashboard");
      } else {
        dispatch(loginFailure("Invalid username or password"));
      }
    } catch (err) {
      dispatch(loginFailure("Login failed. Please try again."));
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding: 2,
      }}
    >
      <Container maxWidth="sm">
        <Card
          sx={{
            maxWidth: 400,
            margin: "0 auto",
            borderRadius: 3,
            boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
          }}
        >
          <CardContent sx={{ p: 4 }}>
            {/* Header */}
            <Box sx={{ textAlign: "center", mb: 4 }}>
              <Box
                sx={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 80,
                  height: 80,
                  borderRadius: "50%",
                  background:
                    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  mb: 2,
                }}
              >
                <Security sx={{ fontSize: 40, color: "white" }} />
              </Box>

              <Typography
                variant="h4"
                component="h1"
                sx={{ fontWeight: "bold", mb: 1 }}
              >
                Anomaly Detection
              </Typography>

              <Typography variant="body1" color="text.secondary">
                Multi-Camera Surveillance System
              </Typography>
            </Box>

            {/* Error Alert */}
            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            {/* Login Form */}
            <Box component="form" onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Username"
                type="text"
                value={formData.username}
                onChange={handleChange("username")}
                margin="normal"
                disabled={loading}
                autoComplete="username"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Email />
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <TextField
                fullWidth
                label="Password"
                type={showPassword ? "text" : "password"}
                value={formData.password}
                onChange={handleChange("password")}
                margin="normal"
                disabled={loading}
                autoComplete="current-password"
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Lock />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        aria-label="toggle password visibility"
                        onClick={togglePasswordVisibility}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{ mb: 3 }}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                disabled={loading}
                sx={{
                  py: 1.5,
                  background:
                    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  "&:hover": {
                    background:
                      "linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)",
                  },
                  mb: 2,
                }}
              >
                {loading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  "Sign In"
                )}
              </Button>
            </Box>

            {/* Demo Credentials */}
            <Box
              sx={{
                mt: 3,
                p: 2,
                backgroundColor: "grey.100",
                borderRadius: 1,
                border: "1px solid",
                borderColor: "grey.300",
              }}
            >
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ fontWeight: "bold" }}
              >
                Demo Credentials:
              </Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                Username: <strong>admin</strong>
              </Typography>
              <Typography variant="body2">
                Password: <strong>password</strong>
              </Typography>
            </Box>
          </CardContent>
        </Card>

        {/* Footer */}
        <Box sx={{ textAlign: "center", mt: 3 }}>
          <Typography variant="body2" sx={{ color: "rgba(255,255,255,0.8)" }}>
            © 2024 Multi-Camera Anomaly Detection System
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Login;
