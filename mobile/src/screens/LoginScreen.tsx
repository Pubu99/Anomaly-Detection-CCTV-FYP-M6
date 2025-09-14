/**
 * Login Screen
 * ===========
 *
 * Authentication screen for mobile app login.
 */

import React, { useState } from "react";
import {
  View,
  ScrollView,
  StyleSheet,
  Alert,
  Image,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import {
  Card,
  Title,
  Paragraph,
  TextInput,
  Button,
  Text,
  Divider,
  ActivityIndicator,
} from "react-native-paper";

// Hooks
import { useAppDispatch, useAppSelector } from "../hooks/redux";

const LoginScreen: React.FC = () => {
  const dispatch = useAppDispatch();
  const { isLoading } = useAppSelector((state) => state.auth);

  const [credentials, setCredentials] = useState({
    username: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
    if (!credentials.username.trim() || !credentials.password.trim()) {
      Alert.alert("Error", "Please enter both username and password");
      return;
    }

    try {
      // Simulate login API call
      console.log("Logging in with:", credentials);

      // Mock successful login
      setTimeout(() => {
        // dispatch(loginSuccess({ user: { username: credentials.username } }));
        Alert.alert("Success", "Login successful!");
      }, 1000);
    } catch (error) {
      console.error("Login failed:", error);
      Alert.alert("Error", "Login failed. Please try again.");
    }
  };

  const handleBiometricLogin = () => {
    Alert.alert("Biometric Login", "Biometric authentication coming soon!");
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Logo Section */}
        <View style={styles.logoContainer}>
          <View style={styles.logoPlaceholder}>
            <Text style={styles.logoText}>🛡️</Text>
          </View>
          <Title style={styles.appTitle}>Anomaly Detection</Title>
          <Paragraph style={styles.appSubtitle}>
            Intelligent Security Monitoring
          </Paragraph>
        </View>

        {/* Login Form */}
        <Card style={styles.loginCard}>
          <Card.Content>
            <Title style={styles.loginTitle}>Welcome Back</Title>
            <Paragraph style={styles.loginSubtitle}>
              Sign in to your account
            </Paragraph>

            <View style={styles.formContainer}>
              <TextInput
                label="Username"
                value={credentials.username}
                onChangeText={(text) =>
                  setCredentials({ ...credentials, username: text })
                }
                mode="outlined"
                style={styles.input}
                autoCapitalize="none"
                autoCorrect={false}
                disabled={isLoading}
                left={<TextInput.Icon icon="account" />}
              />

              <TextInput
                label="Password"
                value={credentials.password}
                onChangeText={(text) =>
                  setCredentials({ ...credentials, password: text })
                }
                mode="outlined"
                secureTextEntry={!showPassword}
                style={styles.input}
                disabled={isLoading}
                left={<TextInput.Icon icon="lock" />}
                right={
                  <TextInput.Icon
                    icon={showPassword ? "eye-off" : "eye"}
                    onPress={() => setShowPassword(!showPassword)}
                  />
                }
              />

              <Button
                mode="contained"
                onPress={handleLogin}
                style={styles.loginButton}
                disabled={isLoading}
                loading={isLoading}
              >
                {isLoading ? "Signing In..." : "Sign In"}
              </Button>

              <Divider style={styles.divider} />

              <Button
                mode="outlined"
                onPress={handleBiometricLogin}
                style={styles.biometricButton}
                icon="fingerprint"
                disabled={isLoading}
              >
                Use Biometric Login
              </Button>
            </View>
          </Card.Content>
        </Card>

        {/* Footer */}
        <View style={styles.footer}>
          <Paragraph style={styles.footerText}>
            Powered by Advanced AI Technology
          </Paragraph>
          <Text style={styles.versionText}>Version 1.0.0</Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: "center",
    padding: 20,
  },
  logoContainer: {
    alignItems: "center",
    marginBottom: 40,
  },
  logoPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: "#1976d2",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  logoText: {
    fontSize: 32,
    color: "#ffffff",
  },
  appTitle: {
    fontSize: 24,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 4,
  },
  appSubtitle: {
    textAlign: "center",
    opacity: 0.7,
  },
  loginCard: {
    marginBottom: 30,
  },
  loginTitle: {
    textAlign: "center",
    marginBottom: 8,
  },
  loginSubtitle: {
    textAlign: "center",
    opacity: 0.7,
    marginBottom: 24,
  },
  formContainer: {
    gap: 16,
  },
  input: {
    backgroundColor: "transparent",
  },
  loginButton: {
    marginTop: 8,
  },
  divider: {
    marginVertical: 16,
  },
  biometricButton: {
    marginBottom: 8,
  },
  footer: {
    alignItems: "center",
    marginTop: 20,
  },
  footerText: {
    textAlign: "center",
    opacity: 0.6,
    fontSize: 12,
  },
  versionText: {
    textAlign: "center",
    opacity: 0.4,
    fontSize: 10,
    marginTop: 4,
  },
});

export default LoginScreen;
