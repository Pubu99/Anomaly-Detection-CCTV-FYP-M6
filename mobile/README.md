# Anomaly Detection Mobile App

A React Native mobile application for the AI-powered surveillance anomaly detection system. This app provides real-time monitoring, instant notifications, and comprehensive management of security cameras and alerts.

## Features

### 🔐 Authentication & Security

- Secure login with biometric authentication support
- JWT token-based authentication
- Session management with auto-logout

### 📱 Real-time Monitoring

- Live camera feeds with WebSocket integration
- Instant push notifications for security alerts
- Real-time system status updates
- Multi-camera view support

### 🚨 Alert Management

- Comprehensive alert dashboard with filtering and search
- Alert severity categorization (Low, Medium, High)
- Alert acknowledgment and resolution workflow
- Detailed alert information with evidence viewing

### 📹 Camera Management

- Camera grid and list view modes
- Individual camera status monitoring
- Recording controls and settings
- Camera configuration management
- Storage usage monitoring

### ⚙️ Settings & Preferences

- Dark/Light theme switching
- Notification preferences
- Profile management
- Data export and cache management

### 📊 Analytics & Insights

- System health monitoring
- Alert trend charts
- Camera performance metrics
- Detection confidence analytics

## Technology Stack

### Frontend Framework

- **React Native 0.72.6** - Cross-platform mobile development
- **TypeScript** - Type-safe development
- **React Navigation 6** - Navigation management

### UI Components

- **React Native Paper 5** - Material Design components
- **React Native Vector Icons** - Icon library
- **React Native Chart Kit** - Data visualization

### State Management

- **Redux Toolkit** - Predictable state container
- **React Redux** - React bindings for Redux
- **Redux Persist** - State persistence with AsyncStorage

### Real-time Communication

- **WebSocket** - Real-time data streaming
- **React Native Push Notification** - Local notifications
- **Firebase Cloud Messaging** - Push notifications

### Native Features

- **React Native Permissions** - Device permissions
- **React Native Device Info** - Device information
- **React Native Orientation Locker** - Screen orientation
- **React Native Image Picker** - Camera and gallery access

## Project Structure

```
mobile/
├── src/
│   ├── components/          # Reusable UI components
│   ├── screens/            # Main application screens
│   │   ├── DashboardScreen.tsx
│   │   ├── AlertsScreen.tsx
│   │   ├── CamerasScreen.tsx
│   │   ├── SettingsScreen.tsx
│   │   ├── LoginScreen.tsx
│   │   ├── AlertDetailScreen.tsx
│   │   └── CameraDetailScreen.tsx
│   ├── navigation/         # Navigation configuration
│   │   └── AppNavigator.tsx
│   ├── store/             # Redux store configuration
│   │   └── store.ts
│   ├── services/          # API and utility services
│   │   ├── websocketService.ts
│   │   └── notificationService.ts
│   ├── hooks/             # Custom React hooks
│   │   └── redux.ts
│   ├── theme/             # App theming
│   │   └── theme.ts
│   └── types/             # TypeScript type definitions
├── android/               # Android-specific code
├── ios/                   # iOS-specific code
├── App.tsx               # Main app component
├── index.js              # App entry point
├── package.json          # Dependencies and scripts
└── README.md            # This file
```

## Screens Overview

### 🏠 Dashboard Screen

- System overview with key metrics
- Real-time alert feed
- Camera status summary
- System health indicators
- Alert trend charts
- Emergency contact button

### 🚨 Alerts Screen

- Comprehensive alert list with filtering
- Search functionality
- Alert categorization by severity and status
- Quick action buttons (Acknowledge/Resolve)
- Alert statistics summary

### 📹 Cameras Screen

- Grid and list view modes
- Camera status monitoring
- Recording controls
- Search and filter capabilities
- Camera statistics

### ⚙️ Settings Screen

- User profile management
- Appearance preferences (Dark/Light mode)
- Notification settings
- Data management options
- Support and feedback

### 📋 Alert Detail Screen

- Detailed alert information
- Evidence viewing (images/videos)
- Alert metadata and analytics
- Action buttons for alert management
- Location and camera information

### 🎥 Camera Detail Screen

- Live stream viewing
- Camera controls and settings
- Recording management
- Storage usage monitoring
- Quick actions panel

## Getting Started

### Prerequisites

- Node.js (>=16)
- React Native CLI
- Android Studio (for Android development)
- Xcode (for iOS development)
- CocoaPods (for iOS dependencies)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd mobile
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **iOS Setup** (macOS only)

   ```bash
   cd ios
   pod install
   cd ..
   ```

4. **Android Setup**
   - Ensure Android SDK is installed
   - Create a virtual device or connect a physical device

### Running the App

#### Development Mode

**For Android:**

```bash
npm run android
```

**For iOS:**

```bash
npm run ios
```

**Start Metro bundler:**

```bash
npm start
```

#### Production Build

**Android APK:**

```bash
npm run build:android
```

**iOS Release:**

```bash
npm run build:ios
```

### Environment Configuration

Create a `.env` file in the root directory:

```env
API_BASE_URL=http://localhost:8000
WS_BASE_URL=ws://localhost:8000
FIREBASE_CONFIG_PATH=./firebase-config.json
```

## Key Features Implementation

### Real-time Updates

- WebSocket connection for live data streaming
- Automatic reconnection handling
- Real-time alert notifications
- Live camera status updates

### Offline Support

- Redux Persist for state persistence
- Offline alert queuing
- Cached data viewing
- Sync on reconnection

### Push Notifications

- Firebase Cloud Messaging integration
- Local notification support
- Notification categorization
- Badge count management

### Security Features

- Biometric authentication (fingerprint/face)
- Secure token storage
- Session timeout handling
- Security alert escalation

## API Integration

The mobile app integrates with the backend API through:

### REST API Endpoints

- Authentication: `/auth/login`, `/auth/logout`
- Cameras: `/cameras/`, `/cameras/{id}`
- Alerts: `/alerts/`, `/alerts/{id}`
- Users: `/users/profile`

### WebSocket Connections

- Real-time alerts: `/ws/alerts`
- Camera status: `/ws/cameras`
- System updates: `/ws/system`

## Testing

### Unit Tests

```bash
npm test
```

### Integration Tests

```bash
npm run test:integration
```

### E2E Tests

```bash
npm run test:e2e
```

## Deployment

### Android Play Store

1. Generate signed APK/AAB
2. Upload to Google Play Console
3. Configure app metadata
4. Submit for review

### iOS App Store

1. Archive in Xcode
2. Upload to App Store Connect
3. Configure app information
4. Submit for review

## Performance Optimizations

### Image Optimization

- Lazy loading for camera thumbnails
- Image caching and compression
- Optimized image formats

### Network Optimization

- Request batching
- Caching strategies
- Connection pooling
- Retry mechanisms

### Memory Management

- Component unmounting cleanup
- Image memory management
- WebSocket connection cleanup
- State persistence optimization

## Security Considerations

### Data Protection

- Encrypted local storage
- Secure API communication (HTTPS/WSS)
- Biometric authentication
- Session management

### Privacy Features

- Data export functionality
- Cache clearing options
- Permission management
- Privacy policy compliance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

### Code Style

- Follow TypeScript best practices
- Use ESLint and Prettier
- Write meaningful commit messages
- Document new features

## Troubleshooting

### Common Issues

**Metro bundler issues:**

```bash
npm run reset-cache
```

**Android build issues:**

```bash
npm run clean
cd android && ./gradlew clean
```

**iOS build issues:**

```bash
cd ios && pod install
```

### Debug Mode

Enable debug mode for detailed logging:

1. Shake device or press Cmd+D (iOS) / Cmd+M (Android)
2. Enable "Debug JS Remotely"
3. Open Chrome DevTools

## Support

For technical support or feature requests:

- Create an issue in the project repository
- Check the project documentation
- Review the troubleshooting guide above

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- React Native community for excellent documentation
- Material Design team for design guidelines
- Security research community for anomaly detection insights
- Open source contributors for various libraries used

---

**Version:** 1.0.0  
**Last Updated:** January 2024  
**Minimum React Native Version:** 0.72.0
