# Multi-Camera Anomaly Detection Dashboard

A modern React.js dashboard for the multi-camera anomaly detection system with real-time monitoring, alert management, and system configuration.

## 🚀 Features

### Core Features

- **Real-time Dashboard** - Live system statistics, camera status, and alert monitoring
- **Camera Management** - Configure, monitor, and control multiple cameras
- **Alert System** - Real-time alerts with severity levels and emergency notifications
- **Analytics Dashboard** - Performance metrics, trends, and system analytics
- **User Authentication** - Secure login and session management
- **Responsive Design** - Works on desktop, tablet, and mobile devices

### Technical Features

- **Real-time Updates** - WebSocket integration for live data
- **State Management** - Redux Toolkit for centralized state
- **Material-UI** - Modern, accessible UI components
- **TypeScript** - Type-safe development
- **Dark/Light Mode** - User preference theme switching
- **Progressive Web App** - Offline capabilities and installable

## 🛠 Technology Stack

- **Frontend Framework:** React 18 with TypeScript
- **UI Library:** Material-UI (MUI) v5
- **State Management:** Redux Toolkit with Redux Persist
- **Routing:** React Router v6
- **Data Fetching:** React Query
- **Real-time:** Socket.io Client
- **Charts:** Recharts + MUI X Charts
- **Forms:** Formik with Yup validation
- **Notifications:** Notistack
- **Build Tool:** Create React App
- **Package Manager:** npm

## 📁 Project Structure

```
frontend/
├── public/                 # Public assets
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── Layout/       # Main layout component
│   │   ├── Camera/       # Camera-related components
│   │   ├── Alert/        # Alert components
│   │   └── Common/       # Shared components
│   ├── pages/            # Page components
│   │   ├── Dashboard/    # Main dashboard
│   │   ├── Cameras/      # Camera management
│   │   ├── Alerts/       # Alert management
│   │   ├── Analytics/    # Analytics dashboard
│   │   ├── Settings/     # System settings
│   │   └── Auth/         # Authentication pages
│   ├── store/            # Redux store configuration
│   ├── api/              # API integration
│   ├── hooks/            # Custom React hooks
│   ├── utils/            # Utility functions
│   ├── types/            # TypeScript type definitions
│   ├── theme/            # Material-UI theme configuration
│   └── constants/        # Application constants
├── package.json
├── tsconfig.json
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js 16.x or higher
- npm 8.x or higher

### Installation

1. **Navigate to frontend directory:**

   ```bash
   cd frontend
   ```

2. **Install dependencies:**

   ```bash
   npm install
   ```

3. **Start development server:**

   ```bash
   npm start
   ```

4. **Open browser:**
   Navigate to `http://localhost:3000`

### Demo Credentials

- **Username:** admin
- **Password:** password

## 📱 Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run test suite
- `npm run lint` - Lint code
- `npm run lint:fix` - Fix linting issues
- `npm run format` - Format code with Prettier

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_VERSION=1.0.0
REACT_APP_TITLE=Anomaly Detection Dashboard
```

### Theme Customization

The dashboard supports custom themes. Edit `src/theme/theme.ts`:

```typescript
export const lightTheme: ThemeOptions = {
  palette: {
    primary: { main: "#1976d2" },
    secondary: { main: "#dc004e" },
    // ... other colors
  },
};
```

## 📊 Dashboard Features

### Real-time Dashboard

- System statistics overview
- Camera status grid
- Recent alerts feed
- Performance metrics
- Inference control (start/stop)

### Camera Management

- Camera configuration
- Status monitoring
- Position mapping
- Performance metrics
- Live video feeds

### Alert System

- Real-time alert notifications
- Severity-based filtering
- Acknowledge/resolve actions
- Emergency contact integration
- Alert history and analytics

### Analytics

- Detection accuracy metrics
- Performance trends
- Camera efficiency analysis
- System health monitoring
- False positive tracking

## 🔄 Real-time Updates

The dashboard uses WebSocket connections for real-time updates:

```typescript
// WebSocket connection for real-time alerts
const socket = io(process.env.REACT_APP_WS_URL);

socket.on("new_alert", (alert) => {
  dispatch(addAlert(alert));
  showNotification("New alert detected!");
});
```

## 🎨 UI Components

### Custom Components

- `StatCard` - Dashboard statistics display
- `CameraGrid` - Camera status overview
- `AlertList` - Real-time alert feed
- `PerformanceChart` - System metrics visualization
- `AlertDialog` - Alert details modal

### Material-UI Usage

- Consistent design system
- Responsive grid layout
- Dark/light theme support
- Accessibility compliance
- Mobile-first design

## 🔐 Security

- JWT token authentication
- Secure API communication
- XSS protection
- CSRF protection
- Input validation
- Error boundary handling

## 📱 Mobile Support

The dashboard is fully responsive and includes:

- Mobile-optimized navigation
- Touch-friendly controls
- Adaptive layouts
- PWA capabilities
- Offline support

## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

## 🚀 Deployment

### Production Build

```bash
npm run build
```

### Docker Deployment

```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

### Nginx Configuration

```nginx
server {
    listen 80;
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }
}
```

## 🔧 Troubleshooting

### Common Issues

1. **WebSocket connection fails:**

   - Check backend server is running
   - Verify REACT_APP_WS_URL is correct
   - Check firewall/proxy settings

2. **Authentication issues:**

   - Clear browser storage
   - Check JWT token expiration
   - Verify API endpoint

3. **Build fails:**
   - Clear node_modules and reinstall
   - Check Node.js version
   - Verify all dependencies

## 📈 Performance Optimization

- Code splitting with React.lazy()
- Memoization with React.memo()
- Virtual scrolling for large lists
- Image optimization
- Bundle size analysis
- Service worker caching

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [Backend API](../backend/) - FastAPI backend service
- [Mobile App](../mobile/) - React Native mobile application
- [AI Models](../src/models/) - Machine learning components

## 📞 Support

For support and questions:

- Create an issue in the project repository
- Check the project documentation
- Review the troubleshooting guide in the docs
