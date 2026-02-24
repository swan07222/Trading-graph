# ADR 0002: PyQt6 for Desktop UI

## Status
Accepted

## Date
2024-01-15

## Context
The trading system requires a desktop user interface for:
- Real-time charting and visualization
- Order entry and management
- Portfolio monitoring
- System configuration

We need a UI framework that:
- Provides native desktop experience
- Supports real-time updates
- Has good charting capabilities
- Works on Windows, macOS, and Linux
- Has stable long-term support

## Decision
Use PyQt6 as the desktop UI framework with the following characteristics:

- **PyQt6**: Python bindings for Qt 6
- **pyqtgraph**: High-performance real-time charting
- **QSS styling**: Custom dark theme for trading interface
- **Model-View architecture**: Clean separation of data and presentation

### UI Architecture

```
┌─────────────────────────────────────────┐
│           Main Window (QMainWindow)     │
│  ┌───────────┬───────────┬───────────┐  │
│  │  Market   │   Chart   │   Order   │  │
│  │  Watch    │  Widget   │   Panel   │  │
│  │  Widget   │           │           │  │
│  └───────────┴───────────┴───────────┘  │
│  ┌─────────────────────────────────────┐│
│  │        Portfolio & Positions        ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

## Consequences

### Positive
- Native look and feel across platforms
- Excellent charting performance with pyqtgraph
- Rich widget ecosystem
- Strong typing support
- Active community and documentation

### Negative
- Larger application bundle size
- Steeper learning curve than web frameworks
- Less flexible for remote access scenarios

### Mitigation
- Provide web-based monitoring as future enhancement
- Use Qt's theming system for consistent branding
- Implement proper threading to avoid UI blocking
