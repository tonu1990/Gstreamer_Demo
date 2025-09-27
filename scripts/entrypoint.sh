#!/bin/bash
set -e  # Exit on any error

echo "=== Starting GStreamer PySide6 Application ==="

# Display system information
echo "Python version: $(python3 --version)"
echo "GStreamer version: $(gst-launch-1.0 --version | head -1)"

# Check if display is available
if [ -z "$DISPLAY" ]; then
    echo "WARNING: DISPLAY environment variable not set"
    echo "Setting DISPLAY to :0"
    export DISPLAY=:0
fi

# Check X11 authorization (required for GUI applications in Docker)
if [ ! -f "/home/appuser/.Xauthority" ] && [ -f "/root/.Xauthority" ]; then
    echo "Copying X11 authority file..."
    cp /root/.Xauthority /home/appuser/.Xauthority
    chown appuser:appuser /home/appuser/.Xauthority
fi

# Check if we're running in a terminal or headless
if [ -t 0 ]; then
    echo "Running in terminal mode"
else
    echo "Running in headless mode"
fi

# Check for camera device
if [ -e "/dev/video0" ]; then
    echo "Camera device /dev/video0 detected"
    ls -la /dev/video*
else
    echo "WARNING: No camera device detected at /dev/video0"
    echo "Available video devices:"
    ls -la /dev/video* 2>/dev/null || echo "No video devices found"
fi

# Set GStreamer debug level (adjust as needed)
export GST_DEBUG=2
# export GST_DEBUG=3  # More verbose debugging

echo "=== Starting Application ==="

# Run the Python application
exec python3 main.py "$@"