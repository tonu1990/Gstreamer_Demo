FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-gi gobject-introspection libgirepository-1.0-1 gir1.2-glib-2.0 \
    libgstreamer1.0-0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-gl \
    # Essential X11 and XCB libraries
    libx11-6 libx11-xcb1 libxext6 libxrender1 libxcb1 \
    libxkbcommon-x11-0 libxkbcommon0 libxfixes3 libxi6 \
    libxcomposite1 libxrandr2 libfontconfig1 libxdamage1 \
    # XCB libraries - critical for Qt xcb plugin
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render0 libxcb-render-util0 libxcb-shm0 libxcb-xinerama0 \
    libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-randr0 \
    # Additional Qt dependencies
    libnss3 libxss1 libasound2 \
    # Font support
    fonts-dejavu-core fontconfig \
    # OpenGL/EGL support for Pi
    libegl1 libgles2 libdrm2 \
    # Additional libraries that might be needed
    libxcb-util1 libxcb-xkb1 \
 && rm -rf /var/lib/apt/lists/*

# Setup font cache
RUN mkdir -p /var/cache/fontconfig /usr/share/fonts /tmp/.cache/fontconfig && \
    chmod -R 777 /var/cache/fontconfig /tmp/.cache && \
    fc-cache -f -v

WORKDIR /app

# Install Python dependencies in venv with system site packages access
COPY requirements.txt .
RUN python3 -m venv /opt/venv --system-site-packages && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Set runtime environment
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    QT_QPA_PLATFORM="xcb" \
    QT_X11_NO_MITSHM="1" \
    PYTHONUNBUFFERED="1" \
    # Additional Qt environment variables
    QT_DEBUG_PLUGINS="1" \
    QT_QPA_PLATFORM_PLUGIN_PATH="" \
    LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/lib/arm-linux-gnueabihf"

CMD ["python3", "-m", "app.main"]