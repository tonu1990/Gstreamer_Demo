FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# GI + GStreamer (base/good/gl) + minimal X11 + GL/EGL for glimagesink
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi gobject-introspection gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-gl \
    libx11-6 libx11-xcb1 libxext6 libxrender1 libxcb1 \
    libxkbcommon0 libxkbcommon-x11-0 libxi6 libxrandr2 libxcursor1 libxfixes3 \
    libegl1 libgles2 libdrm2 \
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (pin in requirements.txt for reproducibility)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App
COPY src/ ./src/

# Defaults: X11 platform for Qt; allow override with -e QT_QPA_PLATFORM=wayland if you experiment
ENV PYTHONPATH="/app/src" \
    QT_QPA_PLATFORM="xcb" \
    QT_X11_NO_MITSHM="1" \
    PYTHONUNBUFFERED="1" \
    LANG="C.UTF-8"

CMD ["python", "-m", "app.main"]
