# Debian base so python3-gi matches interpreter (avoids _gi issues)
FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive

# System deps: GI + GStreamer + Qt/X11 + venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-gi gobject-introspection libgirepository-1.0-1 gir1.2-glib-2.0 \
    libgstreamer1.0-0 gir1.2-gstreamer-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-gl \
    # X11/Qt runtime libs commonly needed by PySide6
    libx11-6 libxext6 libxrender1 libxcb1 libxkbcommon-x11-0 \
    libxfixes3 libxi6 libxcomposite1 libxrandr2 \
    libgl1 libglib2.0-0 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (PySide6) go into a venv to avoid PEP 668
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/

# Use venv at runtime; also set Qt + logging envs
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    QT_QPA_PLATFORM="xcb" \
    QT_X11_NO_MITSHM="1" \
    PYTHONUNBUFFERED="1"

# Launch GUI
CMD ["python3", "-m", "app.main"]
