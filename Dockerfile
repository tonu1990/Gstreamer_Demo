FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive

# System deps: Python, GStreamer (with cairooverlay + GL + X sinks), X11 libs, and **PyCairo bindings**
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-gi gobject-introspection libgirepository-1.0-1 gir1.2-glib-2.0 \
    # --- GStreamer core + plugins (cairooverlay is in -plugins-good) ---
    libgstreamer1.0-0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-gl \
    # --- X11 / GL bits for sinks like (glimagesink/ximagesink) ---
    libx11-6 libx11-xcb1 libxext6 libxrender1 \
    libxau6 libxdmcp6 libxcb1 libxcb-util1 \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render0 libxcb-render-util0 libxcb-shm0 libxcb-xinerama0 \
    libxcb-shape0 libxcb-sync1 libxcb-xfixes0 libxcb-randr0 \
    libxkbcommon0 libxkbcommon-x11-0 libxfixes3 libxi6 libxcomposite1 \
    libxrandr2 libfontconfig1 libxdamage1 libxcursor1 \
    libsm6 libice6 libnss3 \
    fontconfig fonts-dejavu-core \
    libegl1 libgles2 libdrm2 \
    # --- *** CRITICAL: PyCairo for cairooverlay draw() in Python *** ---
    python3-gi-cairo python3-cairo libcairo2 \
 && rm -rf /var/lib/apt/lists/*

# Font cache (helps PySide/Qt text rendering)
RUN mkdir -p /var/cache/fontconfig /usr/share/fonts /tmp/.cache/fontconfig && \
    chmod -R 777 /var/cache/fontconfig /tmp/.cache && \
    fc-cache -f -v

WORKDIR /app

# Python deps (PySide6 etc.)
COPY requirements.txt .
RUN python3 -m venv /opt/venv --system-site-packages && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# App source
COPY src/ ./src/

# Runtime env
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    QT_QPA_PLATFORM="xcb" \
    QT_X11_NO_MITSHM="1" \
    PYTHONUNBUFFERED="1" \
    LANG="C.UTF-8"

# Default entry
CMD ["python3", "-m", "app.main"]
