# Use Debian's Python so python3-gi matches the interpreter
FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive

# GI + GStreamer + Qt/X11 runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-gi \
    gobject-introspection libgirepository-1.0-1 gir1.2-glib-2.0 \
    libgstreamer1.0-0 gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-gl \
    libx11-6 libxext6 libxrender1 libxcb1 libxkbcommon-x11-0 \
    libgl1 libglib2.0-0 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Use Debian's python3; no sys/dist-packages juggling needed
ENV PYTHONPATH=/app/src \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1

CMD ["python3", "-m", "app.main"]
