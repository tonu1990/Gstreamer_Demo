FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    # GI + GStreamer
    python3-gi \
    gobject-introspection \
    libgirepository-1.0-1 \
    gir1.2-gstreamer-1.0 \
    libgstreamer1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-gl \
    # X11/GL bits for Qt GUI in container
    libx11-6 libxext6 libxrender1 libxcb1 libxkbcommon-x11-0 \
    libgl1 libglib2.0-0 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Make sure Python sees Debian’s dist-packages (for gi) AND your app
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/app/src \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main"]
