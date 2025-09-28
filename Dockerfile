# Project_Gstreamer_2/Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps for GStreamer, PyGObject (gi), and Qt GUI over X11/XWayland
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-gl \
    libx11-6 libxext6 libxrender1 libxcb1 libxkbcommon-x11-0 \
    libgl1 libglib2.0-0 libdbus-1-3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only PySide6 via pip; gi/GStreamer come from apt (more reliable on ARM)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your source
COPY src/ ./src/

# Make sure Python can import both your src and Debian's dist-packages (for gi)
ENV PYTHONPATH=/usr/lib/python3/dist-packages:/app/src \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    PYTHONUNBUFFERED=1

# Launch the app
CMD ["python", "-m", "app.main"]
