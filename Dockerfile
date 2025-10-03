FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive

# System deps for GTK3, GI, Cairo, and GStreamer plugins
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-gi python3-gi-cairo \
    gir1.2-gtk-3.0 gir1.2-gdkpixbuf-2.0 gir1.2-pango-1.0 \
    libgtk-3-0 libcairo2 libgirepository1.0-dev \
    gstreamer1.0-tools gstreamer1.0-gl gstreamer1.0-gtk3 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
 && rm -rf /var/lib/apt/lists/*

# Your app
WORKDIR /app
COPY src/ /app/src/

# COCO labels live in the repo (not the model)
COPY labels.json /app/labels.json
ENV LABELS_PATH=/app/labels.json

# (If you have a requirements.txt, install it here)
# RUN pip install -r requirements.txt

# Entrypoint as before
CMD ["python3", "-m", "app.main"]
