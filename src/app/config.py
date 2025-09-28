import os

# Camera defaults (can be overridden with environment variables)
DEVICE = os.getenv("CAM_DEVICE", "/dev/video0")
WIDTH = int(os.getenv("CAM_WIDTH", "640"))
HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))

# If you run Wayland and embedding fails, forcing X11 via Qt often helps:
QT_PLATFORM = os.getenv("QT_QPA_PLATFORM", "")  # e.g., "xcb"

def caps_str(width=WIDTH, height=HEIGHT) -> str:
    # keep it simple; add format/framerate later if needed
    return f"video/x-raw,width={width},height={height}"
