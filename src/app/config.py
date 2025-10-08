import os

# Camera defaults (can be overridden with environment variables)
DEVICE = os.getenv("CAM_DEVICE", "/dev/video0")
WIDTH = int(os.getenv("CAM_WIDTH", "640"))
HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))

# If you run Wayland and embedding fails, forcing X11 via Qt often helps:
QT_PLATFORM = os.getenv("QT_QPA_PLATFORM", "")  # e.g., "xcb"

# Object detection model config (mounted at runtime on the Pi)
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/current.onnx")
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "640"))

#optional labels (defaults to /models/labels.json)
LABELS_PATH = os.getenv("LABELS_PATH", f"{MODEL_DIR}/labels.json")
SHOW_CLASS_NAMES = os.getenv("SHOW_CLASS_NAMES", "1") == "1"

def caps_str(width=WIDTH, height=HEIGHT) -> str:
    # keep it simple; add format/framerate later if needed
    return f"video/x-raw,width={width},height={height}"
