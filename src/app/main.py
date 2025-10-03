import os
import sys
import time
import logging
from datetime import datetime
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from gs.pipeline import CameraPipeline
from app.config import DEVICE, caps_str, QT_PLATFORM

def _setup_logging():
    # Level via env
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # File on Desktop (inside container root’s Desktop, typically bind-mounted)
    desktop = os.path.expanduser("~/Desktop")
    os.makedirs(desktop, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(desktop, f"Gstreamer_Demo_{ts}.log")

    fh = logging.FileHandler(filepath, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    root = logging.getLogger()
    root.info("=== Application logging started ===")
    root.info(f"Log file: {filepath}")
    return filepath

def main():
    logfile = _setup_logging()
    log = logging.getLogger(__name__)
    log.info("Main starting")
    log.info(f"Log file at: {logfile}")

    # Optional: force Qt to X11 if Wayland embedding gives trouble
    if QT_PLATFORM:
        os.environ.setdefault("QT_QPA_PLATFORM", QT_PLATFORM)

    # Init GStreamer early
    Gst.init(sys.argv)

    app = QApplication(sys.argv)
    ui = MainWindow()
    pipe = CameraPipeline()

    def on_preview(toggled: bool):
        if toggled:
            built = pipe.build(device=DEVICE, caps_str=caps_str(), window_handle=ui.video_handle())
            if not built or not pipe.start():
                ui.set_preview_failed()
        else:
            pipe.stop()

    def on_detection(toggled: bool):
        # True = start OD, False = stop OD
        pipe.set_detection_enabled(toggled)

    ui.preview_toggled.connect(on_preview)
    ui.detection_toggled.connect(on_detection)

    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
