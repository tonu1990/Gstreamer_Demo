import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from gs.pipeline import CameraPipeline
from app.config import DEVICE, caps_str, QT_PLATFORM

def main():
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

    def on_detection(start: bool):
        # UI already disables the button when preview is off,
        # but keep a safety guard to avoid surprises.
        if not pipe.pipeline:
            print("Detection toggle ignored: preview pipeline not running.")
            return
        pipe.set_detection_enabled(start)

    ui.preview_toggled.connect(on_preview)
    ui.detection_toggled.connect(on_detection)

    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
