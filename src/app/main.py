import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
from gs.pipeline import CameraPipeline
from app.config import DEVICE, caps_str, QT_PLATFORM
from app.logging_setup import init_logging
import logging

def main():
    # Start logging first
    log_path = init_logging("Gstreamer_Demo")
    log = logging.getLogger(__name__)
    log.info("Main starting")
    log.info(f"Log file at: {log_path}")

    if QT_PLATFORM:
        os.environ.setdefault("QT_QPA_PLATFORM", QT_PLATFORM)

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
        if not pipe.pipeline:
            logging.getLogger(__name__).warning("Detection toggle ignored: pipeline not running")
            return
        pipe.set_detection_enabled(start)

    ui.preview_toggled.connect(on_preview)
    ui.detection_toggled.connect(on_detection)

    ui.show()
    return sys.exit(app.exec())

if __name__ == "__main__":
    main()
