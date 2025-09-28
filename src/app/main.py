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

    ui.preview_toggled.connect(on_preview)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
