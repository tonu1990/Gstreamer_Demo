from PySide6.QtWidgets import QWidget

class VideoWidget(QWidget):
    """Widget that hosts the embedded GStreamer sink."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(640, 480)
        self.setStyleSheet("background-color: black;")

    def native_handle(self) -> int:
        return int(self.winId())
