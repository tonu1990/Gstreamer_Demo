from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Signal
from .video_widget import VideoWidget

class MainWindow(QMainWindow):
    preview_toggled   = Signal(bool)  # True = start, False = stop
    detection_toggled = Signal(bool)  # True = start OD, False = stop OD

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tonu -Edge AI")
        self.setFixedSize(700, 600)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.btn_preview   = QPushButton("Start Preview")
        self.btn_detection = QPushButton("Start Object Detection")
        self.btn_detection.setEnabled(False)

        self.video = VideoWidget()

        layout.addWidget(self.btn_preview)
        layout.addWidget(self.btn_detection)
        layout.addWidget(self.video)

        self._is_previewing = False
        self._is_detecting  = False

        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_detection.clicked.connect(self._toggle_detection)

    def _toggle_preview(self):
        self._is_previewing = not self._is_previewing
        self.btn_preview.setText("Stop Preview" if self._is_previewing else "Start Preview")
        self.btn_detection.setEnabled(self._is_previewing)
        self.preview_toggled.emit(self._is_previewing)

        # If preview just stopped, also force-stop detection UI/state
        if not self._is_previewing and self._is_detecting:
            self._is_detecting = False
            self.btn_detection.setText("Start Object Detection")
            self.detection_toggled.emit(False)

    def _toggle_detection(self):
        if not self._is_previewing:
            return
        self._is_detecting = not self._is_detecting
        self.btn_detection.setText("Stop Object Detection" if self._is_detecting else "Start Object Detection")
        self.detection_toggled.emit(self._is_detecting)

    def video_handle(self) -> int:
        return self.video.native_handle()

    def set_preview_failed(self):
        # helper to revert UI state if pipeline start fails
        if self._is_previewing:
            self._toggle_preview()
