from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Signal
from .video_widget import VideoWidget

class MainWindow(QMainWindow):
    preview_toggled = Signal(bool)  # True = start, False = stop
    record_toggled  = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tonu -Edge AI")
        self.setFixedSize(700, 600)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.btn_preview = QPushButton("Start Preview")
        self.btn_record  = QPushButton("Start Recording")
        self.btn_record.setEnabled(False)

        self.video = VideoWidget()

        layout.addWidget(self.btn_preview)
        layout.addWidget(self.btn_record)
        layout.addWidget(self.video)

        self._is_previewing = False
        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_record.clicked.connect(self._toggle_record)

    def _toggle_preview(self):
        self._is_previewing = not self._is_previewing
        self.btn_preview.setText("Stop Preview" if self._is_previewing else "Start Preview")
        self.btn_record.setEnabled(self._is_previewing)
        self.preview_toggled.emit(self._is_previewing)

    def _toggle_record(self):
        start = self.btn_record.text().startswith("Start")
        self.record_toggled.emit(start)

    def video_handle(self) -> int:
        return self.video.native_handle()

    def set_preview_failed(self):
        # helper to revert UI state if pipeline start fails
        if self._is_previewing:
            self._toggle_preview()
