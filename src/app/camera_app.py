#!/usr/bin/env python3
import os
import sys
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import Qt
from gi.repository import Gst, GstVideo, GLib

# Initialize GStreamer
Gst.init(sys.argv)


class VideoWidget(QWidget):
    """Custom widget to display GStreamer video"""
    def __init__(self):
        super().__init__()
        self.setFixedSize(640, 480)  # Standard camera resolution
        self.setStyleSheet("background-color: black;")

    def get_window_handle(self):
        """Get the native window handle for GStreamer"""
        # Qt returns a WId; cast to int for GStreamer
        return int(self.winId())


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI Camera System - Embedded Video")
        self.setFixedSize(700, 600)  # room for buttons + video

        # GStreamer state
        self.pipeline = None
        self.is_playing = False
        self.video_widget = None
        self._handle_set = False

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface - buttons on top"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Buttons
        self.preview_button = QPushButton("Start Preview")
        self.record_button = QPushButton("Start Recording")
        self.record_button.setEnabled(False)

        layout.addWidget(self.preview_button)
        layout.addWidget(self.record_button)

        # Video area
        self.video_widget = VideoWidget()
        layout.addWidget(self.video_widget)

        # Signals
        self.preview_button.clicked.connect(self.toggle_preview)
        self.record_button.clicked.connect(self.toggle_recording)

    def toggle_preview(self):
        if not self.is_playing:
            self.start_preview()
        else:
            self.stop_preview()

    def _choose_sink(self):
        """
        Try GL first (works well on Pi with GLES), then X11 sinks.
        Returns a created sink element or None.
        """
        for name in ("glimagesink", "ximagesink", "xvimagesink"):
            sink = Gst.ElementFactory.make(name, "sink")
            if sink:
                # Optional: keep aspect off so it fills your widget
                if sink.list_properties():
                    try:
                        sink.set_property("force-aspect-ratio", False)
                    except Exception:
                        pass
                return sink
        return None

    def start_preview(self):
        print("Starting camera preview with proper resolution...")

        self.video_widget.show()
        window_handle = self.video_widget.get_window_handle()
        print(f"Window handle: {window_handle}")

        # Build pipeline
        self.pipeline = Gst.Pipeline.new("camera-pipeline")

        source = Gst.ElementFactory.make("v4l2src", "source")
        caps_filter = Gst.ElementFactory.make("capsfilter", "capsfilter")
        convert = Gst.ElementFactory.make("videoconvert", "convert")
        sink = self._choose_sink()

        if not all([source, caps_filter, convert, sink]):
            print("ERROR: Could not create GStreamer elements")
            self.pipeline = None
            return

        # Configure source/caps
        source.set_property("device", "/dev/video0")
        # Keep caps simple; add format/framerate if needed for your camera
        caps = Gst.Caps.from_string("video/x-raw,width=640,height=480")
        caps_filter.set_property("caps", caps)

        # Assemble pipeline
        for el in (source, caps_filter, convert, sink):
            self.pipeline.add(el)

        if not source.link(caps_filter):
            print("ERROR: Could not link source -> capsfilter")
            return
        if not caps_filter.link(convert):
            print("ERROR: Could not link capsfilter -> convert")
            return
        if not convert.link(sink):
            print("ERROR: Could not link convert -> sink")
            return

        # Bus: errors/warnings/EOS
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self.on_error)
        bus.connect("message::warning", self.on_warning)
        bus.connect("message::eos", self.on_eos)

        # Also handle prepare-window-handle synchronously (more reliable)
        bus.enable_sync_message_emission()
        bus.connect("sync-message::element", self.on_sync_message)

        # Set the window handle proactively (some sinks accept it immediately)
        try:
            GstVideo.VideoOverlay.set_window_handle(sink, window_handle)
            self._handle_set = True
            print("Embedded sink via VideoOverlay (proactive)")
        except Exception as e:
            # Some sinks need the prepare-window-handle message; handled below
            print(f"Proactive VideoOverlay set failed (will retry on prepare-window-handle): {e}")

        # Start pipeline
        result = self.pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Failed to start pipeline")
            self.stop_preview()
            return

        self.is_playing = True
        self.preview_button.setText("Stop Preview")
        self.record_button.setEnabled(True)
        print("Camera preview started successfully")

    def on_sync_message(self, bus, msg):
        """
        Called synchronously from the streaming thread.
        Use this to catch 'prepare-window-handle' and set the overlay handle.
        """
        if msg.get_structure() and msg.get_structure().get_name() == "prepare-window-handle":
            try:
                if not self._handle_set:
                    handle = self.video_widget.get_window_handle()
                    GstVideo.VideoOverlay.set_window_handle(msg.src, handle)
                    self._handle_set = True
                    print("Embedded sink via VideoOverlay (prepare-window-handle)")
            except Exception as e:
                print("Failed to set window handle on prepare-window-handle:", e)

    def stop_preview(self):
        print("Stopping camera preview...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
        self.is_playing = False
        self._handle_set = False
        self.preview_button.setText("Start Preview")
        self.record_button.setEnabled(False)
        print("Camera preview stopped")

    def toggle_recording(self):
        print("Recording functionality will be added after video embedding")

    def on_error(self, bus, message):
        err, debug = message.parse_error()
        print(f"GStreamer ERROR: {err.message}")
        if debug:
            print(f"Debug: {debug}")
        # Try a more explicit caps if negotiation failed
        if "not-negotiated" in err.message.lower():
            print("Hint: Try adding format/framerate, e.g. YUY2/MJPEG @ 30fps")
        self.stop_preview()

    def on_warning(self, bus, message):
        warn, debug = message.parse_warning()
        print(f"GStreamer WARNING: {warn.message}")
        if debug:
            print(f"Debug: {debug}")

    def on_eos(self, bus, message):
        print("End of stream reached")
        self.stop_preview()


def main():
    # If you are on Wayland and using ximagesink/xvimagesink, consider forcing XCB:
    # os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())

 