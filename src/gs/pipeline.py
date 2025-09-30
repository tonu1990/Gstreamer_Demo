import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo

from gs.sinks import choose_sink
from shared.qt_gst_bridge import set_overlay_handle

class CameraPipeline:
    def __init__(self):
        self.pipeline = None
        self._handle_set = False
        self._bus = None

    def build(self, device: str, caps_str: str, window_handle: int) -> bool:
        if self.pipeline:
            self.stop()

        self.pipeline = Gst.Pipeline.new("camera-pipeline")

        source = Gst.ElementFactory.make("v4l2src", "source")
        capsf  = Gst.ElementFactory.make("capsfilter", "capsf")
        conv   = Gst.ElementFactory.make("videoconvert", "convert")
        sink   = choose_sink()

        if not all([self.pipeline, source, capsf, conv, sink]):
            print("ERROR: element creation failed")
            self.pipeline = None
            return False

        source.set_property("device", device)
        caps = Gst.Caps.from_string(caps_str)
        capsf.set_property("caps", caps)

        for el in (source, capsf, conv, sink):
            self.pipeline.add(el)

        if not source.link(capsf) or not capsf.link(conv) or not conv.link(sink):
            print("ERROR: linking elements failed")
            self.pipeline = None
            return False

        # Bus handlers
        self._bus = self.pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)
        self._bus.connect("message::warning", self._on_warning)
        self._bus.connect("message::eos", self._on_eos)

        # Sync messages for prepare-window-handle
        self._bus.enable_sync_message_emission()
        self._bus.connect("sync-message::element", self._on_sync_message, window_handle)

        # Proactive handle set (some sinks accept immediately)
        if set_overlay_handle(sink, window_handle):
            self._handle_set = True

        return True

    def start(self) -> bool:
        if not self.pipeline:
            return False
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: failed to start pipeline")
            self.stop()
            return False
        print("Camera preview started")
        return True

    def stop(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self._bus = None
            self._handle_set = False
            print("Camera preview stopped")

    # Bus callbacks
    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"GStreamer ERROR: {err.message}")
        if debug:
            print(f"Debug: {debug}")
        if "not-negotiated" in err.message.lower():
            print("Hint: try explicit format/framerate (e.g., YUY2/MJPEG @ 30fps)")
        self.stop()

    def _on_warning(self, bus, msg):
        warn, debug = msg.parse_warning()
        print(f"GStreamer WARNING: {warn.message}")
        if debug:
            print(f"Debug: {debug}")

    def _on_eos(self, bus, msg):
        print("End of stream")
        self.stop()

    def _on_sync_message(self, bus, msg, window_handle: int):
        s = msg.get_structure()
        if s and s.get_name() == "prepare-window-handle":
            if not self._handle_set:
                if set_overlay_handle(msg.src, window_handle):
                    self._handle_set = True
                    print("VideoOverlay handle set (prepare-window-handle)")
