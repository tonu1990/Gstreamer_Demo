import sys
import threading

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GstVideo, GLib

from gs.sinks import choose_sink
from shared.qt_gst_bridge import set_overlay_handle

class CameraPipeline:
    """
    Unified pipeline (preview + detection):
      v4l2src ! capsf ! queue ! tee name=t

      # Display branch
      t. ! queue ! videoconvert ! cairooverlay name=overlay ! <sink>

      # Inference branch (for ONNX worker to read from)
      t. ! queue ! videoscale ! videoconvert ! capsf_infer (RGB, WxH) ! appsink name=appsink

    We embed <sink> into the Qt widget via VideoOverlay handle.
    The cairooverlay draw callback will render boxes when detection is enabled.
    """
    def __init__(self):
        self.pipeline = None
        self._bus = None
        self._handle_set = False

        # Elements we keep references to
        self._sink = None
        self._overlay = None
        self._appsink = None

        # State flags
        self._detection_enabled = False

        # Shared detection results (filled in Step 3 by the ONNX worker)
        self._dets_lock = threading.Lock()
        self._latest_dets = None  # e.g., {"boxes":[(x1,y1,x2,y2),...], "labels":[...], "scores":[...]}

    # -------- Pipeline build --------
    def build(self, device: str, caps_str: str, window_handle: int) -> bool:
        if self.pipeline:
            self.stop()

        self.pipeline = Gst.Pipeline.new("camera-pipeline")

        # Source + initial caps
        source = Gst.ElementFactory.make("v4l2src", "source")
        capsf  = Gst.ElementFactory.make("capsfilter", "capsf")
        q0     = Gst.ElementFactory.make("queue", "q0")
        tee    = Gst.ElementFactory.make("tee", "tee")

        # Display branch
        q_disp   = Gst.ElementFactory.make("queue", "q_disp")
        conv_disp= Gst.ElementFactory.make("videoconvert", "conv_disp")
        overlay  = Gst.ElementFactory.make("cairooverlay", "overlay")
        sink     = choose_sink()  # glimagesink preferred; falls back automatically

        # Inference branch
        q_inf    = Gst.ElementFactory.make("queue", "q_inf")
        scale_inf= Gst.ElementFactory.make("videoscale", "scale_inf")
        conv_inf = Gst.ElementFactory.make("videoconvert", "conv_inf")
        capsf_inf= Gst.ElementFactory.make("capsfilter", "capsf_inf")
        appsink  = Gst.ElementFactory.make("appsink", "appsink")

        elems = [source, capsf, q0, tee,
                 q_disp, conv_disp, overlay, sink,
                 q_inf, scale_inf, conv_inf, capsf_inf, appsink]

        if not all(elems):
            print("ERROR: element creation failed")
            self.pipeline = None
            return False

        # Configure source and caps
        source.set_property("device", device)
        caps = Gst.Caps.from_string(caps_str)  # e.g., "video/x-raw,width=640,height=480"
        capsf.set_property("caps", caps)

        # Inference caps (RGB WxH) – final size for the model; height/width set later if needed
        # We'll keep width/height flexible for now; Step 3 can set exact MODEL_INPUT_SIZE via env if desired.
        caps_inf = Gst.Caps.from_string("video/x-raw,format=RGB")
        capsf_inf.set_property("caps", caps_inf)

        # appsink tuned for real-time: don't block the pipeline
        appsink.set_property("emit-signals", False)  # we'll pull in a worker thread later
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)

        # Add to pipeline
        for el in elems:
            self.pipeline.add(el)

        # Link the common path up to tee
        if not (source.link(capsf) and capsf.link(q0) and q0.link(tee)):
            print("ERROR: linking source->capsf->q0->tee failed")
            self._teardown_on_error()
            return False

        # Link display branch: tee -> q_disp -> conv_disp -> overlay -> sink
        if not (tee.link(q_disp) and q_disp.link(conv_disp) and conv_disp.link(overlay) and overlay.link(sink)):
            print("ERROR: linking display branch failed")
            self._teardown_on_error()
            return False

        # Link inference branch: tee -> q_inf -> scale_inf -> conv_inf -> capsf_inf -> appsink
        if not (tee.link(q_inf) and q_inf.link(scale_inf) and scale_inf.link(conv_inf) and
                conv_inf.link(capsf_inf) and capsf_inf.link(appsink)):
            print("ERROR: linking inference branch failed")
            self._teardown_on_error()
            return False

        # Keep references
        self._sink = sink
        self._overlay = overlay
        self._appsink = appsink

        # Connect cairooverlay signals
        # draw(self, overlay, context, timestamp, duration)
        overlay.connect("draw", self._on_overlay_draw)
        # caps-changed is optional; useful if you need to recalc coordinates
        overlay.connect("caps-changed", self._on_overlay_caps_changed)

        # Bus handlers (errors, warnings, eos)
        self._bus = self.pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)
        self._bus.connect("message::warning", self._on_warning)
        self._bus.connect("message::eos", self._on_eos)

        # Sync messages for embedding
        self._bus.enable_sync_message_emission()
        self._bus.connect("sync-message::element", self._on_sync_message, window_handle)

        # Proactive handle set (if sink supports it immediately)
        if set_overlay_handle(sink, window_handle):
            self._handle_set = True

        print("Pipeline built (preview + detection branches ready)")
        return True

    def _teardown_on_error(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None
        self._bus = None
        self._handle_set = False
        self._appsink = None
        self._overlay = None
        self._sink = None

    # -------- Control --------
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
            self._appsink = None
            self._overlay = None
            self._sink = None
            self._detection_enabled = False
            with self._dets_lock:
                self._latest_dets = None
            print("Camera preview stopped")

    def set_detection_enabled(self, enable: bool):
        """
        Toggle detection mode. For now, this only flips the overlay/drawing flag.
        In Step 3, we'll start/stop the ONNX worker that reads from appsink.
        """
        self._detection_enabled = bool(enable)
        print(f"Object detection {'ENABLED' if self._detection_enabled else 'DISABLED'}")

    # -------- Overlay callbacks --------
    def _on_overlay_draw(self, overlay, context, timestamp, duration):
        """
        Cairo drawing callback. 'context' is a cairo.Context targeting the current frame.
        For this step, we draw nothing unless detection is enabled AND we have results.
        Step 3 will populate self._latest_dets.
        """
        if not self._detection_enabled:
            return  # preview mode: no-op

        # No detections yet (Step 3 will fill them)
        with self._dets_lock:
            dets = self._latest_dets

        if not dets:
            return

        # Placeholder: we'll render boxes in Step 3.
        # Keeping this function wired now ensures smooth integration later.
        return

    def _on_overlay_caps_changed(self, overlay, caps):
        # Useful if you later need to recalc scaling for drawing
        pass

    # -------- Bus callbacks --------
    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        print(f"GStreamer ERROR: {err.message}")
        if debug:
            print(f"Debug: {debug}")
        if "not-negotiated" in (err.message or "").lower():
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
            if not self._handle_set and self._sink and msg.src == self._sink:
                if set_overlay_handle(msg.src, window_handle):
                    self._handle_set = True
                    print("VideoOverlay handle set (prepare-window-handle)")
