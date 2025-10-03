import sys
import threading
import time

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstVideo, GstApp, GLib

from gs.sinks import choose_sink
from shared.qt_gst_bridge import set_overlay_handle
from app.config import MODEL_PATH, MODEL_INPUT_SIZE

import numpy as np

from detect.onnx_yolov8 import OnnxYoloV8

class CameraPipeline:
    """
    Unified pipeline (preview + detection):
      v4l2src ! capsf ! queue ! tee name=t

      # Display branch
      t. ! queue ! videoconvert ! cairooverlay name=overlay ! <sink>

      # Inference branch (ONNX worker reads here)
      t. ! queue ! videoscale ! videoconvert !
          capsf_inf (video/x-raw,format=RGB,width=MODEL_INPUT_SIZE,height=MODEL_INPUT_SIZE)
          ! appsink name=appsink
    """
    def __init__(self):
        self.pipeline = None
        self._bus = None
        self._handle_set = False

        # Elements
        self._sink = None
        self._overlay = None
        self._appsink = None

        # Overlay state
        self._disp_w = None
        self._disp_h = None

        # Detection state
        self._detection_enabled = False
        self._dets_lock = threading.Lock()
        self._latest_dets = None  # {"boxes":[(x1,y1,x2,y2)], "scores":[...], "classes":[...]}

        # Worker thread
        self._worker = None
        self._stop_evt = threading.Event()
        self._model = None  # OnnxYoloV8

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
        q_disp    = Gst.ElementFactory.make("queue", "q_disp")
        conv_disp = Gst.ElementFactory.make("videoconvert", "conv_disp")
        overlay   = Gst.ElementFactory.make("cairooverlay", "overlay")
        sink      = choose_sink()

        # Inference branch
        q_inf     = Gst.ElementFactory.make("queue", "q_inf")
        scale_inf = Gst.ElementFactory.make("videoscale", "scale_inf")
        conv_inf  = Gst.ElementFactory.make("videoconvert", "conv_inf")
        capsf_inf = Gst.ElementFactory.make("capsfilter", "capsf_inf")
        appsink   = Gst.ElementFactory.make("appsink", "appsink")

        elems = [source, capsf, q0, tee,
                 q_disp, conv_disp, overlay, sink,
                 q_inf, scale_inf, conv_inf, capsf_inf, appsink]

        if not all(elems):
            print("ERROR: element creation failed")
            self._teardown_on_error()
            return False

        # Configure source caps (from UI env)
        source.set_property("device", device)
        caps = Gst.Caps.from_string(caps_str)  # e.g., video/x-raw,width=640,height=480
        capsf.set_property("caps", caps)

        # Inference caps: square RGB for model
        caps_inf = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={MODEL_INPUT_SIZE},height={MODEL_INPUT_SIZE}"
        )
        capsf_inf.set_property("caps", caps_inf)

        # appsink tuned for realtime
        appsink.set_property("emit-signals", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)

        # Add & link
        for el in elems:
            self.pipeline.add(el)

        if not (source.link(capsf) and capsf.link(q0) and q0.link(tee)):
            print("ERROR: linking source->capsf->q0->tee failed")
            self._teardown_on_error()
            return False

        if not (tee.link(q_disp) and q_disp.link(conv_disp) and conv_disp.link(overlay) and overlay.link(sink)):
            print("ERROR: linking display branch failed")
            self._teardown_on_error()
            return False

        if not (tee.link(q_inf) and q_inf.link(scale_inf) and scale_inf.link(conv_inf) and
                conv_inf.link(capsf_inf) and capsf_inf.link(appsink)):
            print("ERROR: linking inference branch failed")
            self._teardown_on_error()
            return False

        # Keep refs
        self._sink = sink
        self._overlay = overlay
        self._appsink = appsink

        # Overlay callbacks
        overlay.connect("draw", self._on_overlay_draw)
        overlay.connect("caps-changed", self._on_overlay_caps_changed)

        # Bus
        self._bus = self.pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)
        self._bus.connect("message::warning", self._on_warning)
        self._bus.connect("message::eos", self._on_eos)

        # Embed sink
        self._bus.enable_sync_message_emission()
        self._bus.connect("sync-message::element", self._on_sync_message, window_handle)
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
        # Stop worker first to avoid reading from a dying appsink
        self._stop_worker()

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
        enable = bool(enable)
        if enable == self._detection_enabled:
            print(f"Object detection already {'ENABLED' if enable else 'DISABLED'}")
            return

        self._detection_enabled = enable
        if enable:
            # Lazy-load model on first enable
            if self._model is None:
                try:
                    self._model = OnnxYoloV8(MODEL_PATH, MODEL_INPUT_SIZE)
                    print(f"ONNX model loaded: {MODEL_PATH} (input={MODEL_INPUT_SIZE})")
                except Exception as e:
                    print(f"ERROR loading ONNX model: {e}")
                    self._detection_enabled = False
                    return
            self._start_worker()
            print("Object detection ENABLED")
        else:
            self._stop_worker()
            print("Object detection DISABLED")

    # -------- Worker management --------
    def _start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop_evt.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="onnx-worker", daemon=True)
        self._worker.start()

    def _stop_worker(self):
        if not self._worker:
            return
        self._stop_evt.set()
        self._worker.join(timeout=1.5)
        self._worker = None

    # -------- Worker loop: pull from appsink -> ONNX -> store dets --------
    def _worker_loop(self):
        appsink = self._appsink
        if appsink is None:
            return
        # Pull samples non-blocking-ish
        while not self._stop_evt.is_set():
            try:
                # Try to pull a sample with a short timeout (ns)
                sample = appsink.try_pull_sample(10000000)  # 10ms
                if sample is None:
                    continue
                buf = sample.get_buffer()
                caps = sample.get_caps()
                s = caps.get_structure(0)
                w = s.get_value('width')
                h = s.get_value('height')
                # Map buffer to numpy
                success, mapinfo = buf.map(Gst.MapFlags.READ)
                if not success:
                    continue
                try:
                    frame = np.frombuffer(mapinfo.data, dtype=np.uint8)
                    frame = frame.reshape((h, w, 3))  # RGB from capsf_inf
                    # Run model (expects square RGB already)
                    dets_sq = self._model.infer_rgb_square(frame, conf_thres=0.25, iou_thres=0.45, top_k=300)
                    # Convert model-space boxes (input_size×input_size) to display-space (self._disp_w/h)
                    with self._dets_lock:
                        self._latest_dets = {
                            "boxes": dets_sq["boxes"],      # still in square coords; scale in draw
                            "scores": dets_sq["scores"],
                            "classes": dets_sq["classes"],
                        }
                finally:
                    buf.unmap(mapinfo)
            except Exception as e:
                # Keep worker resilient
                print(f"[worker] warning: {e}")
                time.sleep(0.005)

    # -------- Overlay callbacks --------
    def _on_overlay_caps_changed(self, overlay, caps):
        # Track display size for scaling
        s = caps.get_structure(0)
        self._disp_w = s.get_value('width')
        self._disp_h = s.get_value('height')

    def _on_overlay_draw(self, overlay, context, timestamp, duration):
        # Draw only if detection enabled and we have detections
        if not self._detection_enabled:
            return
        with self._dets_lock:
            dets = self._latest_dets
        if not dets or not dets.get("boxes"):
            return

        # If we don't yet know display size, skip drawing this frame
        if not self._disp_w or not self._disp_h:
            return

        # Scale from model square to display aspect
        inp = float(MODEL_INPUT_SIZE)
        sx = self._disp_w / inp
        sy = self._disp_h / inp

        # Cairo drawing
        try:
            # Outline
            context.set_line_width(2.0)
            for (x1, y1, x2, y2), score, cls_id in zip(dets["boxes"], dets["scores"], dets["classes"]):
                # Scale
                X1 = x1 * sx; Y1 = y1 * sy; X2 = x2 * sx; Y2 = y2 * sy
                w = max(0.0, X2 - X1); h = max(0.0, Y2 - Y1)
                # Box
                context.set_source_rgb(0.0, 1.0, 0.0)  # green box
                context.rectangle(X1, Y1, w, h)
                context.stroke()
                # Optional: score label (simple)
                label = f"{int(cls_id)}:{score:.2f}"
                context.move_to(X1 + 3, Y1 + 12)
                context.show_text(label)
                context.stroke()
        except Exception as e:
            # Don't crash drawing on occasional errors
            print(f"[overlay] draw error: {e}")

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
