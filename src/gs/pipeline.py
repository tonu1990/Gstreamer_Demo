import logging
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
from pathlib import Path

from detect.onnx_yolov8 import OnnxYoloV8

log = logging.getLogger(__name__)

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

        # Overlay/display size
        self._disp_w = None
        self._disp_h = None

        # Detection state
        self._detection_enabled = False
        self._dets_lock = threading.Lock()
        self._latest_dets = None  # {"boxes":[(x1,y1,x2,y2)], "scores":[...], "classes":[...]}

        # Worker
        self._worker = None
        self._stop_evt = threading.Event()
        self._model = None

        # Debug counters
        self._frames_pulled = 0
        self._frames_with_dets = 0
        self._last_stat_ts = time.time()

    # -------- Pipeline build --------
    def build(self, device: str, caps_str: str, window_handle: int) -> bool:
        if self.pipeline:
            self.stop()

        self.pipeline = Gst.Pipeline.new("camera-pipeline")

        source = Gst.ElementFactory.make("v4l2src", "source")
        capsf  = Gst.ElementFactory.make("capsfilter", "capsf")
        q0     = Gst.ElementFactory.make("queue", "q0")
        tee    = Gst.ElementFactory.make("tee", "tee")

        q_disp    = Gst.ElementFactory.make("queue", "q_disp")
        conv_disp = Gst.ElementFactory.make("videoconvert", "conv_disp")
        overlay   = Gst.ElementFactory.make("cairooverlay", "overlay")
        sink      = choose_sink()

        q_inf     = Gst.ElementFactory.make("queue", "q_inf")
        scale_inf = Gst.ElementFactory.make("videoscale", "scale_inf")
        conv_inf  = Gst.ElementFactory.make("videoconvert", "conv_inf")
        capsf_inf = Gst.ElementFactory.make("capsfilter", "capsf_inf")
        appsink   = Gst.ElementFactory.make("appsink", "appsink")

        elems = [source, capsf, q0, tee,
                 q_disp, conv_disp, overlay, sink,
                 q_inf, scale_inf, conv_inf, capsf_inf, appsink]

        if not all(elems):
            log.error("Element creation failed")
            self._teardown_on_error()
            return False

        source.set_property("device", device)
        caps = Gst.Caps.from_string(caps_str)
        capsf.set_property("caps", caps)

        caps_inf = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={MODEL_INPUT_SIZE},height={MODEL_INPUT_SIZE}"
        )
        capsf_inf.set_property("caps", caps_inf)

        appsink.set_property("emit-signals", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)
        appsink.set_property("sync", False)

        for el in elems:
            self.pipeline.add(el)

        if not (source.link(capsf) and capsf.link(q0) and q0.link(tee)):
            log.error("Linking source->capsf->q0->tee failed")
            self._teardown_on_error()
            return False

        if not (tee.link(q_disp) and q_disp.link(conv_disp) and conv_disp.link(overlay) and overlay.link(sink)):
            log.error("Linking display branch failed")
            self._teardown_on_error()
            return False

        if not (tee.link(q_inf) and q_inf.link(scale_inf) and scale_inf.link(conv_inf) and
                conv_inf.link(capsf_inf) and capsf_inf.link(appsink)):
            log.error("Linking inference branch failed")
            self._teardown_on_error()
            return False

        self._sink = sink
        self._overlay = overlay
        self._appsink = appsink

        overlay.connect("draw", self._on_overlay_draw)
        overlay.connect("caps-changed", self._on_overlay_caps_changed)

        self._bus = self.pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_error)
        self._bus.connect("message::warning", self._on_warning)
        self._bus.connect("message::eos", self._on_eos)

        self._bus.enable_sync_message_emission()
        self._bus.connect("sync-message::element", self._on_sync_message, window_handle)
        if set_overlay_handle(sink, window_handle):
            self._handle_set = True

        log.info("Pipeline built (preview + detection branches ready)")
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
            log.error("Failed to start pipeline")
            self.stop()
            return False
        log.info("Camera preview started")
        return True

    def stop(self):
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
        log.info("Camera preview stopped")

    def set_detection_enabled(self, enable: bool):
        enable = bool(enable)
        if enable == self._detection_enabled:
            log.info(f"Detection already {'ENABLED' if enable else 'DISABLED'}")
            return

        self._detection_enabled = enable
        if enable:
            # Sanity: model path must exist
            if not Path(MODEL_PATH).exists():
                log.error(f"MODEL_PATH not found: {MODEL_PATH}")
                self._detection_enabled = False
                return

            if self._model is None:
                try:
                    self._model = OnnxYoloV8(MODEL_PATH, MODEL_INPUT_SIZE, apply_sigmoid=False)
                    log.info(f"ONNX model loaded: {MODEL_PATH} (input={MODEL_INPUT_SIZE})")
                except Exception as e:
                    log.exception(f"Error loading ONNX model: {e}")
                    self._detection_enabled = False
                    return
            self._start_worker()
            log.info("Object detection ENABLED")
        else:
            self._stop_worker()
            log.info("Object detection DISABLED")

    # -------- Worker management --------
    def _start_worker(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop_evt.clear()
        self._frames_pulled = 0
        self._frames_with_dets = 0
        self._last_stat_ts = time.time()
        self._worker = threading.Thread(target=self._worker_loop, name="onnx-worker", daemon=True)
        self._worker.start()
        log.info("ONNX worker thread started")

    def _stop_worker(self):
        if not self._worker:
            return
        self._stop_evt.set()
        self._worker.join(timeout=2.0)
        self._worker = None
        log.info("ONNX worker thread stopped")

    # -------- Worker loop --------
    def _worker_loop(self):
        appsink = self._appsink
        if appsink is None:
            log.error("Appsink is None; worker exiting")
            return

        while not self._stop_evt.is_set():
            try:
                sample = appsink.try_pull_sample(10000000)  # 10 ms
                if sample is None:
                    self._report_stats()
                    continue

                buf = sample.get_buffer()
                caps = sample.get_caps()
                s = caps.get_structure(0)
                w = s.get_value('width')
                h = s.get_value('height')

                success, mapinfo = buf.map(Gst.MapFlags.READ)
                if not success:
                    continue
                try:
                    frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))  # RGB
                    self._frames_pulled += 1

                    # Inference
                    dets_sq = self._model.infer_rgb_square(frame, conf_thres=0.15, iou_thres=0.45, top_k=300)
                    n = len(dets_sq["boxes"])
                    if n > 0:
                        self._frames_with_dets += 1

                    with self._dets_lock:
                        self._latest_dets = {
                            "boxes": dets_sq["boxes"],
                            "scores": dets_sq["scores"],
                            "classes": dets_sq["classes"],
                        }
                finally:
                    buf.unmap(mapinfo)

                self._report_stats()
            except Exception as e:
                log.warning(f"Worker warning: {e}")
                time.sleep(0.01)

    def _report_stats(self):
        now = time.time()
        if now - self._last_stat_ts >= 2.0:
            if self._frames_pulled:
                ratio = (self._frames_with_dets / self._frames_pulled) * 100.0
            else:
                ratio = 0.0
            log.info(f"[worker] pulled={self._frames_pulled}, frames_with_dets={self._frames_with_dets} ({ratio:.1f}%)")
            self._last_stat_ts = now

    # -------- Overlay callbacks --------
    def _on_overlay_caps_changed(self, overlay, caps):
        s = caps.get_structure(0)
        self._disp_w = s.get_value('width')
        self._disp_h = s.get_value('height')
        log.info(f"Overlay caps-changed: {self._disp_w}x{self._disp_h}")

    def _on_overlay_draw(self, overlay, context, timestamp, duration):
        # Always draw a tiny watermark when detection mode is ON (sanity check)
        if self._detection_enabled:
            try:
                context.set_source_rgb(1.0, 0.0, 0.0)  # red
                context.set_line_width(1.5)
                context.rectangle(6, 6, 90, 20)
                context.stroke()
                context.move_to(10, 20)
                context.show_text("DETECT ON")
                context.stroke()
            except Exception as e:
                log.debug(f"Overlay watermark error: {e}")

        # Draw detections (if any)
        with self._dets_lock:
            dets = self._latest_dets

        if not self._detection_enabled or not dets or not dets.get("boxes"):
            return
        if not self._disp_w or not self._disp_h:
            return

        inp = float(MODEL_INPUT_SIZE)
        sx = self._disp_w / inp
        sy = self._disp_h / inp

        try:
            context.set_line_width(2.0)
            for (x1, y1, x2, y2), score, cls_id in zip(dets["boxes"], dets["scores"], dets["classes"]):
                X1 = x1 * sx; Y1 = y1 * sy; X2 = x2 * sx; Y2 = y2 * sy
                w = max(0.0, X2 - X1); h = max(0.0, Y2 - Y1)

                context.set_source_rgb(0.0, 1.0, 0.0)  # green box
                context.rectangle(X1, Y1, w, h)
                context.stroke()

                # label
                try:
                    context.move_to(X1 + 3, max(12.0, Y1 + 12))
                    context.show_text(f"{int(cls_id)}:{score:.2f}")
                    context.stroke()
                except Exception:
                    pass
        except Exception as e:
            log.debug(f"Overlay draw error: {e}")

    # -------- Bus callbacks --------
    def _on_error(self, bus, msg):
        err, debug = msg.parse_error()
        log.error(f"GStreamer ERROR: {err.message}")
        if debug:
            log.error(f"Debug: {debug}")
        if "not-negotiated" in (err.message or "").lower():
            log.error("Hint: try explicit format/framerate (e.g., YUY2/MJPEG @ 30fps)")
        self.stop()

    def _on_warning(self, bus, msg):
        warn, debug = msg.parse_warning()
        log.warning(f"GStreamer WARNING: {warn.message}")
        if debug:
            log.warning(f"Debug: {debug}")

    def _on_eos(self, bus, msg):
        log.info("End of stream")
        self.stop()

    def _on_sync_message(self, bus, msg, window_handle: int):
        s = msg.get_structure()
        if s and s.get_name() == "prepare-window-handle":
            if not self._handle_set and self._sink and msg.src == self._sink:
                if set_overlay_handle(msg.src, window_handle):
                    self._handle_set = True
                    log.info("VideoOverlay handle set (prepare-window-handle)")
