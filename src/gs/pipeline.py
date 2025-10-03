# src/gs/pipeline.py
import os
import json
import logging
import threading
from collections import deque
from typing import List, Tuple, Optional

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GstVideo, GObject  # type: ignore
import cairo

import numpy as np

# Your ONNX runtime wrapper (same one that produced the earlier logs)
from detect.onnx_yolov8 import OnnxYoloV8  # noqa: E402

log = logging.getLogger("gs.pipeline")

# ---------- Labels loader ----------
def load_labels(path: str = "/app/labels.json") -> Optional[List[str]]:
    """
    Load labels from labels.json with schema you provided.
    Returns list such that labels[id] -> name, or None if unavailable.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cls_map = data.get("classes")
        labels: Optional[List[str]] = None
        if isinstance(cls_map, dict):
            max_id = max(int(k) for k in cls_map.keys()) if cls_map else -1
            labels = [cls_map.get(str(i), str(i)) for i in range(max_id + 1)]
        elif isinstance(cls_map, list):
            labels = [str(x) for x in cls_map]
        if labels:
            log.info("Loaded %d class labels from %s", len(labels), path)
            return labels
        log.warning("labels.json loaded but no valid 'classes' found; will show IDs only")
        return None
    except FileNotFoundError:
        log.warning("labels.json not found at %s (labels will be IDs only)", path)
        return None
    except Exception as e:
        log.warning("labels.json load error (%s); labels will be IDs only", e)
        return None


# ---------- Simple container for a detection frame ----------
class DetectionFrame:
    __slots__ = ("frame_id", "width", "height", "boxes_xyxy", "scores", "classes")

    def __init__(
        self,
        frame_id: int,
        width: int,
        height: int,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
    ):
        self.frame_id = frame_id
        self.width = width
        self.height = height
        self.boxes_xyxy = boxes_xyxy  # (N, 4) in model space (e.g., 640x640)
        self.scores = scores          # (N,)
        self.classes = classes        # (N,)


# ---------- Core Pipeline Controller ----------
class _PipelineController:
    """
    Builds a camera pipeline with a tee:
      A) Display branch: ... -> BGRA -> cairooverlay(draw) -> sink
      B) Detection branch: ... -> RGB 640x640 -> appsink -> ONNX worker

    Exposes start/stop preview and toggle detection.
    """

    def __init__(self, model_path: str, model_input: int = 640, overlay_enabled: bool = True):
        self.model_path = model_path
        self.model_input = int(os.getenv("MODEL_INPUT_SIZE", model_input))
        self.overlay_enabled = overlay_enabled

        # Load labels once
        labels_path = os.getenv("LABELS_PATH", "/app/labels.json")
        self.labels = load_labels(labels_path)

        # State
        self.preview_on = False
        self.detect_on = False
        self._frame_counter = 0

        # Latest disp size (from overlay caps)
        self._disp_w = 640
        self._disp_h = 480

        # Latest detections for the overlay to draw
        self._latest_det: Optional[DetectionFrame] = None
        self._latest_lock = threading.Lock()

        # Inference
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._detector: Optional[OnnxYoloV8] = None

        # GStreamer
        Gst.init(None)
        self._build_pipeline()

    # ---------- Pipeline build ----------
    def _build_pipeline(self):
        self.pipeline = Gst.Pipeline.new("cam-pipeline")

        # Source
        self.src = Gst.ElementFactory.make("v4l2src", "src")
        if not self.src:
            raise RuntimeError("Failed to create v4l2src")
        # Converters / scalers
        self.conv0 = Gst.ElementFactory.make("videoconvert", "conv0")
        self.scale0 = Gst.ElementFactory.make("videoscale", "scale0")

        # Common caps (camera → 640x480 @ ~30fps) for display branch
        self.caps_disp = Gst.ElementFactory.make("capsfilter", "caps_disp")
        self.caps_disp.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw,width=640,height=480,framerate=30/1"),
        )

        # Tee
        self.tee = Gst.ElementFactory.make("tee", "t")
        self.q_display = Gst.ElementFactory.make("queue", "q_display")
        self.q_detect = Gst.ElementFactory.make("queue", "q_detect")

        # --- Display branch A ---
        self.convA0 = Gst.ElementFactory.make("videoconvert", "convA0")
        self.capsA_bgra = Gst.ElementFactory.make("capsfilter", "capsA_bgra")
        self.capsA_bgra.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRA"))

        self.overlay = Gst.ElementFactory.make("cairooverlay", "overlay")
        # signals:
        self.overlay.connect("draw", self._on_overlay_draw)
        self.overlay.connect("caps-changed", self._on_overlay_caps_changed)

        self.convA1 = Gst.ElementFactory.make("videoconvert", "convA1")

        sink_name = os.getenv("GST_SINK", "").strip()
        if sink_name:
            self.sink = Gst.ElementFactory.make(sink_name, "sink")
            if not self.sink:
                log.warning("[sinks] WARNING: forced sink '%s' not available; falling back", sink_name)
        if not hasattr(self, "sink") or self.sink is None:
            # Default to GPU capable sink if present; else autovideosink
            self.sink = (Gst.ElementFactory.make("glimagesink", "sink")
                         or Gst.ElementFactory.make("autovideosink", "sink"))
            chosen = self.sink.get_name() if self.sink else "unknown"
            log.info("[sinks] Using sink: %s", chosen)

        # --- Detection branch B ---
        # Convert to RGB, scale to model_input x model_input, send to appsink
        self.convB0 = Gst.ElementFactory.make("videoconvert", "convB0")
        self.capsB_rgb = Gst.ElementFactory.make("capsfilter", "capsB_rgb")
        self.capsB_rgb.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))

        self.scaleB = Gst.ElementFactory.make("videoscale", "scaleB")
        self.capsB_size = Gst.ElementFactory.make("capsfilter", "capsB_size")
        self.capsB_size.set_property(
            "caps",
            Gst.Caps.from_string(f"video/x-raw,width={self.model_input},height={self.model_input}")
        )

        self.appsink = Gst.ElementFactory.make("appsink", "det_sink")
        self.appsink.set_property("emit-signals", False)
        self.appsink.set_property("sync", False)
        # we will pull samples in the worker via try_pull_sample
        self.appsink.set_property("max-buffers", 4)
        self.appsink.set_property("drop", True)

        # Add elements
        for elem in [
            self.src, self.conv0, self.scale0, self.caps_disp, self.tee,
            self.q_display, self.convA0, self.capsA_bgra, self.overlay, self.convA1, self.sink,
            self.q_detect, self.convB0, self.capsB_rgb, self.scaleB, self.capsB_size, self.appsink,
        ]:
            self.pipeline.add(elem)

        # Link common path to tee
        if not Gst.Element.link_many(self.src, self.conv0, self.scale0, self.caps_disp, self.tee):
            raise RuntimeError("Failed to link source -> conv0 -> scale0 -> caps_disp -> tee")

        # Link display branch
        okA = Gst.Element.link_many(
            self.q_display, self.convA0, self.capsA_bgra, self.overlay, self.convA1, self.sink
        )
        # Tee pad request + link
        pad_t_srcA = self.tee.get_request_pad("src_%u")
        pad_qA_sink = self.q_display.get_static_pad("sink")
        if pad_t_srcA is None or pad_qA_sink is None or pad_t_srcA.link(pad_qA_sink) != Gst.PadLinkReturn.OK or not okA:
            log.error("Linking display branch failed")
            raise RuntimeError("Linking display branch failed")

        # Link detection branch
        okB = Gst.Element.link_many(
            self.q_detect, self.convB0, self.capsB_rgb, self.scaleB, self.capsB_size, self.appsink
        )
        pad_t_srcB = self.tee.get_request_pad("src_%u")
        pad_qB_sink = self.q_detect.get_static_pad("sink")
        if pad_t_srcB is None or pad_qB_sink is None or pad_t_srcB.link(pad_qB_sink) != Gst.PadLinkReturn.OK or not okB:
            raise RuntimeError("Linking detection branch failed")

        log.info("Display chain A: conv->BGRA->cairooverlay->conv->sink")
        log.info("Pipeline built (overlay_enabled=%s)", self.overlay_enabled)

    # ---------- Public controls ----------
    def start_preview(self) -> bool:
        if self.preview_on:
            return True
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            log.error("Failed to set pipeline to PLAYING")
            self.pipeline.set_state(Gst.State.NULL)
            return False
        self.preview_on = True
        log.info("Camera preview started")
        return True

    def stop_preview(self):
        if not self.preview_on:
            return
        # Ensure detection stopped first
        if self.detect_on:
            self._stop_detection()
        self.pipeline.set_state(Gst.State.NULL)
        self.preview_on = False
        log.info("Camera preview stopped")

    def toggle_detection(self):
        if not self.preview_on:
            log.info("Detection requested while preview is off; ignoring")
            return
        if self.detect_on:
            self._stop_detection()
        else:
            self._start_detection()

    # ---------- Detection worker ----------
    def _start_detection(self):
        if self.detect_on:
            return
        # Lazy create detector here so app can start preview w/o model present
        if self._detector is None:
            self._detector = OnnxYoloV8(self.model_path, input_size=self.model_input)
            log.info("ONNX model loaded: %s (input=%d)", self.model_path, self.model_input)

        self._stop_evt.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="onnx-worker", daemon=True)
        self._worker.start()
        self.detect_on = True
        log.info("Object detection ENABLED")

    def _stop_detection(self):
        self._stop_evt.set()
        if self._worker:
            self._worker.join(timeout=1.5)
            self._worker = None
        self.detect_on = False
        log.info("Object detection DISABLED")

    def _worker_loop(self):
        frame_with_dets = 0
        pulled = 0
        while not self._stop_evt.is_set():
            # Pull a sample with a small timeout to stay responsive
            sample = self.appsink.try_pull_sample(0.2)  # seconds
            if sample is None:
                continue
            pulled += 1

            buf = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            w = s.get_value("width")
            h = s.get_value("height")

            # Map buffer -> numpy (RGB)
            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                continue
            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                arr = arr.reshape((h, w, 3))
                self._frame_counter += 1
                fid = self._frame_counter
                log.info("[worker] F%d: pulled frame %dx%d RGB", fid, w, h)

                # Inference
                boxes, scores, classes = self._detector.infer(arr)
                n = 0 if boxes is None else int(len(boxes))
                if n > 0:
                    frame_with_dets += 1
                # Store latest for overlay to draw
                df = DetectionFrame(
                    frame_id=fid,
                    width=w, height=h,
                    boxes_xyxy=np.array(boxes, dtype=np.float32) if n else np.zeros((0, 4), np.float32),
                    scores=np.array(scores, dtype=np.float32) if n else np.zeros((0,), np.float32),
                    classes=np.array(classes, dtype=np.int32) if n else np.zeros((0,), np.int32),
                )
                with self._latest_lock:
                    self._latest_det = df

                # Periodic progress log
                info_every = int(os.getenv("OD_INFO_EVERY", "30"))
                if info_every > 0 and (pulled % info_every) == 0:
                    pct = 100.0 * (frame_with_dets / max(1, pulled))
                    log.info("[worker] pulled=%d, frames_with_dets=%d (%.1f%%)", pulled, frame_with_dets, pct)
            finally:
                buf.unmap(mapinfo)

    # ---------- Overlay callbacks ----------
    def _on_overlay_caps_changed(self, overlay, caps):
        info = GstVideo.VideoInfo.from_caps(caps)
        self._disp_w = info.width
        self._disp_h = info.height
        log.info("Overlay caps-changed: %dx%d", self._disp_w, self._disp_h)

    def _draw_text(self, cr: cairo.Context, text: str, x: float, y: float, rgb: Tuple[float, float, float], size: float):
        cr.save()
        cr.select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        cr.set_font_size(size)  # smaller, as requested
        cr.set_source_rgb(*rgb)
        cr.move_to(x, y)
        cr.show_text(text)
        cr.restore()

    def _draw_box_with_label(self, cr: cairo.Context, x1: float, y1: float, x2: float, y2: float,
                             label: str, score: float):
        cr.save()
        cr.set_line_width(1.5)  # slightly slimmer
        # box color (greenish)
        cr.set_source_rgba(0.0, 1.0, 0.0, 0.9)
        cr.rectangle(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))
        cr.stroke()

        # label background
        text = f"{label} {score:.2f}"
        cr.select_font_face("Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD)
        cr.set_font_size(14.0)
        xb, yb, tw, th, xa, ya = cr.text_extents(text)
        pad = 3
        bx = x1
        by = max(0.0, y1 - th - 2 * pad)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.6)
        cr.rectangle(bx, by, tw + 2 * pad, th + 2 * pad)
        cr.fill()

        # label text
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.move_to(bx + pad, by + th + pad - 2)
        cr.show_text(text)

        cr.restore()

    def _on_overlay_draw(self, overlay, cr: cairo.Context, timestamp, duration):
        """
        Cairo draw callback. 'cr' is a cairo.Context (via gi.repository.cairo),
        so set_source_rgb/line_width/etc. are available.
        """
        # 1) Always show "Preview ON" (yellow, small) when preview is running
        if self.preview_on:
            self._draw_text(cr, "Preview ON", 10, 20, (1.0, 1.0, 0.0), 14.0)

        # 2) If detection is running, show "DETECT ON" (red, small)
        if self.detect_on:
            self._draw_text(cr, "DETECT ON", self._disp_w - 120, 20, (1.0, 0.0, 0.0), 14.0)

        # 3) Draw detections (if any)
        det: Optional[DetectionFrame] = None
        with self._latest_lock:
            det = self._latest_det

        if det is None or det.boxes_xyxy.size == 0:
            return

        # Map from model space (model_input x model_input) to display space (_disp_w x _disp_h)
        sx = self._disp_w / float(self.model_input)
        sy = self._disp_h / float(self.model_input)

        n = det.boxes_xyxy.shape[0]
        log_detections = True  # since you wanted max visibility
        if log_detections:
            # show first few mapped boxes for diagnostics
            log.info("[draw] have %d dets; disp=%dx%d model=%d", n, self._disp_w, self._disp_h, self.model_input)

        # If labels are present, build a quick mapping
        for i in range(n):
            x1, y1, x2, y2 = det.boxes_xyxy[i].tolist()
            dx1, dy1, dx2, dy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            cls_id = int(det.classes[i])
            score = float(det.scores[i])
            # label lookup (fallback to id)
            label = str(cls_id)
            if self.labels and 0 <= cls_id < len(self.labels):
                label = self.labels[cls_id]

            if log_detections and i < 3:  # cap logs a bit to avoid spam each frame
                log.info("[draw] box%d: model=[%.1f, %.1f, %.1f, %.1f] -> disp=[%.1f, %.1f, %.1f, %.1f] score=%.2f cls=%s",
                         i, x1, y1, x2, y2, dx1, dy1, dx2, dy2, score, label)

            self._draw_box_with_label(cr, dx1, dy1, dx2, dy2, label, score)


# ---------- Backwards-compatible public class ----------
class CameraPipeline(_PipelineController):
    """
    Alias class so main.py can continue importing CameraPipeline
    even if internal name changes.
    """
    pass


__all__ = ["CameraPipeline", "DetectionFrame"]
