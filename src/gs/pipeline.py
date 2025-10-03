# src/gs/pipeline.py
import os
import json
import threading
import queue
import time
import logging
from typing import List, Tuple, Optional

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("Gtk", "3.0")
from gi.repository import Gst, GstVideo, GObject

# --- logging ---
log = logging.getLogger("gs.pipeline")

# ------- labels loader (project-local) -------
def load_labels_from_project() -> Optional[dict]:
    """
    Loads /app/labels.json (default WORKDIR) unless LABELS_PATH overrides it.
    Expects the schema the user posted (classes is a dict of str(id) -> name).
    """
    path = os.environ.get("LABELS_PATH", "/app/labels.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = data.get("classes", {})
        if isinstance(classes, dict) and classes:
            # normalize to int->str map
            out = {}
            for k, v in classes.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            log.info("Loaded labels: %d entries from %s", len(out), path)
            return out
        log.warning("labels.json loaded but no 'classes' found: %s", path)
    except FileNotFoundError:
        log.warning("labels.json not found at %s (labels will be IDs only)", path)
    except Exception as e:
        log.exception("Failed to load labels.json at %s: %s", path, e)
    return None


LABELS = load_labels_from_project()

# ------- env knobs -------
DRAW_PREVIEW_BANNER = bool(int(os.environ.get("DRAW_PROBE_BOX", "1")))  # repurpose old flag
PREVIEW_BANNER_TEXT = os.environ.get("PREVIEW_BANNER_TEXT", "Preview ON")
DETECT_BANNER_TEXT  = os.environ.get("DETECT_BANNER_TEXT",  "DETECT ON")

# make both banners a bit smaller
BANNER_FONT_SIZE_PREVIEW = float(os.environ.get("BANNER_FONT_SIZE_PREVIEW", "16"))
BANNER_FONT_SIZE_DETECT  = float(os.environ.get("BANNER_FONT_SIZE_DETECT",  "16"))

# box/label drawing sizes
BOX_LINE_WIDTH = float(os.environ.get("BOX_LINE_WIDTH", "2.0"))
LABEL_FONT_SIZE = float(os.environ.get("LABEL_FONT_SIZE", "14"))

# how often to log overlay events (already very chatty, keep at 1)
OD_LOG_EVERY = max(1, int(os.environ.get("OD_LOG_EVERY", "30")))

# ------- detection queues / state containers -------
class DetectionResult:
    __slots__ = ("timestamp", "dets")  # dets: List[Tuple[x1,y1,x2,y2,score,cls]]
    def __init__(self, dets: List[Tuple[float, float, float, float, float, int]]):
        self.timestamp = time.time()
        self.dets = dets


class OverlayState:
    """Holds latest detection results (model-space), caps & flags."""
    def __init__(self):
        self.model_input = 640
        self.display_w = 640
        self.display_h = 480
        self.have_dets = False
        self.latest: Optional[DetectionResult] = None
        self.detection_enabled = False
        self.draw_counter = 0


OVERLAY = OverlayState()

# ------- GStreamer init -------
Gst.init(None)

def _scale_model_to_disp(box, disp_w, disp_h, model_size):
    # model coords assumed in [0, model_size] with origin at top-left
    x1, y1, x2, y2 = box
    scale_y = disp_h / float(model_size)
    scale_x = disp_w / float(model_size)
    return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


def _draw_banner(cr, text, x, y, font_size, rgba):
    # cr is a cairo.Context
    try:
        import cairo  # noqa
        cr.save()
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(font_size)
        cr.set_source_rgba(*rgba)
        cr.move_to(x, y)
        cr.show_text(text)
        cr.restore()
    except Exception as e:
        log.info("[overlay] banner draw error: %s", e)


def _draw_box_and_label(cr, x1, y1, x2, y2, score, cls_id):
    # rectangle
    try:
        import cairo  # noqa
        cr.save()
        cr.set_source_rgba(0.0, 1.0, 0.0, 0.9)  # green
        cr.set_line_width(BOX_LINE_WIDTH)
        cr.rectangle(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
        cr.stroke()
        # label (name + score) above top-left of box
        name = LABELS.get(cls_id, str(cls_id)) if LABELS else str(cls_id)
        label = f"{name} {score:.2f}"
        cr.select_font_face("Sans", 0, 0)
        cr.set_font_size(LABEL_FONT_SIZE)
        # simple text shadow for readability
        tx, ty = x1 + 4.0, max(12.0, y1 - 6.0)
        cr.set_source_rgba(0, 0, 0, 0.6)
        cr.move_to(tx + 1, ty + 1)
        cr.show_text(label)
        cr.set_source_rgba(1, 1, 0, 0.95)  # yellow text
        cr.move_to(tx, ty)
        cr.show_text(label)
        cr.restore()
    except Exception as e:
        log.info("[overlay] draw error: %s", e)


# --------- cairooverlay callbacks ----------
def on_draw(overlay, cr, timestamp, duration):
    """
    Cairo draw callback. 'cr' is a cairo.Context.
    """
    OVERLAY.draw_counter += 1
    n = OVERLAY.draw_counter

    disp_w, disp_h = OVERLAY.display_w, OVERLAY.display_h
    # 1) Preview banner (replaces old yellow rectangle)
    if DRAW_PREVIEW_BANNER:
        _draw_banner(
            cr,
            PREVIEW_BANNER_TEXT,
            10, 20,  # top-left, small offset
            BANNER_FONT_SIZE_PREVIEW,
            (1.0, 1.0, 0.0, 0.9),  # yellow
        )

    # 2) Detect banner when detection on
    if OVERLAY.detection_enabled:
        _draw_banner(
            cr,
            DETECT_BANNER_TEXT,
            disp_w - 120, 20,  # top-right-ish
            BANNER_FONT_SIZE_DETECT,
            (1.0, 0.1, 0.1, 0.95),  # red
        )

    # 3) Boxes + labels if we have detections
    dets = OVERLAY.latest.dets if (OVERLAY.have_dets and OVERLAY.latest) else []
    if dets:
        if n % OD_LOG_EVERY == 0:
            log.info("[draw] have %d dets; disp=%dx%d model=%d", len(dets), disp_w, disp_h, OVERLAY.model_input)
        for i, (mx1, my1, mx2, my2, score, cls_id) in enumerate(dets[:50]):  # cap for sanity
            x1, y1, x2, y2 = _scale_model_to_disp((mx1, my1, mx2, my2), disp_w, disp_h, OVERLAY.model_input)
            if n % OD_LOG_EVERY == 0 and i < 3:
                log.info("[draw] box%d: model=[%.1f, %.1f, %.1f, %.1f] -> disp=[%.1f, %.1f, %.1f, %.1f] score=%.2f cls=%s",
                         i, mx1, my1, mx2, my2, x1, y1, x2, y2, score, LABELS.get(cls_id, cls_id) if LABELS else cls_id)
            _draw_box_and_label(cr, x1, y1, x2, y2, score, cls_id)


def on_caps_changed(overlay, caps):
    s = caps.get_structure(0)
    w = s.get_value("width")
    h = s.get_value("height")
    OVERLAY.display_w = int(w)
    OVERLAY.display_h = int(h)
    log.info("Overlay caps-changed: %dx%d", w, h)


# ------- pipeline builder / controller (unchanged guts, summarized) -------
class PipelineController:
    def __init__(self):
        self.pipeline = None
        self.bus = None
        self._queue = queue.Queue(maxsize=8)
        self._worker_thr: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def build(self, with_overlay=True):
        # build v4l2src → convert → scale/model → tee → display branch
        # (left intact from your working version, only overlay wiring below)
        #
        # ... your camera / convert / scale / appsink setup ...
        #
        # display branch with cairooverlay
        overlay = Gst.ElementFactory.make("cairooverlay", "overlay")
        overlay.connect("draw", on_draw)
        overlay.connect("caps-changed", on_caps_changed)
        # add overlay and rest of display chain; ensure BGRA into overlay
        #
        # IMPORTANT: leave your earlier, working element creation & linking here.
        #
        log.info("Display chain A: conv->BGRA->cairooverlay->conv->sink")
        self.overlay = overlay
        log.info("Pipeline built (overlay_enabled=True)")

    def set_detection_enabled(self, enabled: bool):
        OVERLAY.detection_enabled = bool(enabled)

    def push_detections(self, dets: List[Tuple[float, float, float, float, float, int]], model_input: int):
        OVERLAY.model_input = model_input
        OVERLAY.latest = DetectionResult(dets)
        OVERLAY.have_dets = True

    # start/stop preview/detection methods remain as in your working version,
    # calling set_detection_enabled(True/False) appropriately and running the
    # ONNX worker that feeds push_detections(...).
