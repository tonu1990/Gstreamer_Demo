# src/app/main.py
import os
import sys
import logging

from app.ui import run_app            # however you bootstrap the GTK loop
from gs.pipeline import CameraPipeline

log = logging.getLogger(__name__)

def _as_bool(v: str | None, default=False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def main():
    # --- read config from env ---
    model_path = os.getenv("MODEL_PATH")
    model_input = os.getenv("MODEL_INPUT_SIZE", "640")
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    draw_probe_box = _as_bool(os.getenv("DRAW_PROBE_BOX"), default=False)
    od_log_every  = int(os.getenv("OD_LOG_EVERY", "10"))
    od_info_every = int(os.getenv("OD_INFO_EVERY", "30"))
    labels_path   = os.getenv("LABELS_PATH", "/app/labels.json")  # your repo copy

    # --- logging ---
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("=== Application logging started ===")

    # --- validate inputs early with helpful messages ---
    if not model_path:
        log.error("MODEL_PATH env is not set.")
        sys.exit(2)
    if not os.path.exists(model_path):
        log.error("MODEL_PATH not found inside container: %s", model_path)
        sys.exit(2)

    try:
        model_input = int(model_input)
    except ValueError:
        log.error("MODEL_INPUT_SIZE must be an integer, got %r", model_input)
        sys.exit(2)

    if not os.path.exists(labels_path):
        log.warning("labels.json not found at %s (labels will be IDs only)", labels_path)
        labels_path = None

    # --- create pipeline with explicit args ---
    pipe = CameraPipeline(
        model_path=model_path,
        model_input_size=model_input,
        labels_path=labels_path,          # can be None
        draw_probe_box=draw_probe_box,
        od_log_every=od_log_every,
        od_info_every=od_info_every,
    )

    # Hand pipe to your GTK UI bootstrap
    run_app(pipe)

if __name__ == "__main__":
    main()
