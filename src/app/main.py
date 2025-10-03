# src/app/main.py
import os
import sys
import logging

log = logging.getLogger(__name__)

def _as_bool(v, default=False):
    if v is None: return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def run_headless(pipe):
    log.warning("GTK not available or HEADLESS=1: running headless.")
    auto_prev   = _as_bool(os.getenv("AUTOSTART_PREVIEW"), True)
    auto_detect = _as_bool(os.getenv("AUTOSTART_DETECT"), False)

    if auto_prev:
        pipe.start_preview()
    if auto_detect:
        pipe.toggle_detection()

    # Keep process alive until Ctrl+C
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop_preview()

def main():
    # logging setup
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("=== Application logging started ===")

    # pipeline args from env
    model_path = os.getenv("MODEL_PATH")
    model_input = int(os.getenv("MODEL_INPUT_SIZE", "640"))
    labels_path = os.getenv("LABELS_PATH", "/app/labels.json")
    draw_probe_box = _as_bool(os.getenv("DRAW_PROBE_BOX"), False)
    od_log_every  = int(os.getenv("OD_LOG_EVERY", "10"))
    od_info_every = int(os.getenv("OD_INFO_EVERY", "30"))

    if not model_path or not os.path.exists(model_path):
        log.error("MODEL_PATH not found inside container: %s", model_path)
        sys.exit(2)
    if not os.path.exists(labels_path):
        log.warning("labels.json not found at %s (labels will be IDs only)", labels_path)
        labels_path = None

    from gs.pipeline import CameraPipeline
    pipe = CameraPipeline(
        model_path=model_path,
        model_input_size=model_input,
        labels_path=labels_path,
        draw_probe_box=draw_probe_box,
        od_log_every=od_log_every,
        od_info_every=od_info_every,
    )

    # Try GTK UI unless HEADLESS=1
    if _as_bool(os.getenv("HEADLESS"), False):
        return run_headless(pipe)

    try:
        from app.ui import run_app
        return run_app(pipe)
    except Exception as e:
        log.exception("Failed to start GTK UI; falling back to headless: %s", e)
        return run_headless(pipe)

if __name__ == "__main__":
    main()
