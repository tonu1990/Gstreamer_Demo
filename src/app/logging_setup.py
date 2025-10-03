import logging
import os
from datetime import datetime
from pathlib import Path

def init_logging(app_name: str = "Gstreamer_Demo") -> str:
    """
    Initialize logging to a timestamped file on the user's Desktop.
    Returns the log file path.
    """
    # Resolve Desktop (fallback to home if missing)
    home = Path.home()
    desktop = home / "Desktop"
    if not desktop.exists():
        desktop = home

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = desktop / f"{app_name}_{stamp}.log"

    # Root logger config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler()  # keep console too
        ]
    )
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)  # quiet down ORT chatter
    logging.info("=== Application logging started ===")
    logging.info(f"Log file: {logfile}")
    return str(logfile)
