import os
import logging
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

log = logging.getLogger("sinks")

PREFERRED_SINKS = ("glimagesink", "ximagesink", "xvimagesink")

def choose_sink():
    forced = os.getenv("GST_SINK", "").strip()
    if forced:
        sink = Gst.ElementFactory.make(forced, "sink")
        if sink:
            log.info(f"Using sink: {forced}")
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            return sink
        else:
            log.warning(f"forced sink '{forced}' not available; falling back")

    for name in PREFERRED_SINKS:
        sink = Gst.ElementFactory.make(name, "sink")
        if sink:
            log.info(f"Using sink: {name}")
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            return sink
    return None
