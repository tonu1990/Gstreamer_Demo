import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# You can override with: GST_SINK=ximagesink or GST_SINK=glimagesink
FORCED_SINK = os.getenv("GST_SINK", "").strip()

# Default order: prefer ximagesink first to avoid GL issues in Docker
PREFERRED_SINKS = ("ximagesink", "xvimagesink", "glimagesink")

def choose_sink():
    # If user explicitly requested a sink
    if FORCED_SINK:
        sink = Gst.ElementFactory.make(FORCED_SINK, "sink")
        if sink:
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            print(f"[sinks] Using forced sink: {FORCED_SINK}")
            return sink
        else:
            print(f"[sinks] WARNING: forced sink '{FORCED_SINK}' not available; falling back")

    for name in PREFERRED_SINKS:
        sink = Gst.ElementFactory.make(name, "sink")
        if sink:
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            print(f"[sinks] Using sink: {name}")
            return sink
    return None
