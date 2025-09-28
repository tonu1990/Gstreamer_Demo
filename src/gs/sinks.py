import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

PREFERRED_SINKS = ("glimagesink", "ximagesink", "xvimagesink")

def choose_sink():
    for name in PREFERRED_SINKS:
        sink = Gst.ElementFactory.make(name, "sink")
        if sink:
            # best-effort: some sinks have force-aspect-ratio
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            return sink
    return None
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

PREFERRED_SINKS = ("glimagesink", "ximagesink", "xvimagesink")

def choose_sink():
    for name in PREFERRED_SINKS:
        sink = Gst.ElementFactory.make(name, "sink")
        if sink:
            # best-effort: some sinks have force-aspect-ratio
            try:
                sink.set_property("force-aspect-ratio", False)
            except Exception:
                pass
            return sink
    return None
