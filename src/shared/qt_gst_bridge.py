import gi
gi.require_version('GstVideo', '1.0')
from gi.repository import GstVideo

def set_overlay_handle(sink_or_src, handle: int):
    """Safely set the native window handle on a VideoOverlay implementer."""
    try:
        GstVideo.VideoOverlay.set_window_handle(sink_or_src, handle)
        return True
    except Exception:
        return False
