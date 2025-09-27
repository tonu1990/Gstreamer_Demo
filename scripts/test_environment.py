#!/usr/bin/env python3
import sys
import gi

try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(sys.argv)
    print("✓ GStreamer imported successfully")
    
    # Test element creation
    pipeline = Gst.Pipeline.new("test-pipeline")
    source = Gst.ElementFactory.make("videotestsrc", "source")
    sink = Gst.ElementFactory.make("autovideosink", "sink")
    
    if pipeline and source and sink:
        print("✓ GStreamer elements created successfully")
    else:
        print("✗ GStreamer element creation failed")
        
except Exception as e:
    print(f"✗ GStreamer test failed: {e}")

try:
    from PySide6.QtWidgets import QApplication
    print("✓ PySide6 imported successfully")
except ImportError as e:
    print(f"✗ PySide6 import failed: {e}")

print("Environment test completed.")