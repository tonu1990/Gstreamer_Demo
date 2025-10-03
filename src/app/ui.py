# src/app/ui.py
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

import logging
log = logging.getLogger("app.ui")

class MainWindow(Gtk.Window):
    def __init__(self, pipeline_ctrl):
        super().__init__(title="GStreamer Demo")
        self.set_default_size(900, 600)
        self.pipeline = pipeline_ctrl

        # --- buttons row ---
        btn_preview = Gtk.Button(label="Start Preview")
        btn_detect  = Gtk.Button(label="Start Object Detection")
        btn_stop    = Gtk.Button(label="Stop Preview")
        btn_detect.set_sensitive(False)

        btn_preview.connect("clicked", self.on_start_preview, btn_detect)
        btn_stop.connect("clicked", self.on_stop_preview, btn_detect)
        btn_detect.connect("clicked", self.on_toggle_detect)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_top(8); header.set_margin_bottom(8)
        header.set_margin_start(8); header.set_margin_end(8)
        header.pack_start(btn_preview, False, False, 0)
        header.pack_start(btn_detect,  False, False, 0)
        header.pack_start(btn_stop,    False, False, 0)

        # --- video area inside an Overlay ---
        # Assume your sink is already set to draw into 'self.video_area' widget window;
        # many examples use a Gtk.DrawingArea for that.
        self.video_area = Gtk.DrawingArea()
        self.video_area.set_size_request(800, 480)

        self.placeholder = Gtk.Label(label="Object Detection Application POC- Tonu James")
        self.placeholder.get_style_context().add_class("large-placeholder")
        self.placeholder.set_halign(Gtk.Align.CENTER)
        self.placeholder.set_valign(Gtk.Align.CENTER)

        self.overlay = Gtk.Overlay()
        self.overlay.add(self.video_area)
        self.overlay.add_overlay(self.placeholder)
        self.placeholder.show()

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        root.pack_start(header, False, False, 0)
        root.pack_start(self.overlay, True, True, 0)
        self.add(root)

        # basic CSS for larger placeholder text (optional)
        css = b"""
        .large-placeholder {
            font: 18px Sans;
            color: #dddddd;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.show_all()

    def on_start_preview(self, btn, btn_detect):
        log.info("UI: Start Preview clicked")
        self.placeholder.hide()
        started = self.pipeline.start_preview()  # your existing method
        if started:
            btn_detect.set_sensitive(True)

    def on_stop_preview(self, btn, btn_detect):
        log.info("UI: Stop Preview clicked")
        self.pipeline.stop_preview()             # your existing method
        btn_detect.set_sensitive(False)
        # show placeholder again after a tiny delay to ensure sink stops
        GLib.timeout_add(150, lambda: (self.placeholder.show(), False))

    def on_toggle_detect(self, btn):
        # your existing toggle logic; label update omitted for brevity
        self.pipeline.toggle_detection()
