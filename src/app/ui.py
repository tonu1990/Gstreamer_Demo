# src/app/ui.py
import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

import logging
log = logging.getLogger("app.ui")

class MainWindow(Gtk.Window):
    def __init__(self, pipeline_ctrl):
        super().__init__(title="GStreamer Demo")
        self.set_default_size(900, 600)
        self.pipeline = pipeline_ctrl
        self._detect_on = False

        # --- buttons row ---
        self.btn_preview = Gtk.Button(label="Start Preview")
        self.btn_detect  = Gtk.Button(label="Start Object Detection")
        self.btn_stop    = Gtk.Button(label="Stop Preview")
        self.btn_detect.set_sensitive(False)

        self.btn_preview.connect("clicked", self.on_start_preview)
        self.btn_stop.connect("clicked", self.on_stop_preview)
        self.btn_detect.connect("clicked", self.on_toggle_detect)
        self.connect("destroy", self.on_destroy)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_top(8); header.set_margin_bottom(8)
        header.set_margin_start(8); header.set_margin_end(8)
        header.pack_start(self.btn_preview, False, False, 0)
        header.pack_start(self.btn_detect,  False, False, 0)
        header.pack_start(self.btn_stop,    False, False, 0)

        # --- video area inside an Overlay ---
        self.video_area = Gtk.DrawingArea()
        self.video_area.set_size_request(800, 480)

        # Placeholder text shown before preview starts
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

        # Basic CSS for larger placeholder text (keep modest size)
        css = b"""
        .large-placeholder {
            font: 16px Sans;
            color: #dddddd;
        }
        """
        provider = Gtk.CssProvider()
        provider.load_from_data(css)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        self.show_all()

    def on_start_preview(self, *_):
        log.info("UI: Start Preview clicked")
        self.placeholder.hide()
        started = self.pipeline.start_preview()
        if started:
            self.btn_detect.set_sensitive(True)

    def on_stop_preview(self, *_):
        log.info("UI: Stop Preview clicked")
        # If detection is on, turn it off first (keeps UI state tidy)
        if self._detect_on:
            self.pipeline.toggle_detection()
            self._detect_on = False
            self.btn_detect.set_label("Start Object Detection")

        self.pipeline.stop_preview()
        self.btn_detect.set_sensitive(False)

        # show placeholder again after a tiny delay to ensure sink stops
        GLib.timeout_add(150, lambda: (self.placeholder.show(), False))

    def on_toggle_detect(self, *_):
        log.info("UI: Toggle Object Detection clicked")
        self.pipeline.toggle_detection()
        self._detect_on = not self._detect_on
        self.btn_detect.set_label(
            "Stop Object Detection" if self._detect_on else "Start Object Detection"
        )

    def on_destroy(self, *_):
        # Make sure we stop everything on window close
        try:
            if self._detect_on:
                self.pipeline.toggle_detection()
            self.pipeline.stop_preview()
        except Exception as e:
            log.warning(f"UI: cleanup on destroy raised: {e}")
        Gtk.main_quit()
