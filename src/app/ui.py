# src/app/ui.py
import logging
log = logging.getLogger("app.ui")

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gtk, Gdk, GLib

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
        for w in (btn_preview, btn_detect, btn_stop):
            header.pack_start(w, False, False, 0)
        header.set_margin_top(8); header.set_margin_bottom(8)
        header.set_margin_start(8); header.set_margin_end(8)

        # --- video area with placeholder overlay ---
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

        # CSS for placeholder (guarded if no screen)
        css = b"""
        .large-placeholder {
            font: 18px Sans;
            color: #dddddd;
        }
        """
        try:
            provider = Gtk.CssProvider()
            provider.load_from_data(css)
            screen = Gdk.Screen.get_default()
            if screen is not None:
                Gtk.StyleContext.add_provider_for_screen(
                    screen, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
                )
        except Exception as e:
            log.warning("UI CSS load skipped: %s", e)

        self.show_all()

    def on_start_preview(self, _btn, btn_detect):
        log.info("UI: Start Preview clicked")
        self.placeholder.hide()
        if self.pipeline.start_preview():
            btn_detect.set_sensitive(True)

    def on_stop_preview(self, _btn, btn_detect):
        log.info("UI: Stop Preview clicked")
        self.pipeline.stop_preview()
        btn_detect.set_sensitive(False)
        GLib.timeout_add(150, lambda: (self.placeholder.show(), False))

    def on_toggle_detect(self, _btn):
        self.pipeline.toggle_detection()


def run_app(pipeline_ctrl):
    win = MainWindow(pipeline_ctrl)
    win.connect("destroy", Gtk.main_quit)
    Gtk.main()
