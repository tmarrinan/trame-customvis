from trame.app import get_server
from trame.widgets import vuetify, rca
from trame.ui.vuetify import SinglePageLayout
from examplevis import ExampleCircle
import time

def main():
    # Create image with circle
    init_w = 800
    init_h = 600
    vis = ExampleCircle(init_w, init_h, (128, 15, 196))
    
    # Create Trame server
    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller

    @ctrl.add("on_server_ready")
    def init_rca(**kwargs):
        view_handler = ViewAdapter(vis, "view")
        ctrl.rc_area_register(view_handler)
    
    # Define webpage layout
    state.active_display_mode = "raw-image" # RGB: "raw-image", JPEG: "image", MP4: "media-source"
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Custom-Vis")
        with layout.content:
            with vuetify.VContainer(fluid=True, classes="pa-0 fill-height",):
                view = rca.RemoteControlledArea(name="view", display=("active_display_mode", "image"))
    
    # Start server
    server.start()

# ViewAdapter Class for Controller RCA
class ViewAdapter:
    def __init__(self, circle_vis, name):
        self._view = circle_vis
        self._streamer = None
        self._last_meta = None
        self.area_name = name
        
    def _get_metadata(self):
        # Dictionary:
        #  - type:  mime-type (RGB: "image/rgb24", JPEG: "image/jpeg", MP4: "video/mp4")
        #  - codec: video codec (H264: avc1.<profile><constraints><level>  -->  e.g. "avc1.64003e")
        #                        H265: hvc1.<profile>.<compatibility>.L<level>.<constraints>  -->  e.g. "hvc1.2.4.L153.B0")
        #  - w:     width
        #  - h:     height
        #  - st:    time (milliseconds as integer)
        #  - key:   keyframe or deltaframe ("key" or "delta")
    
        height, width = self._view.getSize()
        return dict(
            type="image/rgb24",
            codec="",
            w=width,
            h=height,
            st=time.time_ns() // 1000000,
            key="key"
        )

    def set_streamer(self, stream_manager):
        self.streamer = stream_manager
        
    def update_size(self, origin, size):
        width = int(size.get("w", 400))
        height = int(size.get("h", 300))
        print(f"new size: {width}x{height}")
        
    def on_interaction(self, origin, event):
        event_type = event["type"]
        print(f"Event: {event_type}")
        if event_type == "LeftButtonPress":
            # Raw RGB
            pixels = self._view.getRawImage()
            self.streamer.push_content(self.area_name, self._get_metadata(), pixels.data)
            # JPEG
            #jpeg_image = BytesIO()
            #self._view.save(jpeg_image, "jpeg")
            #self.streamer.push_content(self.area_name, self._get_metadata(), memoryview(jpeg_image.getvalue()))


if __name__ == "__main__":
    main()
