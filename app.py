from trame.app import get_server
from trame.widgets import vuetify, rca
from trame.ui.vuetify import SinglePageLayout
from examplevis import ExVisCircle, ExVisViewAdapter

def main():
    # Create image with circle
    init_w = 800
    init_h = 600
    vis = ExVisCircle(init_w, init_h, (128, 15, 196))
    
    # Create Trame server
    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller

    # Register RCA view with Trame controller
    @ctrl.add("on_server_ready")
    def init_rca(**kwargs):
        view_handler = ExVisViewAdapter(vis, "view")
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


if __name__ == "__main__":
    main()
