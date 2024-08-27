from trame.app import get_server
from trame.widgets import vuetify, rca
from trame.ui.vuetify import SinglePageLayout
from examplevis import ExVisCircle, ExVisViewAdapter
from examplevulkan import ExVkTriangle

def main():
    # Create image with circle
    init_w = 800
    init_h = 600
    vis = ExVisCircle(init_w, init_h, (196, 15, 128)) # color in BGR
    test = ExVkTriangle(init_w, init_h)
    test.renderFrame()
    x = test._getRawImage()
    print(x)
    
    # Create Trame server
    server = get_server(client_type="vue2")
    state = server.state
    ctrl = server.controller

    # Register RCA view with Trame controller
    @ctrl.add("on_server_ready")
    def initRca(**kwargs):
        view_handler = ExVisViewAdapter(vis, "view")
        ctrl.rc_area_register(view_handler)
    
    @ctrl.trigger("my_method")
    def on_method(*args, **kwargs):
        print(f"server::method {args} {kwargs}")
        
    # Callback for encoder type change
    def uiStateEncoderUpdate(stream_encoder, **kwargs):
        print(stream_encoder)
        if stream_encoder == "rgb":
            state.active_display_mode = "raw-image"
            vis.setImageType(stream_encoder)
        elif stream_encoder == "jpeg":
            state.active_display_mode = "image"
            vis.setImageType(stream_encoder, options={"quality": 90})
        elif stream_encoder == "h264":
            state.active_display_mode = "media-source"
            vis.setImageType(stream_encoder)
    
    # Register callback
    state.change("stream_encoder")(uiStateEncoderUpdate)
    
    # Define webpage layout
    state.active_display_mode = "raw-image" # RGB: "raw-image", JPEG: "image", MP4: "media-source"
    with SinglePageLayout(server) as layout:
        layout.title.set_text("Custom-Vis")
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Encoder",
                v_model=("stream_encoder", "rgb"),
                items=(
                    "['rgb', 'jpeg', 'h264']",
                ),
                hide_details=True,
                dense=True,
            )
        with layout.content:
            with vuetify.VContainer(v_mutate="trigger('my_method')", fluid=True, classes="pa-0 fill-height",):
                view = rca.RemoteControlledArea(name="view", display=("active_display_mode", "image"))
    
    # Start server
    server.start()


if __name__ == "__main__":
    main()
