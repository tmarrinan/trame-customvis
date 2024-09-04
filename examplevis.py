import time
import asyncio
import cv2
import numpy as np

# Custom visualization class for drawing a circle
class ExVisCircle:
    def __init__(self, width, height, color):
        self._clear_canvas = np.zeros((height, width, 3), np.uint8)
        self._image = None
        self._center = [width // 2, height // 2]
        self._color = color
        self._radius = 100
        self._thickness = 4
        self._velocity_x = 100
        self._velocity_y = 60
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}
        self._start_time = round(time.time_ns() / 1000000)
        self._prev_time = self._start_time

        self.renderFrame()

    def getSize(self):
        height, width, channels = self._clear_canvas.shape
        return (width, height)
    
    def setSize(self, width, height):
        self._clear_canvas = np.zeros((height, width, 3), np.uint8)

    def getImageType(self):
        return self._image_type

    def setImageType(self, itype, options={}):
        self._image_type = itype
        if self._image_type == "rgb":
            pass # do nothing
        elif self._image_type == "jpeg":
            self._jpeg_quality = options.get("quality", 92)
        elif self._image_type == "h264":
            self._video_options = options

    def getFrame(self):
        if self._image_type == "rgb":
            return self._getRawImage()
        elif self._image_type == "jpeg":
            return self._getJpegImage()
        elif self._image_type == "h264":
            return self._getH264VideoFrame()
        else:
            return None

    def getRenderTime(self):
        return self._prev_time

    def renderFrame(self):
        # Animate
        now = round(time.time_ns() / 1000000)
        dt = (now - self._prev_time) / 1000
        
        dx = round(self._velocity_x * dt)
        dy = round(self._velocity_y * dt)
        height, width, channels = self._clear_canvas.shape
        if self._center[0] + dx < 0:
            self._center[0] = 0
            self._velocity_x *= -1
        elif self._center[0] + dx > width:
            self._center[0] = width
            self._velocity_x *= -1
        else:
            self._center[0] += dx
        if self._center[1] + dy < 0:
            self._center[1] = 0
            self._velocity_y *= -1
        elif self._center[1] + dy > height:
            self._center[1] = height
            self._velocity_y *= -1
        else:
            self._center[1] += dy
        
        # Render
        self._image = self._clear_canvas.copy()
        cv2.circle(self._image, self._center, self._radius, self._color, self._thickness)
        
        # Update render time
        self._prev_time = now

    def _getRawImage(self):
        start_ms = round(time.time_ns() / 1000000)
    
        rgb_img = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        rgb_img = rgb_img.flatten()
        
        end_ms = round(time.time_ns() / 1000000)
        print(f"Copy RGB framebuffer: {end_ms - start_ms} ms")
        
        return rgb_img
        
    def _getJpegImage(self):
        result, encoded_img = cv2.imencode('.jpg', self._image, (cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality))
        if result:
            return encoded_img
        return None
        
    def _getH264VideoFrame(self):
        return None


# ViewAdapter class for Controller RCA
class ExVisViewAdapter:
    def __init__(self, vis, name):
        self._view = vis
        self._streamer = None
        self._last_meta = None
        self.area_name = name
        self._target_fps = 30
        
        #asynchronous.create_task(self._animate())
        asyncio.create_task(self._animate())
        
    def _get_metadata(self):
        # Dictionary:
        #  - type:  mime-type (RGB: "image/rgb24", JPEG: "image/jpeg", MP4: "video/mp4")
        #  - codec: video codec (H264: avc1.<profile><constraints><level>  -->  e.g. "avc1.64003e")
        #                        H265: hvc1.<profile>.<compatibility>.L<level>.<constraints>  -->  e.g. "hvc1.2.4.L153.B0")
        #  - w:     width
        #  - h:     height
        #  - st:    time (milliseconds as integer)
        #  - key:   keyframe or deltaframe ("key" or "delta")
    
        mime_types = {"rgb": "image/rgb24", "jpeg": "image/jpeg", "h264": "video/mp4"}
        width, height = self._view.getSize()
        return dict(
            type=mime_types[self._view.getImageType()],
            codec="",
            w=width,
            h=height,
            st=self._view.getRenderTime(),
            key="key"
        )

    async def _animate(self):
        while True:
            if self._streamer != None:
                self._view.renderFrame()
                frame_data = self._view.getFrame()
                self._streamer.push_content(self.area_name, self._get_metadata(), frame_data.data)
                await asyncio.sleep(1.0 / self._target_fps)
            await asyncio.sleep(0)

    def set_streamer(self, stream_manager):
        self._streamer = stream_manager
        
    def update_size(self, origin, size):
        width = int(size.get("w", 400))
        height = int(size.get("h", 300))
        print(f"new size: {width}x{height}")
        self._view.setSize(width, height)
        
    def on_interaction(self, origin, event):
        event_type = event["type"]
        print(f"Event: {event_type}")
        if event_type == "LeftButtonPress":
            pass
            #frame_data = self._view.getFrame()
            #self._streamer.push_content(self.area_name, self._get_metadata(), frame_data.data)
