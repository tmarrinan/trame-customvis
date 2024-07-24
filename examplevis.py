import time
import cv2
import numpy as np

# Custom visualization class for drawing a circle
class ExVisCircle:
    def __init__(self, width, height, color):
        self._image = np.zeros((height, width, 3), np.uint8)
        self._center = (width // 2, height // 2)
        self._color = color
        self._radius = 100
        self._thickness = 4
        self._image_type = "rgb"
        self._jpeg_quality = 92
        self._video_options = {}

        cv2.circle(self._image, self._center, self._radius, self._color, self._thickness)

    def getSize(self):
        return self._image.shape[:2]
        
    def setImageType(self, itype, options={}):
        if self._image_type == itype:
            return
        
        self._image_type = itype
        if type == "rgb":
            pass # do nothing
        elif type == "jpeg":
            self._jpeg_quality = options.get("quality", 92)
        elif type == "h264":
            self._video_options = options

    def getImageType(self):
        return self._image_type

    def getFrame(self):
        if self._image_type == "rgb":
            return self._getRawImage()
        elif self._image_type == "jpeg":
            return self._getJpegImage()
        elif self._image_type == "h264":
            return self._getH264VideoFrame()
        else:
            return None

    def _getRawImage(self):
        rgb_img = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        return rgb_img.flatten()
        
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
        height, width = self._view.getSize()
        return dict(
            type=mime_types[self._view.getImageType()],
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
            frame_data = self._view.getFrame()
            self.streamer.push_content(self.area_name, self._get_metadata(), frame_data.data)
            # Raw RGB
            #pixels = self._view.getRawImage()
            #self.streamer.push_content(self.area_name, self._get_metadata(), pixels.data)
            # JPEG
            #jpeg_image = self._view.getJpegImage(quality=92)
            #self.streamer.push_content(self.area_name, self._get_metadata(), jpeg_image.data)
