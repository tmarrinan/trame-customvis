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

        cv2.circle(self._image, self._center, self._radius, self._color, self._thickness)

    def getSize(self):
        return self._image.shape[:2]

    def getRawImage(self):
        return self._image.flatten()
        
    def getJpegImage(self, quality=92):
        result, encoded_img = cv2.imencode('.jpg', self._image, (cv2.IMWRITE_JPEG_QUALITY, quality))
        if result:
            return encoded_img
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
    
        height, width = self._view.getSize()
        return dict(
            type="image/jpeg", #"image/rgb24",
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
            #pixels = self._view.getRawImage()
            #self.streamer.push_content(self.area_name, self._get_metadata(), pixels.data)
            # JPEG
            jpeg_image = self._view.getJpegImage(quality=92)
            self.streamer.push_content(self.area_name, self._get_metadata(), jpeg_image.data)
