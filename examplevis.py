import cv2
import numpy as np

class ExampleCircle:
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
