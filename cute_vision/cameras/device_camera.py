import cv2
import numpy as np
from cute_vision.cameras.abstract_camera import AbstractCamera


class DeviceCamera(AbstractCamera):

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self.cap.isOpened()

    def disconnect(self):
        self.cap.release()

    def fetch_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    @property
    def frame_width(self):
        return self.width

    @property
    def frame_height(self):
        return self.height
