import cv2
import numpy as np
from cute_vision.cameras.abstract_camera import AbstractCamera


class VideoCamera(AbstractCamera):

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.cap = None

    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
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
        raise NotImplementedError

    @property
    def frame_height(self):
        raise NotImplementedError
