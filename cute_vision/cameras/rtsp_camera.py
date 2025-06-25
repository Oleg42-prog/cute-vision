import cv2
import numpy as np
from cute_vision.cameras.abstract_camera import AbstractCamera


class RTSPCamera(AbstractCamera):

    def __init__(self, url: str, width: int = 640, height: int = 480):
        super().__init__()
        self.url = url
        self.width = width
        self.height = height
        self.cap = None

    def connect(self) -> bool:
        width, height = self.width, self.height
        gst_pipeline = (
            f'rtspsrc location={self.url} latency=0 buffer-mode=auto '
            f'drop-on-latency=true timeout=5000000 ! '
            'rtph264depay ! h264parse ! avdec_h264 ! '
            f'videorate ! video/x-raw,framerate=30/1 ! '
            f'videoconvert ! videoscale ! '
            f'video/x-raw,width={width},height={height} ! '
            'appsink drop=true max-buffers=1 sync=false'
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        return self.cap.isOpened()

    def fetch_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def disconnect(self):
        self.cap.release()

    @property
    def frame_width(self):
        return self.width

    @property
    def frame_height(self):
        return self.height
