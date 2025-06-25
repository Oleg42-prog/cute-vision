import mss
import numpy as np
from cute_vision.cameras.abstract_camera import AbstractCamera


class ScreenCamera(AbstractCamera):

    def __init__(self, monitor: int = 1, region: tuple[int, int, int, int] | None = None):
        """
        Initialize ScreenCamera.

        Args:
            monitor (int): Monitor number to capture (1-based, default: 1 for primary monitor)
            region (tuple[int, int, int, int] | None): Region to capture as (left, top, width, height).
            If region is None, captures entire monitor.
        """
        super().__init__()
        self.monitor = monitor
        self.region = region
        self.sct = None
        self._monitor_info = None
        self._width = None
        self._height = None
        self._capture_area = None

    def connect(self) -> bool:
        try:
            self.sct = mss.mss()

            if self.monitor <= len(self.sct.monitors) - 1:
                self._monitor_info = self.sct.monitors[self.monitor]
            else:
                self._monitor_info = self.sct.monitors[1]

            if self.region is not None:
                left, top, width, height = self.region
                self._capture_area = {
                    "left": self._monitor_info["left"] + left,
                    "top": self._monitor_info["top"] + top,
                    "width": width,
                    "height": height
                }
                self._width = width
                self._height = height
            else:
                self._capture_area = self._monitor_info
                self._width = self._monitor_info["width"]
                self._height = self._monitor_info["height"]

            return True
        except Exception as e:
            print(f"Failed to connect to screen capture: {e}")
            return False

    def disconnect(self):
        if self.sct is not None:
            self.sct.close()
            self.sct = None

    def fetch_frame(self) -> np.ndarray | None:
        if self.sct is None:
            return None

        try:
            screenshot = self.sct.grab(self._capture_area)
            frame_bgra = np.array(screenshot)
            frame_bgr = frame_bgra[:, :, :3]
            return frame_bgr
        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None

    @property
    def frame_width(self) -> int:
        return self._width if self._width is not None else 0

    @property
    def frame_height(self) -> int:
        return self._height if self._height is not None else 0
