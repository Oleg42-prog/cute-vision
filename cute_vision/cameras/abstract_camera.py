from abc import ABC, abstractmethod
import numpy as np


class AbstractCamera(ABC):

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self):
        ...

    @abstractmethod
    def fetch_frame(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def frame_width(self):
        ...

    @property
    @abstractmethod
    def frame_height(self):
        ...

    @property
    def frame_width_center(self):
        return self.frame_width // 2

    @property
    def frame_height_center(self):
        return self.frame_height // 2

    @property
    def frame_center(self):
        return self.frame_width_center, self.frame_height_center

    def __enter__(self):
        if not self.connect():
            raise RuntimeError("Failed to connect to camera")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return None

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.fetch_frame()
        if frame is None:
            raise StopIteration("Failed to get frame from camera")
        return frame
