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
