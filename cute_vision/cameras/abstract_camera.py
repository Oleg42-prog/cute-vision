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
