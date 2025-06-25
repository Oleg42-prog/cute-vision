from typing import Iterable
import cv2
import numpy as np


class Viewer:

    def __init__(self, title: str = 'cute-window', wait_time: int = 1):
        self.title = title
        self.wait_time = wait_time

    def __call__(self, image: np.ndarray):
        cv2.imshow(self.title, image)
        key = cv2.waitKey(self.wait_time) & 0xFF
        if key == ord('q') or key == 27:
            return False
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()

    def play(self, iterable: Iterable[np.ndarray]):
        with self as viewer:
            for image in iterable:
                if image is None:
                    continue
                if not viewer(image):
                    break
