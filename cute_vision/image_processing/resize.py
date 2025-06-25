import cv2
import numpy as np


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height))


def resize_scale(image: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)
