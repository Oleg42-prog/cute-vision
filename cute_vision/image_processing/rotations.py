import cv2
import numpy as np


def rotate_90_degrees_clockwise(image: np.ndarray) -> np.ndarray:
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
