import numpy as np


class PursuitTransformer:

    def __init__(self, origin_point: np.ndarray, frame_width: int, frame_height: int):

        if len(origin_point.shape) != 1:
            raise ValueError('origin_point must be a 1D array')

        if origin_point.shape[0] != 2:
            raise ValueError('origin_point must be a point (x, y)')

        self.origin_point = origin_point
        self.frame_width = frame_width
        self.frame_height = frame_height

    def __call__(self, target_point: np.ndarray | None) -> np.ndarray | None:

        if target_point is None:
            return None

        if len(target_point.shape) != 1:
            return np.array([0, 0])

        if target_point.shape[0] != 2:
            return np.array([0, 0])

        discrepancy_pixels = target_point - self.origin_point
        discrepancy_ratio = discrepancy_pixels / np.array([self.frame_width / 2, self.frame_height / 2])

        return discrepancy_ratio
