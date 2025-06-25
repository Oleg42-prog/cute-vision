import cv2
import numpy as np


class ConvexHull:

    def __init__(self, path: str, name: str | None = None):
        data = np.load(path, allow_pickle=True)

        if data.ndim == 0:
            data_dict = data.item()
        else:
            data_dict = data

        self._name = name
        if name:
            array = np.array(data_dict[name], dtype=np.int32)
        else:
            array = np.array(data_dict, dtype=np.int32)

        self._hull = cv2.convexHull(array)

    @property
    def name(self):
        return self._name

    def is_point_in_hull(self, point: tuple[int, int] | None) -> bool:

        if not point:
            return False

        hull = self._hull.copy()
        list_hull = hull.reshape(-1, 2).tolist()
        list_hull.append(point)

        new_hull = np.array(list_hull).astype(np.int32).astype(np.float32)
        ch = cv2.convexHull(new_hull)

        if ch.shape == hull.shape:
            if np.abs(ch - hull).sum() < 0.01:
                return True

        return False

    def is_any_point_in_hull(self, points: list[tuple[int, int]]) -> bool:
        for point in points:
            if self.is_point_in_hull(point):
                return True
        return False

    def how_many_points_in_hull(self, points: list[tuple[int, int]]) -> int:
        return sum(1 for point in points if self.is_point_in_hull(point))

    @property
    def hull(self):
        return self._hull

    def draw(self, image: np.ndarray, color: tuple[int, int, int]):
        image = image.copy()
        return cv2.drawContours(image, [self._hull], 0, color, 2)
