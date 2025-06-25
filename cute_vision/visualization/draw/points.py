import cv2
import numpy as np


def draw_np_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], radius: int = 5) -> np.ndarray:
    points = points.astype(int).tolist()
    return draw_points(image, points, color, radius)


def draw_np_point(image: np.ndarray, point: np.ndarray, color: tuple[int, int, int], radius: int = 5) -> np.ndarray:
    point = point.astype(int).tolist()
    return draw_point(image, point, color, radius)


def draw_point(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int], radius: int = 5) -> np.ndarray:
    image = image.copy()
    cv2.circle(image, point, radius, color, -1)
    return image


def draw_points(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], radius: int = 5) -> np.ndarray:
    image = image.copy()
    for point in points:
        image = draw_point(image, point, color, radius)
    return image
