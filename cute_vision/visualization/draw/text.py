import cv2
import numpy as np


def draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_size: float = 1.0,
    font_thickness: int = 2
) -> np.ndarray:
    image = image.copy()
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)
    return image
