import os
from dataclasses import dataclass, field
import cv2
import numpy as np
from cute_vision.visualization.draw.colors import BGR_RED, BGR_GREEN, BGR_BLUE, BGR_YELLOW, BGR_CYAN
from cute_vision.visualization.draw.text import draw_text


COLORS = [
    BGR_RED,
    BGR_GREEN,
    BGR_BLUE,
    BGR_YELLOW,
    BGR_CYAN,
]


@dataclass
class ConvexHullProcess:

    name: str
    path: list[tuple[int, int]] = field(default_factory=list)

    def add_point(self, x: int, y: int):
        self.path.append((x, y))

    def clear(self):
        self.path = []

    def draw(self, image: np.ndarray, color: tuple[int, int, int]):

        if not self.path:
            return image

        img = image.copy()
        hull = cv2.convexHull(np.array(self.path))

        for x, y in self.path:
            cv2.circle(img, (x, y), 3, color, -1)

        for i in hull:
            x, y = i[0]
            cv2.circle(img, (x, y), 3, color, -1)

        cv2.drawContours(img, [hull], 0, color, 2)

        return img


class ConvexHullCreator:

    def __init__(self, image_path: str, names: list[str]):
        self.image_path = image_path
        self.image = self._load_image(image_path)
        self.image = self.image.copy()
        self.convex_hulls = [ConvexHullProcess(name) for name in names]
        self.current_index = 0

    def _load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'File {image_path} not found')
        return cv2.imread(image_path)

    def save(self):
        named_convex_hulls = {convex_hull.name: convex_hull.path for convex_hull in self.convex_hulls}
        np.save(self.image_path, named_convex_hulls)

    def draw(self):
        img = self.image.copy()
        for convex_hull, color in zip(self.convex_hulls, COLORS):
            img = convex_hull.draw(img, color)
        return img

    def next(self):
        self.current_index += 1
        if self.current_index >= len(self.convex_hulls):
            self.current_index = 0
        return self.convex_hulls[self.current_index]

    @property
    def current_convex_hull(self):
        return self.convex_hulls[self.current_index]

    @property
    def image_name(self):
        return os.path.basename(self.image_path).split('.')[0]


def handle_mouse_event(event, x, y, flags, convex_hull_creator: ConvexHullCreator):
    if event == cv2.EVENT_LBUTTONDOWN:
        convex_hull_creator.current_convex_hull.add_point(x, y)


input_path = input('Input image path: ')
names = input('Input names: ').split(' ')
convex_hull_creator = ConvexHullCreator(input_path, names)

cv2.namedWindow('image')
cv2.setMouseCallback('image', handle_mouse_event, convex_hull_creator)

while True:

    image = convex_hull_creator.draw()
    image = draw_text(image, convex_hull_creator.current_convex_hull.name, (10, 30), BGR_RED)
    cv2.imshow('image', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        convex_hull_creator.current_convex_hull.clear()

    if cv2.waitKey(1) & 0xFF == ord('n'):
        convex_hull_creator.next()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        convex_hull_creator.save()


cv2.destroyAllWindows()
convex_hull_creator.save()
