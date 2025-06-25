class HumanPosePoint:

    x: int
    y: int

    def __init__(self, xy: tuple[int, int]):
        self.x = xy[0]
        self.y = xy[1]

    @property
    def is_valid(self) -> bool:
        return self.x > 0 and self.y > 0
