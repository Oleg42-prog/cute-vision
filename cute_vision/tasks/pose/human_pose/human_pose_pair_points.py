from cute_vision.tasks.pose.human_pose.human_pose_point import HumanPosePoint


class HumanPosePairPoints:

    def __init__(self, left_point: HumanPosePoint, right_point: HumanPosePoint):
        self.left_point = left_point
        self.right_point = right_point

    @property
    def middle_point(self) -> tuple[int, int] | None:

        if self.left_point.is_valid and self.right_point.is_valid:
            return (self.left_point.x + self.right_point.x) / 2, (self.left_point.y + self.right_point.y) / 2

        if self.left_point.is_valid:
            return self.left_point.x, self.left_point.y

        if self.right_point.is_valid:
            return self.right_point.x, self.right_point.y

        return None
