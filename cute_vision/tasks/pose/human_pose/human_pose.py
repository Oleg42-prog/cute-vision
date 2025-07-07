from typing import Self
from dataclasses import dataclass
import numpy as np
from cute_vision.tasks.pose.human_pose.human_pose_point import HumanPosePoint
from cute_vision.tasks.pose.human_pose.human_pose_pair_points import HumanPosePairPoints


@dataclass
class HumanPose:

    nose: HumanPosePoint
    eyes: HumanPosePairPoints
    shoulders: HumanPosePairPoints
    hips: HumanPosePairPoints
    wrists: HumanPosePairPoints

    @classmethod
    def from_keypoints_list(cls, keypoints_list: list[np.ndarray]) -> list[Self]:
        if keypoints_list is None or len(keypoints_list) == 0:
            return []
        return [cls.from_keypoints(keypoints) for keypoints in keypoints_list]

    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray) -> Self | None:

        if keypoints.shape[0] == 0:
            return None

        nose = HumanPosePoint(keypoints[0])

        left_eye = HumanPosePoint(keypoints[1])
        right_eye = HumanPosePoint(keypoints[2])
        eyes = HumanPosePairPoints(left_eye, right_eye)

        left_shoulder = HumanPosePoint(keypoints[5])
        right_shoulder = HumanPosePoint(keypoints[6])
        shoulders = HumanPosePairPoints(left_shoulder, right_shoulder)

        left_hip = HumanPosePoint(keypoints[11])
        right_hip = HumanPosePoint(keypoints[12])
        hips = HumanPosePairPoints(left_hip, right_hip)

        left_wrist = HumanPosePoint(keypoints[9])
        right_wrist = HumanPosePoint(keypoints[10])
        wrists = HumanPosePairPoints(left_wrist, right_wrist)

        return cls(
            nose=nose,
            eyes=eyes,
            shoulders=shoulders,
            hips=hips,
            wrists=wrists
        )

    @property
    def middle_point(self) -> tuple[int, int] | None:

        if self.nose.is_valid:
            return self.nose.x, self.nose.y

        if self.eyes.middle_point:
            return self.eyes.middle_point

        if self.shoulders.middle_point:
            return self.shoulders.middle_point

        if self.hips.middle_point:
            return self.hips.middle_point

        return None

    @property
    def is_hand_up_old(self) -> bool:

        shoulders_y = [self.shoulders.left_point.y, self.shoulders.right_point.y]
        shoulders_y = list(filter(lambda y: y != 0, shoulders_y))
        if not shoulders_y:
            return False
        shoulders_line_y = min(shoulders_y)

        wrists_y = [self.wrists.left_point.y, self.wrists.right_point.y]
        wrists_y = list(filter(lambda y: y != 0, wrists_y))
        if not wrists_y:
            return False
        wrists_line_y = min(wrists_y)

        if wrists_line_y < shoulders_line_y:
            return True

        return False

    def is_hand_up(self, threshold: int = 100) -> bool:

        eyes_y = [self.eyes.left_point.y, self.eyes.right_point.y]
        eyes_y = list(filter(lambda y: y != 0, eyes_y))
        if not eyes_y:
            return False
        eyes_line_y = min(eyes_y)

        wrists_y = [self.wrists.left_point.y, self.wrists.right_point.y]
        wrists_y = list(filter(lambda y: y != 0, wrists_y))
        if not wrists_y:
            return False
        wrists_line_y = min(wrists_y)

        if wrists_line_y < eyes_line_y - threshold:
            return True

        return False
