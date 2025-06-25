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

        return cls(
            nose=nose,
            eyes=eyes,
            shoulders=shoulders,
            hips=hips
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
