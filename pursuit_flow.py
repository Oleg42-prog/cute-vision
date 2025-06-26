import threading
import cv2
import numpy as np
from pipes_and_filters import Flow, Splitter, Pipe
from cute_vision.cameras import AbstractCamera
from cute_vision.tasks.pose.human_pose import HumanPose
from cute_vision.geometry.pursuit_transformer import PursuitTransformer
from cute_vision.events import EventEmitter
from cute_vision.visualization import Viewer
from cute_vision.visualization.draw import draw_np_points
from cute_vision.visualization.draw.text import draw_text
from cute_vision.visualization.draw.colors import BGR_RED
from cute_vision.utils import first


class PursuitAdjustmentEvent:
    pass


class PursuitFlow(threading.Thread):

    def __init__(self, camera: AbstractCamera, model, window_name: str, resize_to: tuple[int, int] = (640, 360)):
        super().__init__(daemon=True)

        self.camera = camera
        self.model = model
        self.event_emitter = EventEmitter(PursuitAdjustmentEvent)
        self.viewer = Viewer(window_name)
        self.pursuit_transformer = self.init_pursuit_transformer()

        self.splitter = self.init_splitter()
        self.flow = Flow(
            source=self.camera.frames(),
            splitter=self.splitter,
            sink=self.sink
        )

        self.resize_to = resize_to

    def init_pursuit_transformer(self):
        return PursuitTransformer(
            origin_point=np.array([self.camera.frame_width // 2, self.camera.frame_height // 2]),
            frame_width=self.camera.frame_width,
            frame_height=self.camera.frame_height
        )

    def init_splitter(self):
        return Splitter(
            input_pipe=Pipe(
                lambda x: self.model(x, classes=[0], conf=0.6, verbose=False),
                first
            ),
            outputs_pipes=[
                Pipe(
                    lambda result: result.keypoints.cpu().numpy().xy,
                    HumanPose.from_keypoints_list,
                    lambda poses: poses[0] if len(poses) > 0 else None,
                    lambda pose: [] if not pose or not pose.middle_point else [pose.middle_point],
                    np.array
                ),
                Pipe(
                    lambda result: result.plot()
                ),
                Pipe(
                    lambda result: result.keypoints.cpu().numpy().xy,
                    HumanPose.from_keypoints_list,
                    lambda poses: poses[0] if len(poses) > 0 else None,
                    lambda pose: [] if not pose else pose.middle_point,
                    np.array,
                    lambda point: self.pursuit_transformer(point) if point is not None else np.array([0, 0]),
                    lambda point: point[0].item(),
                    lambda discrepancy_ratio_x: self.event_emitter(
                        emit=True,
                        passthrough=discrepancy_ratio_x,
                        event_kwargs={'discrepancy_ratio_x': discrepancy_ratio_x}
                    )
                )
            ]
        )

    def sink(self, points: np.ndarray, frame: np.ndarray, discrepancy_ratio_x: float):
        if points is not None:
            frame = draw_np_points(frame, points, BGR_RED)
        if discrepancy_ratio_x is not None:
            frame = draw_text(frame, f'Discrepancy ratio: {discrepancy_ratio_x:.2f}', (10, 30), BGR_RED)
        frame = cv2.resize(frame, self.resize_to)
        return frame

    def run(self):
        self.viewer.play(self.flow())
