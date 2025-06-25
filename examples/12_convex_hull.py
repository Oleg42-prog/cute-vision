import numpy as np
from ultralytics import YOLO
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras import DeviceCamera
from cute_vision.geometry.convex_hull import ConvexHull
from cute_vision.visualization.viewer import Viewer
from cute_vision.visualization.draw.points import draw_np_points
from cute_vision.visualization.draw.colors import BGR_RED, BGR_BLUE
from cute_vision.tasks.pose import HumanPose
from cute_vision.utils import first, safe_first


model = YOLO('yolov8n-pose.pt')

CONVEX_HULL_PATH = 'area.npy'
convex_hull = ConvexHull(CONVEX_HULL_PATH, 'danger_area')

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: result.plot()
        ),
        Pipe(
            lambda result: result.keypoints.cpu().numpy().xy,
            HumanPose.from_keypoints_list,
            safe_first,
            lambda pose: [] if not pose else [pose.middle_point],
            np.array
        ),
        Pipe(
            lambda result: result.keypoints.cpu().numpy().xy,
            HumanPose.from_keypoints_list,
            safe_first,
            lambda pose: [] if not pose else [pose.middle_point],
            convex_hull.is_any_point_in_hull
        ),

    ]
)


def sink(frame, points, is_any_in_hull):

    frame = draw_np_points(frame, points, BGR_RED)

    hull_color = BGR_BLUE if is_any_in_hull else BGR_RED
    frame = convex_hull.draw(frame, hull_color)

    return frame


flow = Flow(
    source=DeviceCamera(device_index=0, width=1280, height=720).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer()
viewer.play(flow())
