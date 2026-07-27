import cv2
import numpy as np
from ultralytics import YOLO
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras import DeviceCamera
from cute_vision.geometry.convex_hull import ConvexHull
from cute_vision.visualization.viewer import Viewer
from cute_vision.geometry.xyxy_to_points import reduce_xyxy_to_points
from cute_vision.visualization.draw.points import draw_np_points
from cute_vision.visualization.draw.colors import BGR_RED, BGR_BLUE
from cute_vision.tasks.pose import HumanPose
from cute_vision.utils import first, safe_first

def oprint(x):
    print(x)
    return x

model = YOLO('yolov8s.pt')
CAMERA_NAME = '1'
DEVICE_INDEX = r'C:\Users\odudnik\Desktop\stream\stream2.mp4'
CONVEX_HULL_PATH = f'convex_hulls/A0{CAMERA_NAME}.npy'
convex_hull = ConvexHull(CONVEX_HULL_PATH)

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: cv2.resize(x, (int(x.shape[1] / 1.5), int(x.shape[0] / 1.5))),
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: result.plot()
        ),
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
            lambda xyxy: reduce_xyxy_to_points(xyxy, 0.5, 0.9)
        ),
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
            lambda xyxy: reduce_xyxy_to_points(xyxy, 0.5, 0.9).tolist(),
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
    source=DeviceCamera(device_index=DEVICE_INDEX, width=1280, height=720).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer()
viewer.play(flow())
