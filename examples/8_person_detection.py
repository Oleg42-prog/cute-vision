from ultralytics import YOLO
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.visualization.viewer import Viewer
from cute_vision.filters.reduces.xyxy_to_points import reduce_xyxy_to_points
from cute_vision.visualization.draw.points import draw_np_points
from cute_vision.visualization.draw.colors import BGR_RED
from cute_vision.utils import first


model = YOLO('yolov8n.pt')

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
            lambda xyxy: reduce_xyxy_to_points(xyxy, 0.5, 0.5)
        ),
        Pipe(
            lambda result: result.plot()
        )
    ]
)

flow = Flow(
    source=DeviceCamera(device_index=0).frames(),
    splitter=splitter,
    sink=lambda points, frame: draw_np_points(frame, points, BGR_RED)
)

viewer = Viewer('Person Detection Example')
viewer.play(flow())
