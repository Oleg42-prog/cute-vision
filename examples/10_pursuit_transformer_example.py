import numpy as np
from pydispatch import dispatcher
from ultralytics import YOLO
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.events import EventEmitter
from cute_vision.visualization.viewer import Viewer
from cute_vision.geometry.xyxy_to_points import reduce_xyxy_to_points
from cute_vision.geometry.pursuit_transformer import PursuitTransformer
from cute_vision.visualization.draw.points import draw_np_points
from cute_vision.visualization.draw.colors import BGR_RED
from cute_vision.visualization.draw.text import draw_text
from cute_vision.utils import first, safe_first


def pursuit_event_handler(sender, **kwargs):
    print(f'Event received: {kwargs}')


dispatcher.connect(pursuit_event_handler, signal='pursuit-event')
event_emitter = EventEmitter('pursuit-event')

pursuit_transformer = PursuitTransformer(
    origin_point=np.array([320, 240]),
    frame_width=640,
    frame_height=480
)

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
        ),
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
            lambda xyxy: reduce_xyxy_to_points(xyxy, 0.5, 0.5),
            safe_first,
            pursuit_transformer,
            safe_first,
            lambda discrepancy_ratio_x: event_emitter(
                emit=discrepancy_ratio_x is not None,
                passthrough=discrepancy_ratio_x,
                event_kwargs={'discrepancy_ratio': discrepancy_ratio_x}
            )
        )
    ]
)


def sink(points: np.ndarray | None, frame: np.ndarray, discrepancy_ratio_x: float | None) -> np.ndarray:
    if points is not None:
        frame = draw_np_points(frame, points, BGR_RED)
    if discrepancy_ratio_x is not None:
        frame = draw_text(frame, f'Discrepancy Ratio: {discrepancy_ratio_x:.2f}', (10, 30), BGR_RED)
    return frame


flow = Flow(
    source=DeviceCamera(device_index=0).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer('Pursuit Transformer Example')
viewer.play(flow())
