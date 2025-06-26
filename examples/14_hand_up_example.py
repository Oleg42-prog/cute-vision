from ultralytics import YOLO
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.tasks.pose import HumanPose
from cute_vision.visualization.viewer import Viewer
from cute_vision.visualization.draw.text import draw_text
from cute_vision.visualization.draw.colors import BGR_RED, BGR_GREEN
from cute_vision.visualization.draw.boxes import draw_boxes
from cute_vision.utils import first


model = YOLO('yolov8n-pose.pt')

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: result.boxes.xyxy.cpu().numpy(),
        ),
        Pipe(
            lambda result: result.keypoints.cpu().numpy().xy,
            HumanPose.from_keypoints_list,
            lambda poses: [pose.is_hand_up for pose in poses]
        ),
        Pipe(
            lambda result: result.plot()
        )
    ]
)


def sink(boxes, hand_ups, frame):
    colors = [BGR_RED if hand_up else BGR_GREEN for hand_up in hand_ups]
    frame = draw_boxes(frame, boxes, colors)
    frame = draw_text(frame, 'Hand Up' if any(hand_ups) else 'Hand Down', (30, 30), BGR_RED)
    return frame


flow = Flow(
    source=DeviceCamera(device_index=0, width=1280, height=720).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer('Hand Up Example')
viewer.play(flow())
