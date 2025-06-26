from ultralytics import YOLO
from boxmot import ByteTrack
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.tasks.pose import HumanPose
from cute_vision.visualization.viewer import Viewer
from cute_vision.visualization.draw.boxes import draw_box, draw_boxes
from cute_vision.tasks.track import result_to_boxmot_dets
from cute_vision.visualization.draw.colors import BGR_RED, BGR_BLUE
from cute_vision.utils import first, safe_first


model = YOLO('yolov8n-pose.pt')
tracker = ByteTrack(
    track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30
)
selected_track_id = None

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
            lambda poses: [index for index, pose in enumerate(poses) if pose is not None and pose.is_hand_up],
            safe_first
        ),
        Pipe(
            lambda result: result.orig_img
        ),
        Pipe(
            lambda result: (result_to_boxmot_dets(result), result.orig_img),
            lambda data: tracker.update(data[0], data[1])
        )
    ]
)


def sink(boxes, hand_up_index, frame, _):

    global selected_track_id
    global tracker

    frame = draw_boxes(frame, boxes, BGR_BLUE)

    if hand_up_index is not None:
        track = safe_first([t for t in tracker.active_tracks if t.det_ind == hand_up_index])
        if track is not None:
            selected_track_id = track.id

    if selected_track_id is not None:
        selected_track = safe_first([t for t in tracker.active_tracks if t.id == selected_track_id])
        if selected_track is not None:
            frame = draw_box(frame, selected_track.xyxy, BGR_RED)

    return frame


flow = Flow(
    source=DeviceCamera(device_index=0, width=1280, height=720).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer('Hand Up Example')
viewer.play(flow())
