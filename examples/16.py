import cv2
import numpy as np
from pydispatch import dispatcher
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
from cute_vision.events import EventEmitter
from cute_vision.geometry.pursuit_transformer import PursuitTransformer
from cute_vision.visualization.draw.points import draw_point
from cute_vision.visualization.draw.text import draw_text


def draw_border_lines(frame, borders):

    np_borders = np.array(borders)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    pixel_borders = np_borders * frame_width // 2 + frame_width // 2
    pixel_borders = pixel_borders.astype(int)

    frame = cv2.line(frame, (pixel_borders[0], 0), (pixel_borders[0], frame_height), BGR_BLUE, 1)
    frame = cv2.line(frame, (pixel_borders[1], 0), (pixel_borders[1], frame_height), BGR_BLUE, 1)

    return frame


def pursuit_event_handler(sender, **kwargs):
    print(f'Event received: {kwargs}')


dispatcher.connect(pursuit_event_handler, signal='pursuit-event')
event_emitter = EventEmitter('pursuit-event')

pursuit_transformer = PursuitTransformer(
    origin_point=np.array([640, 360]),
    frame_width=1280,
    frame_height=720
)


model = YOLO('yolov8l-pose.pt')
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
            lambda result: result.keypoints.cpu().numpy().xy,
            HumanPose.from_keypoints_list,
            lambda poses: [index for index, pose in enumerate(poses) if pose is not None and pose.is_hand_up],
            safe_first
        ),
        Pipe(
            lambda result: result.keypoints.cpu().numpy().xy,
            HumanPose.from_keypoints_list
        ),
        Pipe(
            lambda result: result.orig_img
        ),
        Pipe(
            lambda result: (result_to_boxmot_dets(result), result.orig_img),
            lambda data: tracker.update(data[0], data[1])
        ),
        Pipe(
            lambda result: draw_boxes(result.orig_img, result.boxes.xyxy.cpu().numpy(), BGR_BLUE)
        ),
    ]
)

pursuit_pipe = Pipe(
    pursuit_transformer,
    safe_first,
    lambda x: None if -0.3 < x < 0.3 else x,
    lambda discrepancy_ratio_x: event_emitter(
        emit=discrepancy_ratio_x is not None,
        passthrough=discrepancy_ratio_x,
        event_kwargs={'discrepancy_ratio': discrepancy_ratio_x}
    )
)


def sink(hand_up_index, human_poses, frame, *args):

    global selected_track_id
    global tracker

    frame = draw_text(frame, f'Selected track id: {selected_track_id}', (10, 60), BGR_RED)
    frame = draw_border_lines(frame, [-0.3, 0.3])

    if hand_up_index is not None:
        track = safe_first([t for t in tracker.active_tracks if t.det_ind == hand_up_index])
        if track is not None:
            selected_track_id = track.id

    if selected_track_id is None:
        selected_track = safe_first(tracker.active_tracks)
        if selected_track is not None:
            selected_track_id = selected_track.id

    if selected_track_id is None:
        pursuit_pipe(pursuit_transformer.origin_point)
        return frame

    selected_track = safe_first([t for t in tracker.active_tracks if t.id == selected_track_id])
    if selected_track is None:
        selected_track_id = None
        pursuit_pipe(pursuit_transformer.origin_point)
        return frame

    frame = draw_box(frame, selected_track.xyxy, BGR_RED)
    human_pose = human_poses[int(selected_track.det_ind)]
    if human_pose.middle_point is not None:
        middle_point = np.array(human_pose.middle_point)
        discrepancy_ratio_x = pursuit_pipe(middle_point)
        frame = draw_point(frame, middle_point.astype(int), BGR_RED, 10)
        if discrepancy_ratio_x is not None:
            frame = draw_text(frame, f'Discrepancy Ratio: {discrepancy_ratio_x:.2f}', (10, 30), BGR_RED)

    return frame


flow = Flow(
    source=DeviceCamera(device_index=0, width=1280, height=720).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer('Hand Up Example')
viewer.play(flow())
