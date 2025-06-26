from ultralytics import YOLO
from boxmot import ByteTrack
from cute_vision.tasks.track import result_to_boxmot_dets
from pipes_and_filters import Flow, Pipe, Splitter
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.visualization.viewer import Viewer
from cute_vision.utils import first


model = YOLO('yolov8n.pt')
tracker = ByteTrack(
    track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30
)

splitter = Splitter(
    input_pipe=Pipe(
        lambda x: model(x, classes=[0], conf=0.6, verbose=False),
        first
    ),
    outputs_pipes=[
        Pipe(
            lambda result: (result_to_boxmot_dets(result), result.orig_img),
            lambda data: tracker.update(data[0], data[1])
        ),
        Pipe(
            lambda result: result.orig_img
        )
    ]
)


def sink(tracker_results, frame):
    print(tracker_results)
    return tracker.plot_results(frame, show_trajectories=True)


flow = Flow(
    source=DeviceCamera(device_index=0).frames(),
    splitter=splitter,
    sink=sink
)

viewer = Viewer('Tracking Example')
viewer.play(flow())
