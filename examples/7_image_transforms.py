from pipes_and_filters import Pipeline, Pipe
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.filters import bgr_to_grayscale
from cute_vision.filters import flip_horizontal
from cute_vision.view import Viewer


pipeline = Pipeline(
    source=DeviceCamera(device_index=0).frames(),
    pipe=Pipe(
        flip_horizontal,
        bgr_to_grayscale
    )
)

viewer = Viewer('main-window')
viewer.play(pipeline())
