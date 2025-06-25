from pipes_and_filters import Pipeline, Pipe
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.image_processing import bgr_to_grayscale
from cute_vision.image_processing import flip_horizontal
from cute_vision.visualization import Viewer


pipeline = Pipeline(
    source=DeviceCamera(device_index=0).frames(),
    pipe=Pipe(
        flip_horizontal,
        bgr_to_grayscale
    )
)

viewer = Viewer('Image Transforms Example')
viewer.play(pipeline())
