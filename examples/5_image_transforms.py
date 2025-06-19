import cv2
from pipes_and_filters import Pipeline, Pipe
from cute_vision.cameras.device_camera import DeviceCamera
from cute_vision.filters import bgr_to_grayscale
from cute_vision.filters import flip_horizontal


pipeline = Pipeline(
    source=DeviceCamera(device_index=1).frames(),
    pipe=Pipe(
        flip_horizontal,
        bgr_to_grayscale
    )
)

for frame in pipeline():
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
