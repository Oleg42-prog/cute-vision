from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

viewer = Viewer('Viewer Example')

for frame in DeviceCamera(device_index=0).frames():
    if not viewer(frame):
        break
