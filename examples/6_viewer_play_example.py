from cute_vision.cameras import DeviceCamera
from cute_vision.view import Viewer

viewer = Viewer('main-window')
viewer.play(DeviceCamera(device_index=0).frames())
