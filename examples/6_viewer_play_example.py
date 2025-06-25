from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

viewer = Viewer('Viewer Play Example')
viewer.play(DeviceCamera(device_index=0).frames())
