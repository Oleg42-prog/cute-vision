from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

DEVICE_INDEX = 0
WIDTH = 1280
HEIGHT = 720

viewer = Viewer('Device Camera Example')
viewer.play(DeviceCamera(device_index=DEVICE_INDEX, width=WIDTH, height=HEIGHT).frames())
