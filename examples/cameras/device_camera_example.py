from cute_vision.cameras import DeviceCamera
from cute_vision.visualization import Viewer

DEVICE_INDEX = 'rtsp://admin:rumicon_rumicon@172.16.70.110:554/ISAPI/Streaming/Channels/101'
WIDTH = 1280
HEIGHT = 720

viewer = Viewer('Device Camera Example')
viewer.play(DeviceCamera(device_index=DEVICE_INDEX, width=WIDTH, height=HEIGHT).frames())
