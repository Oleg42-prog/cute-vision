from cute_vision.cameras import RTSPCamera
from cute_vision.view import Viewer

RTSP_URL = ''
WIDTH = 1280
HEIGHT = 720

viewer = Viewer('RTSP Camera Example')
viewer.play(RTSPCamera(url=RTSP_URL, width=WIDTH, height=HEIGHT).frames())
