from cute_vision.cameras import RTSPCamera
from cute_vision.visualization import Viewer

RTSP_URL = 'rtsp://admin:rumicon_rumicon@172.16.70.110:554/ISAPI/Streaming/Channels/101'
WIDTH = 1280
HEIGHT = 720

viewer = Viewer('RTSP Camera Example')
viewer.play(RTSPCamera(url=RTSP_URL, width=WIDTH, height=HEIGHT).frames())
