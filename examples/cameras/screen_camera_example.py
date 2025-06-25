from cute_vision.cameras import ScreenCamera
from cute_vision.visualization import Viewer

MONITOR = 1
LEFT = 0
TOP = 0
WIDTH = 800
HEIGHT = 600

viewer = Viewer('Screen Camera Example')
viewer.play(ScreenCamera(monitor=MONITOR, region=(LEFT, TOP, WIDTH, HEIGHT)).frames())
