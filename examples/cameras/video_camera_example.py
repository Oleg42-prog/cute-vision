from cute_vision.cameras import VideoCamera
from cute_vision.view import Viewer

VIDEO_PATH = 'your_video_path.mp4'

viewer = Viewer('Video Camera Example')
viewer.play(VideoCamera(path=VIDEO_PATH).frames())
