"""Windows-специфичные модули для работы с окнами и захватом изображений."""

from cute_vision.platform.windows.window_handle import WindowHandle
from cute_vision.platform.windows.bitmap_capture import BitmapCapture

__all__ = [
    'WindowHandle',
    'BitmapCapture'
]
