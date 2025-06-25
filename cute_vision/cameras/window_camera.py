import numpy as np
from typing import Optional
from cute_vision.cameras.abstract_camera import AbstractCamera
from cute_vision.platform.windows import WindowHandle, BitmapCapture


class WindowCamera(AbstractCamera):
    """
    Камера для захвата кадров из окна Windows.

    Упрощённая версия с разделением ответственности между компонентами.
    """

    def __init__(self, window_title: str):
        super().__init__()
        self.window = WindowHandle(window_title)
        self.capture = BitmapCapture(self.window)
        self.connected = False

    def connect(self) -> bool:
        """Подключается к окну."""
        if self.window.find_and_connect():
            self.connected = True
            return True
        else:
            return False

    def disconnect(self) -> None:
        """Отключается от окна."""
        self.connected = False
        self.window.hwnd = None

    def fetch_frame(self) -> Optional[np.ndarray]:
        """Получает кадр из окна."""
        if not self.connected:
            return None

        if not self.window.is_valid():
            self.connected = False
            return None

        return self.capture.capture()

    @property
    def frame_width(self) -> int:
        """Ширина кадра."""
        if self.connected and self.window.is_valid():
            self.window._update_size()
            return self.window.width
        return 0

    @property
    def frame_height(self) -> int:
        """Высота кадра."""
        if self.connected and self.window.is_valid():
            self.window._update_size()
            return self.window.height
        return 0

    def __repr__(self) -> str:
        status = "подключено" if self.connected else "отключено"
        return f"WindowCamera('{self.window.window_title}', {self.window.width}x{self.window.height}, {status})"
