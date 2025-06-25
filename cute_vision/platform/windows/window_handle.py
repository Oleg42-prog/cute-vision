"""Класс для управления дескриптором окна Windows."""

import win32gui
from typing import Optional


class WindowHandle:
    """Управляет дескриптором окна и его свойствами."""

    def __init__(self, window_title: str):
        if not window_title or not window_title.strip():
            raise ValueError("window_title не может быть пустым")

        self.window_title = window_title.strip()
        self.hwnd: Optional[int] = None
        self.width = 0
        self.height = 0

    def find_and_connect(self) -> bool:
        """Находит окно и получает его параметры."""
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        if not self.hwnd:
            return False

        self._update_size()
        return self.width > 0 and self.height > 0

    def is_valid(self) -> bool:
        """Проверяет, что окно существует."""
        return self.hwnd is not None and win32gui.IsWindow(self.hwnd)

    def _update_size(self):
        """Обновляет размеры окна."""
        if self.hwnd:
            left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
            self.width = right - left
            self.height = bot - top
