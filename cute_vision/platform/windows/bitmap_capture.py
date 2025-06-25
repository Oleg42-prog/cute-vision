"""Класс для захвата содержимого окна через Windows API."""

import win32ui
import win32gui
from ctypes import windll
import numpy as np
from typing import Optional

from .window_handle import WindowHandle


class BitmapCapture:
    """Захватывает содержимое окна через Win32 API."""

    def __init__(self, window_handle: WindowHandle):
        self.window = window_handle

    def capture(self) -> Optional[np.ndarray]:
        """Захватывает кадр из окна."""
        if not self.window.is_valid():
            return None

        windll.user32.SetProcessDPIAware()
        hwndDC = win32gui.GetWindowDC(self.window.hwnd)

        if not hwndDC:
            return None

        result = self._do_capture(hwndDC)
        win32gui.ReleaseDC(self.window.hwnd, hwndDC)
        return result

    def _do_capture(self, hwndDC) -> Optional[np.ndarray]:
        """Выполняет захват через bitmap."""
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfcDC, self.window.width, self.window.height)

        saveDC.SelectObject(bitmap)

        # Захватываем окно
        result = windll.user32.PrintWindow(self.window.hwnd, saveDC.GetSafeHdc(), 0)

        if result != 1:
            self._cleanup(bitmap, saveDC, mfcDC)
            return None

        # Получаем данные
        bitmap_data = bitmap.GetBitmapBits(True)
        frame = self._convert_to_bgr(bitmap_data)

        self._cleanup(bitmap, saveDC, mfcDC)
        return frame

    def _convert_to_bgr(self, bitmap_data: bytes) -> np.ndarray:
        """Конвертирует bitmap данные в BGR формат."""
        frame = np.frombuffer(bitmap_data, dtype=np.uint8)
        frame = frame.reshape((self.window.height, self.window.width, 4))  # BGRX
        return frame[:, :, :3]  # BGR

    def _cleanup(self, bitmap, saveDC, mfcDC):
        """Освобождает ресурсы Windows."""
        win32gui.DeleteObject(bitmap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
