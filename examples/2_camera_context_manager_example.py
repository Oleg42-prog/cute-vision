import cv2
from cute_vision.cameras import DeviceCamera


with DeviceCamera(device_index=0) as camera:
    while True:
        frame = camera.fetch_frame()
        if frame is not None:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
