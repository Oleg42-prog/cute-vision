import cv2
from cute_vision.cameras import DeviceCamera


for frame in DeviceCamera(device_index=0).frames():
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
