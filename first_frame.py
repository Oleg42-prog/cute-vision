import cv2

CAMERA_NAME = '3'
DEVICE_INDEX = f'rtsp://admin:rumicon_rumicon@172.16.70.110:554/ISAPI/Streaming/Channels/{CAMERA_NAME}01'
cap = cv2.VideoCapture(DEVICE_INDEX)

if not cap.isOpened():
    print("Error opening video stream or file")

ret, frame = cap.read()

if ret:
    cv2.imwrite(f'convex_hulls/A0{CAMERA_NAME}.png', cv2.resize(frame, (int(frame.shape[1] / 1.5), int(frame.shape[0] / 1.5))))
    print("Frame saved successfully")
else:
    print("Error reading video frame")

cap.release()
