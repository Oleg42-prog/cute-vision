import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack, BotSort, DeepOcSort, OcSort, StrongSort


tracker = ByteTrack(
    track_thresh=0.25,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30
)

frame = cv2.imread('humans.png')

model = YOLO('yolov8n.pt')

results = model(frame, classes=[0], conf=0.6, verbose=False)
result = results[0]

print(result)
