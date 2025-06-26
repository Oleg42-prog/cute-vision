import cv2


def draw_box(frame, box, color, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return frame


def draw_boxes(frame, boxes, colors, thickness=2):
    for box, color in zip(boxes, colors):
        frame = draw_box(frame, box, color, thickness)
    return frame
