def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    x1 = x
    y1 = y
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2
