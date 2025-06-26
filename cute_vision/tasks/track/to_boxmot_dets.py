import numpy as np


def result_to_boxmot_dets(result):
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    dets = np.hstack([boxes, np.vstack([confs, cls]).T])
    return dets
