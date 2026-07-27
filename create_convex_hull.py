import cv2
import numpy as np

CAMERA_NAME = '04'
path = []


def draw_circle(event, x, y, flags, param):
    global path
    if event == cv2.EVENT_LBUTTONDOWN:
        path.append((x, y))


img = cv2.imread(f'convex_hulls/A{CAMERA_NAME}.png')
original_img = img.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)


def compress(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


while True:

    cv2.imshow('image', img)

    if path:

        img = original_img.copy()
        hull = cv2.convexHull(np.array(path))

        for x, y in path:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

        for i in hull:
            x, y = i[0]
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        cv2.drawContours(img, [hull], 0, (0, 255, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        img = original_img.copy()
        path = []

cv2.destroyAllWindows()
np.save(f'convex_hulls/A{CAMERA_NAME}.npy', np.array(path))
