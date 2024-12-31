import cv2 as cv
import numpy as np
from lib import common, detection


@common.measure_time
def identify_tags(img):
    corners, ids, rejected = detection.aruco_detector.detectMarkers(img)

    new_img = img.copy()
    for id, corner in zip(ids, corners):
        center = corner[0].mean(axis=0).astype(int)
        cv.drawContours(new_img, corner.astype(np.int32), -1, (0, 255, 0), 4)
        cv.putText(new_img, str(id[0]), tuple(center),
                   cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    return new_img


@common.measure_time
def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    result = cv.filter2D(img, -1, kernel)

    return result


source = cv.imread("assets/board_with_tags_2.jpg")
# source = cv.resize(source, (800, 600))

image = identify_tags(source)
common.show_in_window("image", image)

# sharp = sharpen(source)
# image = identify_tags(sharp)
# common.show_in_window("sharp", image)

cv.waitKey(0)
cv.destroyAllWindows()
