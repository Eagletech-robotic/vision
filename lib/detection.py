import cv2 as cv


def build_aruco_detector():
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    aruco_params = cv.aruco.DetectorParameters()
    aruco_params.minMarkerPerimeterRate = 0.003
    # common.print_fields(aruco_params)
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector


def draw_aruco_markers(image, corners, ids):
    if ids is None or len(ids) == 0:
        return

    for id, corner in zip(ids, corners):
        center = corner[0].mean(axis=0).astype(int)
        cv.drawContours(image, corner.astype(int), -1, (0, 255, 0), 4)
        cv.circle(image, corner[0][0].astype(int), 5, (0, 0, 255), 5)
        cv.circle(image, corner[0][1].astype(int), 2, (128, 128, 255), 2)
        cv.putText(image, str(id[0]), center,
                   cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
