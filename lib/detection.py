import cv2 as cv


def init_aruco_detector():
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    aruco_params = cv.aruco.DetectorParameters()
    # common.print_fields(aruco_params)
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector


aruco_detector = init_aruco_detector()
