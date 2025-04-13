import cv2 as cv
from lib import common


def build_aruco_detector():
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    aruco_params = cv.aruco.DetectorParameters()
    aruco_params.minMarkerPerimeterRate = 0.003
    # common.print_fields(aruco_params)
    detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
    return detector
