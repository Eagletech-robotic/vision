# ----------------
# Calibration of a camera using a chessboard with AruCo markers.
# The calibration parameters are saved in the camera-settings folder.
# ----------------


import cv2 as cv
import numpy as np

from lib import common, camera

# ------------------------------
# ENTER CALIBRATION PARAMETERS HERE:
ARUCO_DICT = cv.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 8
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.0327  # square size in meters
MARKER_LENGTH = SQUARE_LENGTH / 2  # marker size in meters
LENGTH_PX = 1280  # total length of the page in pixels
MARGIN_PX = 20  # size of the margin in pixels
BOARD_OUTPUT_FILE = './assets/calibration_board.png'


def generate_calibration_board():
    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX * size_ratio)), marginSize=MARGIN_PX)
    cv.imwrite(BOARD_OUTPUT_FILE, img)
    return img


def add_trackbars(window_name, cap):
    def on_trackbar(prop):
        def value_updated(value):
            camera.monitor_property_changes(cap)(lambda: cap.set(prop, value))

        return value_updated

    global pattern_size
    common.init_window(window_name)
    prop = cv.CAP_PROP_FOCUS
    cv.createTrackbar("Focus", "image", int(cap.get(prop)), 500, on_trackbar(prop))


def main():
    # Generate calibration board
    generate_calibration_board()

    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)

    # Define the chessboard pattern
    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    detector_params = cv.aruco.DetectorParameters()

    # Add trackbars to the window
    add_trackbars("image", cap)

    all_charuco_corners = []
    all_charuco_ids = []
    shape = None
    detection_nb = 0
    save = False

    while True:
        # Take a picture of a chessboard and find corners
        cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
        _, image = cap.read()
        shape = image.shape[:2]
        # image = cv.resize(image, (1280, 720))
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = cv.aruco.detectMarkers(image, dictionary, parameters=detector_params)

        if marker_ids is not None and len(marker_ids) >= 4:
            ret, charuco_corners, charuco_ids = \
                cv.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            cv.aruco.drawDetectedMarkers(image, marker_corners, marker_ids, borderColor=(0, 255, 0))

            if ret and len(charuco_corners) >= 6 and len(charuco_ids) == len(charuco_corners):
                print(f"Detection {detection_nb}: {len(marker_ids)} markers, {len(charuco_ids)} charuco corners")
                detection_nb += 1
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

        common.show_in_window("image", image)

        c = cv.waitKey(500)

        if c == ord("s"):
            save = True
            break

        if c == ord("q"):
            break

    ret, camera_matrix, dist_coeffs, _rvecs, _tvecs, stddev_intrinsics, _stddev_extrinsics, per_view_errors = \
        common.measure_time(
            lambda:
            cv.aruco.calibrateCameraCharucoExtended(all_charuco_corners, all_charuco_ids, board, shape, None, None),
            name="calibrateCameraCharuco")()

    if ret:
        print("Calibration successful. RMS error: ", per_view_errors, " - Stddev intrinsics: ", stddev_intrinsics)
        print(f"Camera matrix: {camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")

        # Save calibration
        if save:
            camera.save_calibration(camera_matrix, dist_coeffs, camera_index)
            print("Calibration saved")

    else:
        print("Calibration failed")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
