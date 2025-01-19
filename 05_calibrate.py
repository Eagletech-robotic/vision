import cv2 as cv
import numpy as np

from lib import common, camera

pattern_size = (4, 3)


def add_trackbars(window_name, cap):
    def on_trackbar(prop):
        def value_updated(value):
            global pattern_size
            if prop == "width" and value > 2:
                pattern_size = (value, pattern_size[1])
            elif prop == "height" and value > 2:
                pattern_size = (pattern_size[0], value)
            else:
                camera.monitor_property_changes(cap)(lambda: cap.set(prop, value))

        return value_updated

    global pattern_size
    common.init_window(window_name)
    cv.createTrackbar("width", "image", pattern_size[0], 9, on_trackbar("width"))
    cv.createTrackbar("height", "image", pattern_size[1], 9, on_trackbar("height"))
    prop = cv.CAP_PROP_FOCUS
    cv.createTrackbar("Focus", "image", int(cap.get(prop)), 500, on_trackbar(prop))


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)

    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    add_trackbars("image", cap)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:

        # Take a picture of a chessboard and find corners
        _, image = cap.read()

        # image = cv.resize(image, (1280, 720))
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corners = common.measure_time(lambda: cv.findChessboardCorners(
            gray_image, pattern_size, None,
            0
            + cv.CALIB_CB_ADAPTIVE_THRESH  # - Very slow
            + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK
        ), name="findChessboardCorners")()

        if ret:
            print(f"Found {len(corners)} corners")
            corners_original = corners.copy()

            imgpoints = common.measure_time(lambda: cv.cornerSubPix(
                gray_image, corners, (11, 11), (-1, -1), criteria
            ), name="cornerSubPix")()

            # Prepare object points (3D coordinates of chessboard corners)
            objpoints = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objpoints[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

            ret, mtx, dist, rvecs, tvecs = common.measure_time(lambda: cv.calibrateCamera(
                [objpoints], [imgpoints], image.shape[1::-1], None, None
            ), name="calibrateCamera")()

            print(f"Camera matrix: {mtx}")
            print(f"Distortion coefficients: {dist}")
            print(f"Rotation vectors: {rvecs}")
            print(f"Translation vectors: {tvecs}")

            cv.drawChessboardCorners(image, pattern_size, corners_original, True)
            position = tuple(corners[0][0].astype(int))
            cv.circle(image, position, 5, (0, 0, 255), 5)

        else:
            print("No corners found")

        common.show_in_window("image", image)

        c = cv.waitKey(100)

        if c == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
