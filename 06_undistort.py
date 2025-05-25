# ----------------
# Calibration of a camera using a chessboard with AruCo markers.
# The calibration parameters are saved in the camera-settings folder.
# ----------------


import cv2 as cv
import numpy as np

from lib import common, camera


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)
    camera_matrix, dist_coeffs = camera.load_calibration(camera_index)

    # Calculate the new camera matrix
    cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
    _, image = cap.read()
    h, w = image.shape[:2]
    new_camera_matrix, _roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    print(f"Image shape: {w}x{h}")
    print("Camera matrix:", camera_matrix)
    print("New camera matrix:", new_camera_matrix)

    while True:
        # Capture an image and draw a rectangle to assess the distortion
        cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
        _, image = cap.read()
        cv.rectangle(image, (50, 50), (image.shape[1] - 50, image.shape[0] - 50), (0, 255, 0), 4)

        # Undistort the image
        undistorted_image = cv.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        cv.rectangle(undistorted_image, (50, 50), (image.shape[1] - 50, image.shape[0] - 50), (0, 0, 255), 2)

        common.show_in_window("image", image)
        common.show_in_window("undistorted_image", undistorted_image)

        c = cv.waitKey(100)

        if c == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
