import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)
    camera_matrix, dist_coeffs = camera.load_calibration(camera_index)

    aruco_detector = detection.build_aruco_detector()

    known_markers_positions = {
        vision.MarkerId.BOARD_TOP_RIGHT: [
            (235, 55, 0),
            (235 + 10, 55, 0),
            (235 + 10, 55 + 10, 0),
            (235, 55 + 10, 0),
        ],
        vision.MarkerId.BOARD_BOTTOM_RIGHT: [
            (235, 135, 0),
            (235 + 10, 135, 0),
            (235 + 10, 135 + 10, 0),
            (235, 135 + 10, 0),
        ],
    }

    while True:
        _, image = cap.read()
        corners, ids, _rejected = aruco_detector.detectMarkers(image)

        for id, corner in zip(ids, corners):
            cv.drawContours(image, corner.astype(int), -1, (0, 0, 255), 4)
            cv.circle(image, corner[0][0].astype(int), 8, (0, 0, 255), 8)
            cv.circle(image, corner[0][1].astype(int), 6, (0, 0, 255), 2)

        ret, camera_pos, camera_rot = \
            vision.estimate_pose(corners, ids, known_markers_positions, camera_matrix, dist_coeffs)
        if ret:
            print(f"Camera Position (mm): X={camera_pos[0, 0]:.1f}, Y={camera_pos[1, 0]:.1f}, Z={camera_pos[2, 0]:.1f}")
            print(
                f"Camera Rotation (deg): Roll={camera_rot[0]:.1f}, Pitch={camera_rot[1]:.1f}, Yaw={camera_rot[2]:.1f}")

        common.show_in_window("image", image)

        c = cv.waitKey(100)
        if c == ord("q"):
            break


if __name__ == "__main__":
    main()
