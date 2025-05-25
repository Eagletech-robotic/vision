# ----------------
# Find the field's Aruco markers and estimate the camera's pose.
# ----------------

import cv2 as cv

from lib import common, camera, detection, vision


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)
    camera_matrix, dist_coeffs = camera.load_calibration(camera_index)

    aruco_detector = detection.build_aruco_detector()

    while True:
        cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
        _, image = cap.read()
        corners, ids, _rejected = aruco_detector.detectMarkers(image)
        detection.draw_aruco_markers(image, corners, ids)

        ret, rvec, tvec = \
            vision.estimate_pose(corners, ids, vision.FIELD_MARKERS, camera_matrix, dist_coeffs)
        if ret:
            euler = vision.rodrigues_to_euler(rvec)
            camera_pos = vision.get_camera_position(rvec, tvec)
            print(f"Camera Position (m): X={camera_pos[0]:.3f}, Y={camera_pos[1]:.3f}, Z={camera_pos[2]:.3f}")
            print(f"Camera Rotation (deg): Roll={euler[0]:.1f}, Pitch={euler[1]:.1f}, Yaw={euler[2]:.1f}")

        common.show_in_window("image", image)

        c = cv.waitKey(100)
        if c == ord("q"):
            break


if __name__ == "__main__":
    main()
