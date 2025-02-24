import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision


def rodrigues_to_euler(rvec):
    """Convert Rodrigues rotation vector to Euler angles (in degrees)"""
    R, _ = cv.Rodrigues(rvec)
    # Get rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # Convert to degrees
    return np.array([x, y, z]) * 180.0 / np.pi


def get_camera_position(rvec, tvec):
    """Get camera position in world coordinates"""
    R, _ = cv.Rodrigues(rvec)
    R = R.T  # Transpose rotation matrix
    pos = -R @ tvec  # Calculate camera position
    return pos


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

        if len(corners) > 0:
            obj_points = []
            image_points = []

            for id, corner in zip(ids, corners):
                cv.drawContours(image, corner.astype(int), -1, (0, 0, 255), 4)
                cv.circle(image, corner[0][0].astype(int), 8, (0, 0, 255), 8)
                cv.circle(image, corner[0][1].astype(int), 6, (0, 0, 255), 2)

                if id[0] in known_markers_positions:
                    image_points.extend(corner[0])
                    obj_points.extend(known_markers_positions[id[0]])

            obj_points = np.array(obj_points, np.float32)
            image_points = np.array(image_points, np.float32)

            if len(obj_points) > 0:
                ret, rvec, tvec = cv.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs)
                R, _ = cv.Rodrigues(rvec)
                print("ret", ret)
                print("rvec", rvec)
                print("tvec", tvec)

                # Get camera position and rotation
                camera_pos = get_camera_position(rvec, tvec)
                print(
                    f"Camera Position (mm): X={camera_pos[0, 0]:.1f}, Y={camera_pos[1, 0]:.1f}, Z={camera_pos[2, 0]:.1f}")

                camera_rot = rodrigues_to_euler(rvec)
                print(
                    f"Camera Rotation (deg): Roll={camera_rot[0]:.1f}, Pitch={camera_rot[1]:.1f}, Yaw={camera_rot[2]:.1f}")

                for id, corner in zip(ids, corners):
                    if id[0] in known_markers_positions:
                        continue

                    cv.drawContours(image, corner.astype(int), -1, (255, 0, 0), 4)
                    cv.circle(image, corner[0][0].astype(int), 8, (255, 0, 0), 8)
                    cv.circle(image, corner[0][1].astype(int), 6, (255, 0, 0), 2)

        common.show_in_window("image", image)

        c = cv.waitKey(100)
        if c == ord("q"):
            break


if __name__ == "__main__":
    main()
