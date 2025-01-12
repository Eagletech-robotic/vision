from http.cookiejar import debug

import cv2 as cv
import numpy as np
from lib import common, detection


@common.measure_time
def find_and_draw_tags(image, aruco_detector):
    corners, ids, rejected = aruco_detector.detectMarkers(image)

    print(f"corners: {corners}")
    print(f"ids: {ids}")
    # print(f"rejected: {rejected}")

    new_image = image.copy()
    for id, corner in zip(ids, corners):
        center = corner[0].mean(axis=0).astype(int)
        cv.drawContours(new_image, corner.astype(int), -1, (0, 255, 0), 4)
        cv.circle(new_image, corner[0][0].astype(int), 5, (0, 0, 255), 5)
        cv.circle(new_image, corner[0][1].astype(int), 2, (128, 128, 255), 2)
        cv.putText(new_image, str(id[0]), center,
                   cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    return new_image


@common.measure_time
def find_aruco_positions(image, aruco_detector, known_markers_positions, marker_size):  # marker_size in centimeters
    # Detect ArUco markers
    new_image = image.copy()
    corners, ids, rejected = aruco_detector.detectMarkers(image)

    if ids is None:
        return None

    # Separate known and unknown markers
    known_corners = []
    known_points_3d = []
    unknown_markers = []

    for i, id in enumerate(ids):
        marker_id = id[0]
        if marker_id in known_markers_positions:
            # Get 3D points of known marker
            pos = known_markers_positions[marker_id]
            known_points_3d.extend([
                [pos[0] - marker_size / 2, pos[1] - marker_size / 2, 0],
                [pos[0] - marker_size / 2, pos[1] + marker_size / 2, 0],
                [pos[0] + marker_size / 2, pos[1] + marker_size / 2, 0],
                [pos[0] + marker_size / 2, pos[1] - marker_size / 2, 0]
            ])
            known_corners.extend(corners[i][0])
        else:
            unknown_markers.append((marker_id, corners[i]))

    # Convert to numpy arrays
    known_corners = np.array(known_corners)
    known_points_3d = np.array(known_points_3d)

    # Compute homography
    H, _ = cv.findHomography(known_corners, known_points_3d[:, :2])

    # Calculate positions of unknown markers
    unknown_positions = []
    for marker_id, corner in unknown_markers:
        # Take center point of marker
        center = corner[0].mean(axis=0)
        center_homogeneous = np.array([center[0], center[1], 1])

        # Apply homography
        world_point = H @ center_homogeneous
        world_point = world_point / world_point[2]  # Normalize homogeneous coordinates
        unknown_positions.append((marker_id, world_point[0], world_point[1], 0))

        # Draw marker with its world position
        cv.drawContours(new_image, corner.astype(int), -1, (0, 255, 0), 2)
        cv.circle(new_image, corner[0][0].astype(int), 3, (0, 0, 255), 3)
        cv.circle(new_image, corner[0][1].astype(int), 2, (128, 128, 255), 2)
        label = f"{marker_id} ({world_point[0]:.2f}, {world_point[1]:.2f})"
        cv.putText(new_image, label, center.astype(int), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return new_image, unknown_positions


aruco_detector = detection.build_aruco_detector()
image = cv.imread("assets/board_with_tags_2.jpg")

known_markers_positions = {
    2: (99, 78.5, 0),
    6: (100, 180, 0),
}
image_with_tags, positions = find_aruco_positions(image, aruco_detector, known_markers_positions, 7)
print(positions)
common.show_in_window("image_with_tags", image_with_tags)
cv.waitKey(0)
