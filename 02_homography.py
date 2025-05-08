# ----------------
# Detect ArUco markers and compute their position by homography using the known markers.
# The positions are drawn on the image.
# ----------------

import cv2 as cv
import numpy as np
from lib import common, detection, vision


@common.measure_time
def find_and_draw_tags(image, aruco_detector):
    corners, ids, rejected = aruco_detector.detectMarkers(image)

    new_image = image.copy()
    detection.draw_aruco_markers(new_image, corners, ids)
    return new_image


@common.measure_time
# Draw unknown markers with their world positions
def draw_aruco_positions(image, corners, ids, H, known_ids):
    new_image = image.copy()
    for id, corner in zip(ids, corners):
        marker_id = id[0]
        corner = corner[0]
        if marker_id in known_ids:
            continue

        # Take center point of marker
        center = corner.mean(axis=0)
        center_homogeneous = np.array([center[0], center[1], 1])

        # Apply homography
        world_point = H @ center_homogeneous
        world_point = world_point / world_point[2]  # Normalize homogeneous coordinates

        # Draw marker with its world position
        cv.drawContours(new_image, [corner.astype(int)], -1, (0, 255, 0), 2)
        cv.circle(new_image, corner[0].astype(int), 3, (0, 0, 255), 3)
        cv.circle(new_image, corner[1].astype(int), 2, (128, 128, 255), 2)
        label = f"{marker_id} ({world_point[0]:.2f}, {world_point[1]:.2f})"
        cv.putText(new_image, label, center.astype(int), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return new_image


def main():
    aruco_detector = detection.build_aruco_detector()
    image = cv.imread("assets/board_with_tags_2.jpg")

    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(image)
    if ids is None:
        return None

    # Compute homography
    known_markers_positions = {
        2: vision.marker_corner_positions(.99, .785, 0, .07, vision.MarkerRotation.BOTTOM_LEFT),
        6: vision.marker_corner_positions(1., 1.8, 0, .07, vision.MarkerRotation.BOTTOM_LEFT),
    }
    known_ids = list(known_markers_positions.keys())
    H = vision.compute_homography(corners, ids, known_markers_positions)

    # Draw tags
    image_with_tags = draw_aruco_positions(image, corners, ids, H, known_ids)
    common.show_in_window("image_with_tags", image_with_tags)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
