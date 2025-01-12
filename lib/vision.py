import cv2 as cv
import numpy as np
from dataclasses import dataclass


@dataclass
class MarkerPosition:
    x: float
    y: float
    z: float
    size: float


def compute_homography(corners, ids, known_markers_positions):
    # Separate known and unknown markers
    known_corners = []
    known_points_3d = []

    for id, corner in zip(ids, corners):
        marker_id = id[0]
        corner = corner[0]
        if marker_id not in known_markers_positions:
            continue

        # Get 3D points of known marker
        known_corners.extend(corner)
        pos = known_markers_positions[marker_id]
        known_points_3d.extend([
            [pos.x - pos.size / 2, pos.y - pos.size / 2, 0],
            [pos.x - pos.size / 2, pos.y + pos.size / 2, 0],
            [pos.x + pos.size / 2, pos.y + pos.size / 2, 0],
            [pos.x + pos.size / 2, pos.y - pos.size / 2, 0]
        ])

    # Convert to numpy arrays
    known_corners = np.array(known_corners)
    known_points_3d = np.array(known_points_3d)

    # Compute homography
    H, _ = cv.findHomography(known_corners, known_points_3d[:, :2])
    return H
