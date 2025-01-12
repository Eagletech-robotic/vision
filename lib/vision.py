import cv2 as cv
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class MarkerId(IntEnum):
    TIN_CAN = 47
    BOARD_BOTTOM_LEFT = 22
    BOARD_TOP_LEFT = 20
    BOARD_TOP_RIGHT = 21
    BOARD_BOTTOM_RIGHT = 23
    ROBOT_BLUE = 2
    ROBOT_YELLOW = 6


class MarkerRotation(IntEnum):
    BOTTOM_LEFT = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_RIGHT = 3


@dataclass
class MarkerPosition:
    x: float
    y: float
    z: float
    size: float
    rotation: MarkerRotation


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

        points_3d = [
            [pos.x - pos.size / 2, pos.y - pos.size / 2, 0],  # Bottom Left
            [pos.x - pos.size / 2, pos.y + pos.size / 2, 0],  # Top Left
            [pos.x + pos.size / 2, pos.y + pos.size / 2, 0],  # Top Right
            [pos.x + pos.size / 2, pos.y - pos.size / 2, 0]  # Bottom Right
        ]
        rotation_steps = int(pos.rotation)
        if rotation_steps > 0:
            points_3d = points_3d[rotation_steps:] + points_3d[:rotation_steps]
        known_points_3d.extend(points_3d)

    # Convert to numpy arrays
    known_corners = np.array(known_corners)
    known_points_3d = np.array(known_points_3d)

    # Compute homography
    H, _ = cv.findHomography(known_corners, known_points_3d[:, :2])
    return H
