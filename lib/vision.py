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
    ROBOT_BLUE = 2  # Legacy
    ROBOT_YELLOW = 6  # Legacy
    ROBOT_BLUE_LO = 1
    ROBOT_BLUE_HI = 5
    ROBOT_YELLOW_LO = 6
    ROBOT_YELLOW_HI = 10


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


def convert_marker_position_to_points(marker_position: MarkerPosition):
    points_3d = [
        [marker_position.x - marker_position.size / 2, marker_position.y - marker_position.size / 2, 0],  # Bottom Left
        [marker_position.x - marker_position.size / 2, marker_position.y + marker_position.size / 2, 0],  # Top Left
        [marker_position.x + marker_position.size / 2, marker_position.y + marker_position.size / 2, 0],  # Top Right
        [marker_position.x + marker_position.size / 2, marker_position.y - marker_position.size / 2, 0]  # Bottom Right
    ]
    rotation_steps = int(marker_position.rotation)
    if rotation_steps > 0:
        points_3d = points_3d[rotation_steps:] + points_3d[:rotation_steps]
    return np.array(points_3d)


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
        points_3d = known_markers_positions[marker_id]
        known_points_3d.extend(points_3d)

    # Convert to numpy arrays
    known_corners = np.array(known_corners)
    known_points_3d = np.array(known_points_3d)

    if len(known_corners) < 4:
        return False, None

    # Compute homography
    H, _ = cv.findHomography(known_corners, known_points_3d[:, :2])
    return True, H


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
    return pos.T[0]


def estimate_pose(corners, ids, known_markers_positions, camera_matrix, dist_coeffs):
    obj_points = []
    image_points = []

    for id, corner in zip(ids, corners):
        if id[0] in known_markers_positions:
            obj_points.extend(known_markers_positions[id[0]])
            image_points.extend(corner[0])

    obj_points = np.array(obj_points, np.float32)
    image_points = np.array(image_points, np.float32)

    if len(obj_points) > 0:
        ret, rvec, tvec = cv.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_IPPE)
        if ret:
            return True, rvec, tvec

    return False, None, None
