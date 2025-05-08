import numpy as np

from models.world import World
from lib import vision


def _find_team_color(captures):
    x_values = [pose[2][0] for capture in captures if (pose := capture.estimate_pose()) is not None]
    if len(x_values) > 0:
        return "blue" if np.mean(x_values) > 150 else "yellow"
    else:
        return None


def _robot_pose_from_capture(capture, team_color: str):
    """Return (x, y, theta_deg) or None if tag not seen."""
    pose = capture.estimate_pose()
    if pose is None:
        return None

    corners, ids = capture._detection()
    if ids is None:
        return None

    # tag id range to keep
    lo = vision.MarkerId.ROBOT_BLUE_LO if team_color == "blue" else vision.MarkerId.ROBOT_YELLOW_LO
    hi = vision.MarkerId.ROBOT_BLUE_HI if team_color == "blue" else vision.MarkerId.ROBOT_YELLOW_HI

    cam_rvec, cam_tvec, *_ = pose
    K, D = capture.camera_matrix, capture.dist_coeffs

    for tag_id, tag_corners in zip(ids, corners):
        if not (lo <= tag_id[0] <= hi):
            continue

        # 4× image → world (mm) ------------------------------------------------
        world_points = []
        for p in tag_corners[0]:
            world_point = vision.image_to_world_point(
                p, z_world=.51,
                rvec=cam_rvec, tvec=cam_tvec,
                camera_matrix=K, dist_coeffs=D
            )
            world_points.append(world_point)
        world_points = np.array(world_points)  # shape (4,3)

        centre = world_points.mean(axis=0)
        x, y = centre[:2]

        # orientation: vector along tag local +X (corner0 → corner3)
        dx, dy = world_points[3][:2] - world_points[0][:2]
        theta_rad = np.arctan2(-dy, dx)  # Y‑flip again
        theta_deg = np.rad2deg(theta_rad)
        if theta_deg > 180:
            theta_deg -= 360
        elif theta_deg < -180:
            theta_deg += 360

        return x, y, theta_deg
    return None


def generate_world(capture_1, capture_2, persistent_state):
    world = World()
    world.score = persistent_state.score or 0

    world.team_color = _find_team_color([capture_1, capture_2]) or persistent_state.team_color

    if world.team_color:
        for cap in (capture_1, capture_2):
            pose = _robot_pose_from_capture(cap, world.team_color)
            if pose:
                world.robot_x, world.robot_y, world.robot_orientation_deg = pose
                break

    return world, persistent_state
