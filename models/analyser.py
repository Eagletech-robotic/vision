import numpy as np

from models.world import World
from lib import vision


def _find_team_color(captures):
    x_values = [pose[2][0] for capture in captures if (pose := capture.estimate_pose()) is not None]
    if len(x_values) > 0:
        return "blue" if np.mean(x_values) > 150 else "yellow"
    else:
        return None


def _robot_pose_from_capture(capture):
    """ Return (x, y, theta_deg) or None if tag not seen."""

    # TO BE DONE
    return None


def _opponent_pose_from_capture(capture, our_color):
    """ Return (x, y, theta_deg) or None if tag not seen."""
    pose = capture.estimate_pose()
    if pose is None:
        return None

    corners, ids = capture._detection()
    if ids is None:
        return None

    # Tag id range of the opponent robot
    opponent_lo, opponent_hi = \
        (vision.MarkerId.ROBOT_BLUE_LO, vision.MarkerId.ROBOT_BLUE_HI) if our_color == "yellow" \
            else (vision.MarkerId.ROBOT_YELLOW_LO, vision.MarkerId.ROBOT_YELLOW_HI)

    cam_rvec, cam_tvec, *_ = pose
    K, D = capture.camera_matrix, capture.dist_coeffs

    for tag_id, tag_corners in zip(ids, corners):
        if not (opponent_lo <= tag_id[0] <= opponent_hi):
            continue

        # 4× image → world  ------------------------------------------------
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

        # theta: vector along tag local +X (corner0 → corner3)
        dx, dy = world_points[3][:2] - world_points[0][:2]
        theta = np.arctan2(-dy, dx)  # Y‑flip again

        return x, y, theta
    return None


def generate_world(capture_1, capture_2, persistent_state):
    world = World()

    world.score = persistent_state.score or 0
    world.team_color = _find_team_color([capture_1, capture_2]) or persistent_state.team_color

    for capture in (capture_1, capture_2):
        if not world.robot_detected:
            robot_pose = _robot_pose_from_capture(capture)
            if robot_pose:
                world.robot_detected = True
                world.robot_x, world.robot_y, world.robot_theta = robot_pose

        if not world.opponent_detected and world.team_color:
            opponent_pose = _opponent_pose_from_capture(capture, world.team_color)
            if opponent_pose:
                world.opponent_detected = True
                world.opponent_x, world.opponent_y, world.opponent_theta = opponent_pose

    return world, persistent_state
