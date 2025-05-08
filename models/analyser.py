import numpy as np

from models.world import World
from lib import vision


class Analyser:
    def __init__(self, capture_1, capture_2):
        self.capture_1 = capture_1
        self.capture_2 = capture_2

    def generate_world(self, persistent_state):
        world = World()

        world.score = persistent_state.score or 0
        world.team_color = self._find_team_color() or persistent_state.team_color

        for capture in (self.capture_1, self.capture_2):
            if not world.robot_detected:
                robot_pose = self._robot_pose_from_capture(capture)
                if robot_pose:
                    world.robot_detected = True
                    world.robot_x, world.robot_y, world.robot_theta = robot_pose

            if not world.opponent_detected and world.team_color:
                opponent_pose = self._opponent_pose_from_capture(capture, world.team_color)
                if opponent_pose:
                    world.opponent_detected = True
                    world.opponent_x, world.opponent_y, world.opponent_theta = opponent_pose

        return world, persistent_state

    def _find_team_color(self):
        x_values = [
            pose[2][0] for capture in [self.capture_1, self.capture_2]
            if (pose := capture.estimate_pose()) is not None
        ]
        if len(x_values) > 0:
            return "blue" if np.mean(x_values) > 1.5 else "yellow"
        else:
            return None

    def _robot_pose_from_capture(self, capture):
        """ Return (x, y, theta_deg) or None if tag not seen."""
        rvec, tvec, corners, ids = self._pose_corner_ids_from_capture(capture)

        # TO BE DONE

        return None

    def _opponent_pose_from_capture(self, capture, our_color):
        """ Return (x, y, theta_deg) or None if tag not seen."""
        rvec, tvec, corners, ids = self._pose_corner_ids_from_capture(capture)

        # Tag id range of the opponent robot
        opponent_lo, opponent_hi = \
            (vision.MarkerId.ROBOT_BLUE_LO, vision.MarkerId.ROBOT_BLUE_HI) if our_color == "yellow" \
                else (vision.MarkerId.ROBOT_YELLOW_LO, vision.MarkerId.ROBOT_YELLOW_HI)

        for tag_id, tag_corners in zip(ids, corners):
            if not (opponent_lo <= tag_id[0] <= opponent_hi):
                continue

            # 4× image → world  ------------------------------------------------
            world_points = []
            for image_point in tag_corners[0]:
                world_point = vision.image_to_world_point(
                    image_point, z_world=.51,
                    rvec=rvec, tvec=tvec, camera_matrix=capture.camera_matrix, dist_coeffs=capture.dist_coeffs
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

    def _pose_corner_ids_from_capture(self, capture):
        pose = capture.estimate_pose()
        if pose is None:
            return None
        rvec, tvec = pose[:2]

        corners, ids = capture._detection()
        if ids is None:
            return None

        return rvec, tvec, corners, ids
