import numpy as np
from math import atan2, cos, sin

from models.world import World
from lib import vision


class Analyser:
    def __init__(self, capture_1, capture_2):
        self.capture_1 = capture_1
        self.capture_2 = capture_2

    # ------------------------------------------------------------------ #
    #  public entry
    # ------------------------------------------------------------------ #
    def generate_world(self, persistent_state):
        world = World()
        world.score = persistent_state.score or 0
        world.team_color = self._find_team_color() or persistent_state.team_color

        # --- our robot ----------------------------------------------------
        robot_pose = self._calculate_pose(vision.OUR_ROBOT_MARKERS)
        if robot_pose:
            world.robot_detected = True
            world.robot_x, world.robot_y, world.robot_theta, rmse = robot_pose
            print(f"[Analyser] Robot pose RMSE = {rmse:.4f} m")

        # --- opponent robot ----------------------------------------------
        if world.team_color:
            opponent_pose = self._calculate_pose(
                self._opponent_marker_lookup(world.team_color)
            )
            if opponent_pose:
                world.opponent_detected = True
                world.opponent_x, world.opponent_y, world.opponent_theta, rmse = opponent_pose
                print(f"[Analyser] Opponent pose RMSE = {rmse:.4f} m")

        return world, persistent_state

    # ------------------------------------------------------------------ #
    #  pose helpers
    # ------------------------------------------------------------------ #
    def _opponent_marker_lookup(self, our_color):
        if our_color == "yellow":
            tag_lo, tag_hi = vision.MarkerId.ROBOT_BLUE_LO, vision.MarkerId.ROBOT_BLUE_HI
        else:
            tag_lo, tag_hi = vision.MarkerId.ROBOT_YELLOW_LO, vision.MarkerId.ROBOT_YELLOW_HI

        opponent_corners = {}
        for tag_id in range(tag_lo, tag_hi + 1):
            opponent_corners[tag_id] = vision.marker_corner_positions(
                0, 0, vision.MarkerHeight.OPPONENT_MARKER, vision.MarkerSize.OPPONENT_MARKER,
                vision.MarkerRotation.TOP_RIGHT
            )
        return opponent_corners

    def _calculate_pose(self, marker_lookup):
        """
        General 2-D rigid fit for any set of tags.

        marker_lookup: dict[tag_id] -> ndarray(shape=(4,3)) of corner (x,y,z) in tag frame
        returns: (x, y, theta, rmse) or None if not enough points
        """
        # (x,y) coordinates of each known point, in the reference frames of the tags and the field
        tag_frame_points = []
        field_frame_points = []

        for capture in [self.capture_1, self.capture_2]:
            ret = self._pose_corner_ids_from_capture(capture)
            if ret is None:
                continue
            rvec, tvec, corners, ids = ret

            for tag_id, image_corners in zip(ids, corners):
                tag_id = int(tag_id[0])
                tag_corners = marker_lookup.get(tag_id)
                if tag_corners is None:
                    continue

                for i in range(4):
                    image_point = image_corners[0][i]
                    world_point = vision.image_to_world_point(
                        image_point,
                        z_world=tag_corners[i][2],
                        rvec=rvec,
                        tvec=tvec,
                        camera_matrix=capture.camera_matrix,
                        dist_coeffs=capture.dist_coeffs,
                    )
                    tag_frame_points.append(tag_corners[i][:2])
                    field_frame_points.append(world_point[:2])

        if len(tag_frame_points) < 2:
            return None

        # Compute the rigid-body transform that converts the tag frame points to the field frame points
        # Return the x, y, theta, and the RMSE of the fit, which is the position and orientation of the robot.
        return self._solve_2d_rigid(tag_frame_points, field_frame_points)

    def _solve_2d_rigid(self, robot_points, world_points):
        """
        Least‑squares 2‑D rigid‑body transform.

        Parameters
        ----------
        robot_points : list/ndarray (N,2)
            Coordinates in the robot frame.
        world_points : list/ndarray (N,2)
            Corresponding coordinates in the world frame.

        Returns
        -------
        (x, y, theta, rmse)
            Translation (x,y), rotation (theta rad) and
            root‑mean‑squared error of the fit.
        """
        P = np.asarray(robot_points, dtype=float)
        W = np.asarray(world_points, dtype=float)

        P_cent = P.mean(axis=0)
        W_cent = W.mean(axis=0)
        P_hat = P - P_cent
        W_hat = W - W_cent

        num = np.sum(P_hat[:, 0] * W_hat[:, 1] - P_hat[:, 1] * W_hat[:, 0])
        den = np.sum(P_hat[:, 0] * W_hat[:, 0] + P_hat[:, 1] * W_hat[:, 1])
        theta = atan2(num, den)

        c, s = cos(theta), sin(theta)
        R = np.array([[c, -s], [s, c]])
        t = W_cent - R @ P_cent

        residuals = (R @ P.T).T + t - W
        rmse = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

        x, y = t
        return float(x), float(y), float(theta), rmse

    # ------------------------------------------------------------------ #
    #  misc helpers
    # ------------------------------------------------------------------ #
    def _find_team_color(self):
        x_values = [
            pose[2][0] for capture in [self.capture_1, self.capture_2]
            if (pose := capture.estimate_pose()) is not None
        ]
        if x_values:
            return "blue" if np.mean(x_values) > 1.5 else "yellow"
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
