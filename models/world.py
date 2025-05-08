import math
import numpy as np

from lib import eagle_packet, common

BLEACHERS_DEFAULT = [
    # (0.075, 0.400, math.pi / 2),
    # (0.075, 1.325, math.pi / 2),
    # (0.775, 0.250, 0.0),
    # (0.825, 1.725, 0.0),
    # (1.100, 0.950, 0.0),
    # (3.0 - 0.075, 0.400, math.pi / 2),
    (3.0 - 0.075, 1.325, math.pi / 2),
    # (3.0 - 0.775, 0.250, 0.0),
    # (3.0 - 0.825, 1.725, 0.0),
    # (3.0 - 1.100, 0.950, 0.0),
]


class World:
    def __init__(self):
        self.team_color = None
        self.score = 0

        self.robot_detected: bool = False
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_orientation: float = 0.0

        self.opponent_detected: bool = False
        self.opponent_x: float = 0.0
        self.opponent_y: float = 0.0
        self.opponent_orientation: float = 0.0

        self.bleachers = BLEACHERS_DEFAULT

    def to_eagle_packet(self):
        robot_pose = (self.robot_x, self.robot_y, self.robot_orientation) if self.robot_x is not None else None
        opponent_pose = \
            (self.opponent_x, self.opponent_y, self.opponent_orientation) if self.opponent_x is not None else None

        return eagle_packet.build_payload(
            robot_colour=self.team_color or "blue",
            robot_detected=bool(robot_pose),
            robot_pose=robot_pose or (0.0, 0.0, 0.0),
            opponent_detected=bool(opponent_pose),
            opponent_pose=opponent_pose or (0.0, 0.0, 0.0),
            bleachers=self.bleachers,
        )

    def debug_image(self):
        """Generate an image with information about the world"""
        IMG_WIDTH, IMG_HEIGHT = 1920, 1080

        def positions(name, detected, x, y, orientation):
            return f"{name} X:{x:.3f} Y:{y:.3f} θ:{math.degrees(orientation):.0f}°" if detected \
                else f"{name} not detected"

        # Create black image
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Draw robot position
        txt = positions("Robot", self.robot_detected, self.robot_x, self.robot_y, self.robot_orientation)
        common.draw_text(img, txt, (10, 30))

        # Draw opponent position
        txt = positions("Opponent", self.opponent_detected, self.opponent_x, self.opponent_y,
                        self.opponent_orientation)
        common.draw_text(img, txt, (10, 60))

        return img
