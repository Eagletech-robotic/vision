import math
import numpy as np

from lib import eagle_packet, common

BLEACHERS_DEFAULT = [
    (0.075, 0.400, 0.0),
    (0.075, 1.325, 0.0),
    (0.775, 0.250, math.pi / 2),
    (0.825, 1.725, math.pi / 2),
    (1.100, 0.950, math.pi / 2),
    (3.0 - 0.075, 0.400, 0.0),
    (3.0 - 0.075, 1.325, 0.0),
    (3.0 - 0.775, 0.250, math.pi / 2),
    (3.0 - 0.825, 1.725, math.pi / 2),
    (3.0 - 1.100, 0.950, math.pi / 2),
]


class World:
    def __init__(self):
        self.team_color = None
        self.score = 0

        self.robot_detected: bool = False
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.robot_theta: float = 0.0

        self.opponent_detected: bool = False
        self.opponent_x: float = 0.0
        self.opponent_y: float = 0.0
        self.opponent_theta: float = 0.0

        self.bleachers = BLEACHERS_DEFAULT

    def to_eagle_packet(self):
        robot_pose = (self.robot_x, self.robot_y, self.robot_theta) if self.robot_detected else None
        opponent_pose = (self.opponent_x, self.opponent_y, self.opponent_theta) if self.opponent_detected else None

        return eagle_packet.build_payload(
            robot_colour=self.team_color or "blue",
            robot_detected=self.robot_detected,
            robot_pose=robot_pose or (0.0, 0.0, 0.0),
            opponent_detected=self.opponent_detected,
            opponent_pose=opponent_pose or (0.0, 0.0, 0.0),
            bleachers=self.bleachers,
        )

    def debug_image(self, log_entries):
        """Generate an image with information about the world"""
        IMG_WIDTH, IMG_HEIGHT = 1920, 1080
        INTERLINE = 30
        last_y = 0

        def interline(factor=1.0):
            nonlocal last_y
            last_y += int(INTERLINE * factor)

        def positions(name, detected, x, y, theta):
            return f"{name} X: {x:.3f} Y: {y:.3f} theta: {math.degrees(theta):.0f}" if detected \
                else f"{name} not detected"

        # Create black image
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)

        # Sort log entries by time and display them
        sorted_entries = sorted(log_entries, key=lambda entry: entry[0])
        for _, line in sorted_entries:
            interline()
            common.draw_text(img, line, (10, last_y))

        # Draw robot position
        interline(0.5)
        txt = positions("Robot", self.robot_detected, self.robot_x, self.robot_y, self.robot_theta)
        interline()
        common.draw_text(img, txt, (10, last_y))

        # Draw opponent position
        txt = positions("Opponent", self.opponent_detected, self.opponent_x, self.opponent_y,
                        self.opponent_theta)
        interline()
        common.draw_text(img, txt, (10, last_y))

        return img
