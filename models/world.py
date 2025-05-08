import numpy as np
import cv2 as cv


class World:
    def __init__(self):
        self.team_color = None
        self.score = 0

        self.robot_detected: bool = False
        self.robot_x: float
        self.robot_y: float
        self.robot_orientation_deg: float

    def debug_image(self):
        img_width, img_height = 1920, 1080
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        if self.robot_detected:
            txt = f"Robot X:{self.robot_x:.3f} Y:{self.robot_y:.3f} θ:{self.robot_orientation_deg:.0f}°"
            cv.putText(img, txt, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img
