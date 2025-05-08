import numpy as np
import cv2 as cv


class World:
    def __init__(self):
        self.team_color = None
        self.score = 0

        self.robot_x_cm: float | None = None
        self.robot_y_cm: float | None = None
        self.robot_orientation_deg: float | None = None

    def debug_image(self):
        img_width, img_height = 1920, 1080
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        if self.robot_x_cm is not None:
            txt = f"Robot  X:{self.robot_x_cm:.0f} cm  Y:{self.robot_y_cm:.0f} cm  θ:{self.robot_orientation_deg:.0f}°"
            cv.putText(img, txt, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img
