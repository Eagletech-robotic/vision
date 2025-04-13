import numpy as np


class World:
    def __init__(self):
        self.team_color = None
        self.score = 0

    def debug_image(self):
        img_width, img_height = 1920, 1080

        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        return img
