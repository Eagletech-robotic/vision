from lib import camera, common
from models.capture import Capture
from datetime import datetime


class Stream:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = camera.capture(camera_index)

        camera.load_properties(self.cap, camera_index)
        self.camera_matrix, self.dist_coeffs = camera.load_calibration(camera_index)

    def capture(self):
        time = datetime.now()

        # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
        self.cap.grab()

        ret, image = self.cap.read()
        if not ret:
            print(f"Error capturing image from camera {self.camera_index}")
            exit(1)

        return Capture(self, image, time)
