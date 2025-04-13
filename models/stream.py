from lib import camera, common
from models.capture import Capture


class Stream:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = camera.capture(camera_index)

        camera.load_properties(self.cap, camera_index)
        self.camera_matrix, self.dist_coeffs = camera.load_calibration(camera_index)

    def capture(self):
        ret, image = self.cap.read()
        if not ret:
            print(f"Error capturing image from camera {self.camera_index}")
            exit(1)

        # common.show_in_window(f"Stream {self.camera_index}", image)
        return Capture(self, image)
