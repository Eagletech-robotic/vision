import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision
from lib.visualisation import World, TinCan, Robot, RobotColor, Position


def show_in_fullscreen(name, img1, img2, reconstructed):
    # Ensure the images have the same width
    width = max(img1.shape[1], img2.shape[1], reconstructed.shape[1])

    # Resize images to match width
    img1 = cv.resize(img1, (width // 2, img1.shape[0]))
    img2 = cv.resize(img2, (width // 2, img2.shape[0]))
    reconstructed = cv.resize(reconstructed, (width, reconstructed.shape[0]))

    # Concatenate top row (two cameras)
    top_row = np.hstack([img1, img2])

    # Stack the top row with the bottom reconstructed image
    final_display = np.vstack([top_row, reconstructed])

    cv.imshow(name, final_display)


known_markers_positions = {
    vision.MarkerId.BOARD_TOP_LEFT: [
        (55, 55, 0),
        (55 + 10, 55, 0),
        (55 + 10, 55 + 10, 0),
        (55, 55 + 10, 0),
    ],
    vision.MarkerId.BOARD_BOTTOM_LEFT: [
        (55, 135, 0),
        (55 + 10, 135, 0),
        (55 + 10, 135 + 10, 0),
        (55, 135 + 10, 0),
    ],
    vision.MarkerId.BOARD_TOP_RIGHT: [
        (235, 55, 0),
        (235 + 10, 55, 0),
        (235 + 10, 55 + 10, 0),
        (235, 55 + 10, 0),
    ],
    vision.MarkerId.BOARD_BOTTOM_RIGHT: [
        (235, 135, 0),
        (235 + 10, 135, 0),
        (235 + 10, 135 + 10, 0),
        (235, 135 + 10, 0),
    ],
}


class Stream:
    def __init__(self, index, camera_index):
        self.index = index
        self.camera_index = camera_index

        cap = camera.capture(self.camera_index)
        camera.load_properties(cap, camera_index)
        self.camera_matrix, self.dist_coeffs = camera.load_calibration(camera_index)
        self.last_image = None
        self.last_pose = None
        self.last_detection = None

    def find_world_positions(self):
        ret = self.capture_image()
        if not ret:
            print(f"Error capturing image from camera {self.index}")
            return
        self.detect()
        self.estimate_pose()
        return self.world_positions()

    def capture_image(self):
        ret, image = camera.capture(self.camera_index).read()
        if ret:
            self.last_image = image
            return True
        return False

    def detect(self):
        aruco_detector = detection.build_aruco_detector()
        corners, ids, _rejected = aruco_detector.detectMarkers(self.last_image)
        self.last_detection = corners, ids

    def estimate_pose(self):
        corners, ids = self.last_detection
        ret, camera_pos, camera_rot = \
            vision.estimate_pose(corners, ids, known_markers_positions, self.camera_matrix, self.dist_coeffs)
        if ret:
            self.last_pose = camera_pos, camera_rot
            print(
                f"Camera {self.index} Position (mm): X={camera_pos[0, 0]:.1f}, Y={camera_pos[1, 0]:.1f}, Z={camera_pos[2, 0]:.1f}")

    def draw(self):
        image = self.last_image.copy()
        corners, ids = self.last_detection
        for id, corner in zip(ids, corners):
            cv.drawContours(image, corner.astype(int), -1, (0, 0, 255), 4)
            cv.circle(image, corner[0][0].astype(int), 8, (0, 0, 255), 8)
            cv.circle(image, corner[0][1].astype(int), 6, (0, 0, 255), 2)
        return image

    def world_positions(self):
        corners, ids = self.last_detection
        H = vision.compute_homography(corners, ids, known_markers_positions)
        known_ids = [
            vision.MarkerId.TIN_CAN,
            vision.MarkerId.ROBOT_BLUE,
            vision.MarkerId.ROBOT_YELLOW,
        ]

        _world_positions = []
        for id, corner in zip(ids, corners):
            marker_id = id[0]
            corner = corner[0]
            if marker_id in known_ids:
                continue

            # Take center point of marker
            center = corner.mean(axis=0)
            center_homogeneous = np.array([center[0], center[1], 1])

            # Apply homography
            world_point = H @ center_homogeneous
            world_point = world_point / world_point[2]  # Normalize homogeneous coordinates

            _world_positions.append((marker_id, world_point[0], world_point[1], world_point[2]))

        return _world_positions


class Reconstruction:
    def __init__(self, stream_1, stream_2):
        self.stream_1 = stream_1
        self.stream_2 = stream_2
        self.world_positions = None

    def run(self):
        world_positions_1 = self.stream_1.find_world_positions()
        world_positions_2 = self.stream_2.find_world_positions()
        self.world_positions = world_positions_1 + world_positions_2

    def draw(self):
        world = World(blocking=False, off_screen=True)
        for marker_id, x, y, z in self.world_positions:
            if marker_id == vision.MarkerId.TIN_CAN:
                world.add_tin_can(TinCan(Position(x, y)))
            elif marker_id == vision.MarkerId.ROBOT_BLUE:
                robot = Robot(Position(x, y, theta=0), RobotColor.BLUE)
                world.add_robot(robot)
            elif marker_id == vision.MarkerId.ROBOT_YELLOW:
                robot = Robot(Position(x, y, theta=180), RobotColor.YELLOW)
                world.add_robot(robot)
        return world.render()


def main():
    # Select and initialize cameras
    common.run_hw_diagnostics()
    available_cameras = camera.list_available_cameras()
    if (len(available_cameras) < 2):
        print("Error: This script requires two cameras to be connected.")
        return
    cam_index_1, cam_index_2 = available_cameras[-2]["index"], available_cameras[-1]["index"]
    print("Available cameras: ", available_cameras, " - Using cameras: ", cam_index_1, cam_index_2)

    # Initialize streams and world
    stream_1 = Stream(1, cam_index_1)
    stream_2 = Stream(2, cam_index_2)
    reconstruction = Reconstruction(stream_1, stream_2)

    # Run reconstruction
    running = True

    while running:
        reconstruction.run()
        reconstructed = reconstruction.draw()
        img1 = stream_1.draw()
        img2 = stream_2.draw()
        show_in_fullscreen("Reconstruction", img1, img2, reconstructed)

        c = cv.waitKey(50)
        if c == ord("q"):
            running = False


if __name__ == "__main__":
    main()
