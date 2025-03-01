import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision
from lib.visualisation import World, TinCan, Robot, RobotColor, Position


def show_in_fullscreen(name, img1, img2, reconstructed):
    # Get screen resolution (adjust dynamically if needed)
    screen_width = 1920
    screen_height = 1080
    gap = 20
    top_height = int(screen_height * 0.5) - gap // 2

    # Create a black canvas
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Resize image while preserving aspect ratio
    def resize_to_fit(img, height):
        h, w = img.shape[:2]
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
        return cv.resize(img, (width, height))

    # Resize and center stream images
    img1_resized = resize_to_fit(img1, top_height)
    img2_resized = resize_to_fit(img2, top_height)

    top_row_width = img1_resized.shape[1] + img2_resized.shape[1] + gap
    x_offset_img1 = (screen_width - top_row_width) // 2
    x_offset_img2 = x_offset_img1 + img1_resized.shape[1] + gap

    canvas[0:top_height, x_offset_img1:x_offset_img1 + img1_resized.shape[1]] = img1_resized
    canvas[0:top_height, x_offset_img2:x_offset_img2 + img2_resized.shape[1]] = img2_resized

    # Resize and center the reconstructed image
    resized_reconstructed = resize_to_fit(reconstructed, screen_height - top_height - gap)
    new_height, new_width = resized_reconstructed.shape[:2]

    x_offset = (screen_width - new_width) // 2
    canvas[top_height + gap:screen_height, x_offset:x_offset + new_width] = resized_reconstructed

    # Show in full-screen mode
    cv.namedWindow(name, cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(name, canvas)


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
        self.aruco_detector = detection.build_aruco_detector()

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
            exit(1)
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
        corners, ids, _rejected = self.aruco_detector.detectMarkers(self.last_image)
        self.last_detection = corners, ids
        print(f"Detected ids", ids)

    def estimate_pose(self):
        corners, ids = self.last_detection
        if ids is None or len(ids) == 0:
            return

        ret, camera_pos, camera_rot = \
            vision.estimate_pose(corners, ids, known_markers_positions, self.camera_matrix, self.dist_coeffs)
        if ret:
            self.last_pose = camera_pos, camera_rot
            print(
                f"Camera {self.index} Position (mm): X={camera_pos[0, 0]:.1f}, Y={camera_pos[1, 0]:.1f}, Z={camera_pos[2, 0]:.1f}")

    def draw(self):
        image = self.last_image.copy()
        corners, ids = self.last_detection
        if ids is not None and len(ids) > 0:
            for id, corner in zip(ids, corners):
                cv.drawContours(image, corner.astype(int), -1, (0, 0, 255), 4)
                cv.circle(image, corner[0][0].astype(int), 8, (0, 0, 255), 8)
                cv.circle(image, corner[0][1].astype(int), 6, (0, 0, 255), 2)
                cv.putText(image, str(id[0]), tuple(corner[0][0].astype(int)), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                           5)
        return image

    def world_positions(self):
        corners, ids = self.last_detection
        if ids is None or len(ids) == 0:
            return []

        ret, H = vision.compute_homography(corners, ids, known_markers_positions)
        if not ret:
            return []

        known_ids = [
            vision.MarkerId.TIN_CAN,
            vision.MarkerId.ROBOT_BLUE_1,
            vision.MarkerId.ROBOT_BLUE_2,
            vision.MarkerId.ROBOT_YELLOW_1,
            vision.MarkerId.ROBOT_YELLOW_2
        ]

        _world_positions = []
        for id, corner in zip(ids, corners):
            marker_id = id[0]
            corner = corner[0]
            if marker_id not in known_ids:
                continue

            # Take center point of marker
            center = corner.mean(axis=0)
            center_homogeneous = np.array([center[0], center[1], 1])

            # Apply homography
            world_point = H @ center_homogeneous
            world_point = world_point / world_point[2]  # Normalize homogeneous coordinates

            print(f"Marker {marker_id} at {world_point}")
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
            elif marker_id == vision.MarkerId.ROBOT_BLUE_1 or marker_id == vision.MarkerId.ROBOT_BLUE_2:
                robot = Robot(Position(x, y, theta=0), RobotColor.BLUE)
                world.add_robot(robot)
            elif marker_id == vision.MarkerId.ROBOT_YELLOW_1 or marker_id == vision.MarkerId.ROBOT_YELLOW_2:
                robot = Robot(Position(x, y, theta=0), RobotColor.YELLOW)
                world.add_robot(robot)
        return world.render()


def main():
    # Select and initialize cameras
    common.run_hw_diagnostics()
    available_cameras = camera.list_available_cameras()

    indices = [cam["index"] for cam in available_cameras if cam["name"] == "W4DS--SN0001"]
    if len(indices) >= 2:
        cam_index_1, cam_index_2 = indices[0], indices[1]
    else:
        raise ValueError("Not enough cameras found with name 'W4DS--SN0001'")
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
