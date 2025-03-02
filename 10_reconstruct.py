import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision
from lib.visualisation import World, TinCan, Robot, RobotColor, Position, Webcam


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

        ret, rvec, tvec = \
            vision.estimate_pose(corners, ids, known_markers_positions, self.camera_matrix, self.dist_coeffs)

        if ret:
            self.last_pose = rvec, tvec
            pos = vision.get_camera_position(rvec, tvec)
            print(f"Camera {self.index} Position (mm): X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")

    def draw_cross(self, image, world_point, text=None):
        if self.last_pose is None:
            return

        camera_rvec, camera_tvec = self.last_pose
        image_points, _ = cv.projectPoints(world_point, camera_rvec, camera_tvec, self.camera_matrix, self.dist_coeffs)
        u, v = map(int, image_points.ravel())
        half_size = 50
        cv.line(image, (u - half_size, v), (u + half_size, v), (0, 255, 0), 3)
        cv.line(image, (u, v - half_size), (u, v + half_size), (0, 255, 0), 3)
        if text:
            cv.putText(image, text, (u, v), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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
        self.draw_cross(image, np.array([[150., 200., 0.]]))
        return image

    def image_to_world_point(self, image_point, z_world):
        rvec, tvec = self.last_pose
        R, _ = cv.Rodrigues(rvec)

        image_points = np.array([[[image_point[0], image_point[1]]]], dtype=np.float32)
        undistorted_points = cv.undistortPoints(image_points, self.camera_matrix, self.dist_coeffs)

        # The undistorted point is in normalized camera coordinates
        x_normalized, y_normalized = undistorted_points[0, 0]

        # Create a ray in camera coordinates
        ray_camera = np.array([x_normalized, y_normalized, 1.0])

        # Transform ray to world coordinates
        ray_world = np.matmul(R.T, ray_camera)

        # Camera center in world coordinates
        camera_center = -np.matmul(R.T, tvec.reshape(3, 1)).flatten()

        # Calculate the scaling factor to reach the plane at z=z_world
        # We need to solve: camera_center[2] + s * ray_world[2] = z_world
        s = (z_world - camera_center[2]) / ray_world[2]

        # Calculate the world point
        world_point = camera_center + s * ray_world

        return world_point

    def world_positions(self):
        corners, ids = self.last_detection
        if ids is None or len(ids) == 0:
            return []

        world_positions = []
        for id, corner in zip(ids, corners):
            marker_id = id[0]
            corner = corner[0]
            if (marker_id == vision.MarkerId.TIN_CAN or
                    vision.MarkerId.ROBOT_BLUE_LO <= marker_id <= vision.MarkerId.ROBOT_BLUE_HI or
                    vision.MarkerId.ROBOT_YELLOW_LO <= marker_id <= vision.MarkerId.ROBOT_YELLOW_HI):
                center = corner.mean(axis=0)
                z_world = -8.5 if marker_id == vision.MarkerId.TIN_CAN else -32.0
                world_point = self.image_to_world_point(center, z_world)

                print(f"Marker {marker_id} at {world_point[:2]}")
                world_positions.append((marker_id, world_point[0], world_point[1], world_point[2]))

        return world_positions


class Reconstruction:
    def __init__(self, streams):
        self.streams = streams
        self.world_positions = None
        self.world = World(blocking=False, off_screen=True)

    def run(self):
        self.world_positions = sum((stream.find_world_positions() for stream in self.streams), [])

    def draw(self):
        self.world.empty()
        for i, stream in enumerate(self.streams):
            last_pose = stream.last_pose
            if last_pose is not None:
                rvec, tvec = last_pose
                position = vision.get_camera_position(rvec, tvec)
                webcam = Webcam(position, rvec)
                self.world.add_webcam(i + 1, webcam)

        for marker_id, x, y, z in self.world_positions:
            if marker_id == vision.MarkerId.TIN_CAN:
                self.world.add_tin_can(TinCan(Position(x, y)))
            elif vision.MarkerId.ROBOT_BLUE_LO <= marker_id <= vision.MarkerId.ROBOT_BLUE_HI:
                robot = Robot(Position(x, y, theta=0), RobotColor.BLUE)
                self.world.add_robot(robot)
            elif vision.MarkerId.ROBOT_YELLOW_LO <= marker_id <= vision.MarkerId.ROBOT_YELLOW_HI:
                robot = Robot(Position(x, y, theta=0), RobotColor.YELLOW)
                self.world.add_robot(robot)

        return self.world.render()


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
    reconstruction = Reconstruction([stream_1, stream_2])

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
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
