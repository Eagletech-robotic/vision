from lib import detection, vision, common
import cv2 as cv


class Capture:
    def __init__(self, stream, image):
        self.camera_index = stream.camera_index
        self.camera_matrix = stream.camera_matrix
        self.dist_coeffs = stream.dist_coeffs

        self.image = image

        self.last_pose = None
        self.last_detection = None

    def _detection(self):
        if self.last_detection is not None:
            return self.last_detection

        self.aruco_detector = detection.build_aruco_detector()
        corners, ids, _rejected = self.aruco_detector.detectMarkers(self.image)
        self.last_detection = corners, ids
        return self.last_detection

    def estimate_pose(self):
        if self.last_pose is not None:
            return self.last_pose

        corners, ids = self._detection()
        if ids is None or len(ids) == 0:
            return

        ret, rvec, tvec = \
            vision.estimate_pose(corners, ids, vision.FIELD_MARKERS, self.camera_matrix, self.dist_coeffs)
        if ret:
            pos = vision.get_camera_position(rvec, tvec)
            euler = vision.rodrigues_to_euler(rvec)
            self.last_pose = rvec, tvec, pos, euler

        return self.last_pose

    def debug_image(self):
        """Generate an image of the capture augmented with detected markers and pose."""
        IMG_WIDTH, IMG_HEIGHT = 1920, 1080

        # Create a black image
        img = cv.resize(self.image, (IMG_WIDTH, IMG_HEIGHT))

        # Show pose
        pose = self.estimate_pose()
        x, y, z = pose[2] if pose else (0, 0, 0)
        common.draw_text_with_background(img, f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f}", (10, 30))
        # Show euler angles
        if pose:
            roll, pitch, yaw = pose[3]
            common.draw_text_with_background(img, f"Roll:{roll:.0f} Pitch:{pitch:.0f} Yaw:{yaw:.0f}", (10, 70))

        # Show the detected markers
        corners, ids = self._detection()
        detection.draw_aruco_markers(img, corners, ids)

        return img
