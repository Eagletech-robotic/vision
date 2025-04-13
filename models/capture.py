from lib import detection, vision
import cv2 as cv

from lib.common import put_text_with_background


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
            vision.estimate_pose(corners, ids, vision.MarkerPositions, self.camera_matrix, self.dist_coeffs)
        if ret:
            pos = vision.get_camera_position(rvec, tvec)
            euler = vision.rodrigues_to_euler(rvec)
            self.last_pose = rvec, tvec, pos, euler

        return self.last_pose

    def debug_image(self):
        img_width, img_height = 1920, 1080
        img = cv.resize(self.image, (img_width, img_height))

        # Show pose
        pose = self.estimate_pose()
        x, y, z = pose[2] if pose else (0, 0, 0)
        put_text_with_background(img, f"X:{x:.0f} Y:{y:.0f} Z:{z:.0f}", (10, 30),
                                 cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the detected markers
        corners, ids = self._detection()
        if ids is not None:
            for i, corner in enumerate(corners):
                cv.polylines(img, [corner[0].astype(int)], True, (0, 255, 0), 2)
                put_text_with_background(
                    img, str(ids[i][0]), tuple(corner[0][0].astype(int)), font_scale=1,
                    text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, padding=5
                )

        return img
