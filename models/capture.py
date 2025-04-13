from lib import detection, vision


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
