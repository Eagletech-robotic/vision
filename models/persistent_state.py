class PersistentState:
    def __init__(self):
        self.team_color = None
        self.score = 40

        # Last reliable poses for each camera
        self.camera_poses = {}  # camera_index -> (rvec, tvec, pos, euler)
