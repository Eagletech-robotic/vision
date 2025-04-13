from models.world import World


def _find_team_color(capture):
    pose = capture.estimate_pose()
    if pose:
        rvec, tvec, pos, euler = pose
        print(f"{capture.camera_index} X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
        return "blue" if pos[0] > 150 else "yellow"
    return None


def generate_world(capture_1, capture_2, persistent_state):
    world = World()
    world.score = persistent_state.score or 0

    world.team_color = _find_team_color(capture_1) or _find_team_color(capture_2) or persistent_state.team_color

    return world, persistent_state
