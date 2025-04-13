import numpy as np

from models.world import World


def _find_team_color(captures):
    x_values = [pose[2][0] for capture in captures if (pose := capture.estimate_pose()) is not None]
    if len(x_values) > 0:
        return "blue" if np.mean(x_values) > 150 else "yellow"
    else:
        return None


def generate_world(capture_1, capture_2, persistent_state):
    world = World()
    world.score = persistent_state.score or 0

    world.team_color = _find_team_color([capture_1, capture_2]) or persistent_state.team_color

    return world, persistent_state
