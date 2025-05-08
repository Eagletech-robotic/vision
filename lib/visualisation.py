import pyvista as pv
import numpy as np
from dataclasses import dataclass
import math
from enum import IntEnum
import cv2 as cv

from lib import common


@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0
    theta: float = 0.0  # rotation around Z axis in radians


class RobotColor(IntEnum):
    BLUE = 0
    YELLOW = 1


class Robot:
    def __init__(self, position: Position, color: RobotColor):
        self.position = position
        self.size = .3  # 30cm cube
        self.color = color

    def draw(self, plotter: pv.Plotter):
        # Cube at robot position
        center = (self.position.x, self.position.y, self.size / 2)
        cube = pv.Cube(
            center=center,
            x_length=self.size,
            y_length=self.size,
            z_length=self.size
        )
        angle_deg = math.degrees(self.position.theta)
        cube.rotate_z(angle_deg, point=center, inplace=True)
        plotter.add_mesh(cube, color=(255, 0, 0) if self.color == RobotColor.BLUE else (0, 255, 255))

        # Direction line
        length = self.size / 2
        start = (self.position.x, self.position.y, self.size + 1)  # 1cm above cube
        end = (
            self.position.x + math.cos(self.position.theta) * length,
            self.position.y + math.sin(self.position.theta) * length,
            self.size + 1
        )
        arrow = pv.Line(start, end)
        plotter.add_mesh(arrow, color='red')


class TinCan:
    def __init__(self, position: Position):
        self.position = position
        self.radius = .033  # tin can radius
        self.height = .115  # tin can height

    def draw(self, plotter: pv.Plotter):
        can = pv.Cylinder(center=(self.position.x, self.position.y, self.position.z + self.height / 2),
                          radius=self.radius,
                          height=self.height)
        plotter.add_mesh(can, color='silver')


class Webcam:
    def __init__(self, position, rvec):
        self.position = position
        self.rvec = rvec

    def draw(self, plotter: pv.Plotter):
        """Draws the camera as a pyramid (frustum) to represent its direction."""
        # Draw camera as a small sphere
        camera_sphere = pv.Sphere(center=self.position, radius=3)
        plotter.add_mesh(camera_sphere, color='blue')

        # Compute frustum corners (in camera space)
        fov = 30  # Field of view in degrees
        depth = 20  # How far the frustum extends
        angle = np.radians(fov / 2)
        h = np.tan(angle) * depth
        w = h * 1.5  # Assume 1.5x aspect ratio

        frustum_points = np.array([
            [0, 0, 0],  # Camera position
            [-w, -h, -depth],  # Bottom-left
            [w, -h, -depth],  # Bottom-right
            [w, h, -depth],  # Top-right
            [-w, h, -depth]  # Top-left
        ])

        # Transform frustum points to world space
        rotation, _ = cv.Rodrigues(self.rvec)
        frustum_points = (rotation @ frustum_points.T).T + self.position

        # Draw frustum lines
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # From camera to frustum corners
            (1, 2), (2, 3), (3, 4), (4, 1)  # Connect frustum corners
        ]
        for edge in edges:
            line = pv.Line(frustum_points[edge[0]], frustum_points[edge[1]])
            plotter.add_mesh(line, color="red", line_width=2)


class World:
    def __init__(self, blocking=True, off_screen=False):
        self.width = 3  # meters
        self.length = 2  # meters
        self.robots = []
        self.tin_cans = []
        self.webcams = {}
        self.blocking = blocking
        self.off_screen = off_screen

        # Initialize plotter
        self.plotter = pv.Plotter(off_screen=self.off_screen)
        self.plotter.window_size = (1920, 800)
        self.plotter.camera.position = (1.5, -2, 2)
        self.plotter.camera.focal_point = (1.5, 1, 0)  # Look at center of board
        self.plotter.camera.up = (0, 0, 1)

        # Load the ground plane's texture
        self.plane = pv.Plane(center=(self.width / 2, self.length / 2, 0),
                              direction=(0, 0, 1),
                              i_size=self.width,
                              j_size=self.length)
        try:
            tex = pv.read_texture('assets/playmat.png')
            self.plane.texture_map_to_plane(inplace=True)
            self.plane.active_texture = tex
        except Exception as e:
            print(f"Error loading texture: {e}")

        if not self.blocking and not self.off_screen:
            self.plotter.show(interactive_update=True)

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def add_tin_can(self, tin_can: TinCan):
        self.tin_cans.append(tin_can)

    def empty(self):
        self.robots = []
        self.tin_cans = []

    def add_webcam(self, index, webcam: Webcam):
        self.webcams[index] = webcam

    def draw_ground(self):
        self.plotter.add_mesh(self.plane, texture=self.plane.active_texture)

        # Draw grid
        grid_lines = pv.PolyData()
        for x in np.arange(0, self.width + 0.1, 50):  # Add 0.1 to include end
            line = pv.Line((x, 0, 1), (x, self.length, 1))  # 1cm above ground
            grid_lines += line
        for y in np.arange(0, self.length + 0.1, 50):
            line = pv.Line((0, y, 1), (self.width, y, 1))
            grid_lines += line
        self.plotter.add_mesh(grid_lines, color='white', line_width=2, opacity=0.8)

    @common.measure_time
    def render(self):
        self.plotter.clear()
        self.draw_ground()

        for robot in self.robots:
            robot.draw(self.plotter)

        for can in self.tin_cans:
            can.draw(self.plotter)

        # for webcam in self.webcams.values():
        #     webcam.draw(self.plotter)

        if self.off_screen:
            return self.plotter.screenshot(return_img=True)
        elif self.blocking:
            self.plotter.show()
            return None
        else:
            self.plotter.update()
            return None

    def close(self):
        self.plotter.close()
