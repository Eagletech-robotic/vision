import pyvista as pv
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class Position:
    x: float
    y: float
    z: float = 0.0
    theta: float = 0.0  # rotation around Z axis in radians


class Robot:
    def __init__(self, position: Position):
        self.position = position
        self.size = 30  # 30cm cube

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
        plotter.add_mesh(cube, color='orange')

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
        self.radius = 3.3  # tin can radius
        self.height = 11.5  # tin can height

    def draw(self, plotter: pv.Plotter):
        can = pv.Cylinder(center=(self.position.x, self.position.y, self.position.z + self.height / 2),
                          direction=(0, 0, 1),
                          radius=self.radius,
                          height=self.height)
        plotter.add_mesh(can, color='silver')


class Webcam:
    def __init__(self, position: Position):
        self.position = position

    def draw(self, plotter: pv.Plotter):
        webcam = pv.Sphere(center=(self.position.x, self.position.y, self.position.z),
                           radius=10)
        plotter.add_mesh(webcam, color='blue')


class World:
    def __init__(self):
        self.width = 300  # 300cm
        self.length = 200  # 200cm
        self.robots = []
        self.tin_cans = []
        self.webcam = None

        # Initialize plotter
        self.plotter = pv.Plotter()
        self.plotter.camera.position = (150, -400, 400)
        self.plotter.camera.focal_point = (150, 100, 0)  # Look at center of board
        self.plotter.camera.up = (0, 0, 1)  # Z is up

    def add_robot(self, robot: Robot):
        self.robots.append(robot)

    def add_tin_can(self, tin_can: TinCan):
        self.tin_cans.append(tin_can)

    def empty(self):
        self.robots = []
        self.tin_cans = []

    def set_webcam(self, webcam: Webcam):
        self.webcam = webcam

    def draw_ground(self):
        # Create the ground plane
        plane = pv.Plane(center=(self.width / 2, self.length / 2, 0),
                         direction=(0, 0, 1),
                         i_size=self.width,
                         j_size=self.length)

        # Load and apply texture
        try:
            tex = pv.read_texture('assets/playmat.png')
            plane.texture_map_to_plane(inplace=True)
            plane.active_texture = tex
            self.plotter.add_mesh(plane, texture=plane.active_texture)
        except Exception as e:
            print(f"Error loading texture: {e}")

        # Draw grid
        grid_lines = pv.PolyData()
        for x in np.arange(0, self.width + 0.1, 50):  # Add 0.1 to include end
            line = pv.Line((x, 0, 1), (x, self.length, 1))  # 1cm above ground
            grid_lines += line
        for y in np.arange(0, self.length + 0.1, 50):
            line = pv.Line((0, y, 1), (self.width, y, 1))
            grid_lines += line
        self.plotter.add_mesh(grid_lines, color='white', line_width=2, opacity=0.8)

    def draw(self):
        self.plotter.clear()

        self.draw_ground()

        for robot in self.robots:
            robot.draw(self.plotter)

        for can in self.tin_cans:
            can.draw(self.plotter)

        if self.webcam:
            self.webcam.draw(self.plotter)

        self.plotter.show()
