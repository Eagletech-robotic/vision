# ----------------
# Draw a static world with robots, tin cans and a webcam
# ----------------

import math
from lib.visualisation import World, Robot, Position, TinCan, Webcam, RobotColor

def main():
    # Create world
    world = World(blocking=True)

    # Add robots
    robot1 = Robot(Position(x=50, y=50, theta=math.radians(30)), color=RobotColor.BLUE)
    robot2 = Robot(Position(x=250, y=150, theta=math.radians(70)), color=RobotColor.YELLOW)
    world.add_robot(robot1)
    world.add_robot(robot2)

    # Add some tin cans
    can_positions = [
        Position(100, 100),
        Position(150, 100),
        Position(200, 100)
    ]
    for pos in can_positions:
        world.add_tin_can(TinCan(pos))

    # Add webcam
    webcam = Webcam(Position(x=150, y=0, z=140, theta=math.pi / 2))
    world.set_webcam(webcam)

    # Show the world
    world.render()


if __name__ == "__main__":
    main()
