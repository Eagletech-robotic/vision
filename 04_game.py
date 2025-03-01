# ----------------
# Detect ArUco markers and draw the world with the detected objects.
# ----------------

import cv2 as cv
import math
from lib.visualisation import World, Robot, Position, TinCan, Webcam, RobotColor
from lib import common, camera, detection, vision


def add_objects(H, corners, ids, world):
    world.empty()

    for id, corner in zip(ids, corners):
        marker_id = id[0]
        corner = corner[0]

        # Take center point of marker
        center = corner.mean(axis=0)
        center_homogeneous = [center[0], center[1], 1]

        # Apply homography
        world_point = H @ center_homogeneous
        world_point = world_point / world_point[2]

        # Add known markers to the world
        if marker_id == vision.MarkerId.TIN_CAN:
            world.add_tin_can(TinCan(Position(world_point[0], world_point[1])))

        elif marker_id == vision.MarkerId.ROBOT_BLUE:
            robot = Robot(Position(world_point[0], world_point[1], theta=math.radians(0)), RobotColor.BLUE)
            world.add_robot(robot)

        elif marker_id == vision.MarkerId.ROBOT_YELLOW:
            robot = Robot(Position(world_point[0], world_point[1], theta=math.radians(180)), RobotColor.YELLOW)
            world.add_robot(robot)


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)

    # Initialize ArUco detector
    aruco_detector = detection.build_aruco_detector()

    # Initialize world
    world = World(blocking=False)
    # webcam = Webcam(Position(x=150, y=0, z=140, theta=math.pi / 2))
    # world.add_webcam(1, webcam)

    # Compute homography
    print("Compute the homography matrix")
    ret, image = cap.read()
    corners, ids, rejected = aruco_detector.detectMarkers(image)
    common.show_in_window("image", image)

    known_markers_positions = {
        vision.MarkerId.BOARD_BOTTOM_LEFT: vision.convert_marker_position_to_points(
            vision.MarkerPosition(50, 50, 0, 8, vision.MarkerRotation.TOP_LEFT)),
        vision.MarkerId.BOARD_TOP_LEFT: vision.convert_marker_position_to_points(
            vision.MarkerPosition(50, 150, 0, 8, vision.MarkerRotation.TOP_LEFT)),
        vision.MarkerId.BOARD_BOTTOM_RIGHT: vision.convert_marker_position_to_points(
            vision.MarkerPosition(250, 50, 0, 8, vision.MarkerRotation.TOP_LEFT)),
        vision.MarkerId.BOARD_TOP_RIGHT: vision.convert_marker_position_to_points(
            vision.MarkerPosition(250, 150, 0, 8, vision.MarkerRotation.TOP_LEFT)),
    }
    H = vision.compute_homography(corners, ids, known_markers_positions)

    # Main loop
    print("\nTo quit, select the live camera window and press 'q'.\n")
    running = True

    while running:
        # Take a picture and detect ArUco markers
        ret, image = cap.read()
        corners, ids, rejected = aruco_detector.detectMarkers(image)
        common.show_in_window("image", image)

        # Add objects to the world
        add_objects(H, corners, ids, world)

        # Draw the world and check if the user wants to quit
        world.render()

        c = cv.waitKey(50)
        if c == ord("q"):
            running = False

    world.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
