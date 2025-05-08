# ----------------
# Detect ArUco markers and draw the world with the detected objects.
# ----------------

import cv2 as cv
import math
from lib.visualisation import World, Robot, Position, TinCan, Webcam, RobotColor
from lib import common, camera, detection, vision


def reset_objects(H, corners, ids, world):
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
            print(f"Adding tin can at {world_point}")
            world.add_tin_can(TinCan(Position(world_point[0], world_point[1])))

        elif vision.MarkerId.ROBOT_BLUE_LO <= marker_id <= vision.MarkerId.ROBOT_BLUE_HI:
            print(f"Adding blue robot at {world_point}")
            robot = Robot(Position(world_point[0], world_point[1], theta=math.radians(0)), RobotColor.BLUE)
            world.add_robot(robot)

        elif vision.MarkerId.ROBOT_YELLOW_LO <= marker_id <= vision.MarkerId.ROBOT_YELLOW_HI:
            print(f"Adding yellow robot at {world_point}")
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

    # Main loop
    print("\nTo quit, select the live camera window and press 'q'.\n")
    running = True

    while running:
        # Take a picture and detect ArUco markers
        ret, image = cap.read()
        corners, ids, rejected = aruco_detector.detectMarkers(image)
        detection.draw_aruco_markers(image, corners, ids)
        common.show_in_window("image", image)

        # Check if any markers were detected
        if ids is not None and len(ids) > 0:
            # Try to compute the homography
            ret, H = vision.compute_homography(corners, ids, vision.FIELD_MARKERS)
            if ret is not None:
                # Add objects to the world
                reset_objects(H, corners, ids, world)

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
