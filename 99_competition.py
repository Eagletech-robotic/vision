import pygame
import cv2 as cv
from datetime import datetime

from lib import board, eagle_packet, camera, common, ble_robot
from lib.eagle_packet import frame_to_human
from lib.image_logger import ImageLogger
from models.analyser import Analyser
from models.persistent_state import PersistentState
from models.stream import Stream

SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
CAMERA_NAME = "W4DS--SN0001"


def init_streams():
    while True:
        available_cameras = camera.list_available_cameras()
        indices = [cam["index"] for cam in available_cameras if cam["name"] == CAMERA_NAME]
        if len(indices) >= 2:
            cam_index_1, cam_index_2 = indices[0], indices[1]
            print("Available cameras: ", available_cameras, " - Using cameras: ", cam_index_1, cam_index_2)
            return Stream(cam_index_1), Stream(cam_index_2)
        else:
            print(f"Could not find 2 cameras with name '{CAMERA_NAME}'. Retrying...")
            pygame.time.delay(1000)


def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("EagleTech Score")
    clock = pygame.time.Clock()
    return screen, clock


def show_cv_image(screen, cv_image):
    cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    pygame_surface = pygame.surfarray.make_surface(cv_image_rgb.swapaxes(0, 1))
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()


def main():
    common.run_hw_diagnostics()
    stream_1, stream_2 = init_streams()

    screen, clock = init_pygame()
    image_logger = ImageLogger()

    ble_robot.start_ble_thread(ble_robot.MacAddress.ROBOT)

    try:
        persistent_state = PersistentState()
        running = True
        debug_mode = False

        while running:
            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:  # Esc or Q to quit
                        running = False
                    if event.key == pygame.K_d:
                        debug_mode = not debug_mode

            # Capture images
            capture_1 = stream_1.capture()
            capture_2 = stream_2.capture()

            # Analyse camera frames
            analyser = Analyser(capture_1, capture_2)
            world, persistent_state = analyser.generate_world(persistent_state)

            # Send Bluetooth frame as quickly as possible after capture
            frame = eagle_packet.frame_payload(
                world.to_eagle_packet()
            )
            # print(frame_to_human(frame))

            # Send the frame
            send_time = datetime.now()
            ble_robot.send_frame(frame)
            print("Send packet:", frame.hex())
            
            # Create log entries with timestamps
            log_entries = [
                (capture_1.time, common.format_time(capture_1.time, f"Capture 1")),
                (capture_2.time, common.format_time(capture_2.time, f"Capture 2")),
                (send_time, common.format_time(send_time, f"Send packet"))
            ]
            
            # Get robot logs and combine with our logs
            robot_logs = ble_robot.read_buffer()
            all_logs = log_entries + robot_logs
            
            # Draw the UI while the packet is being sent in the background
            debug_board_img = board.draw_interface_debug(capture_1, capture_2, world, all_logs)
            if debug_mode:
                board_img = debug_board_img
            else:
                board_img = board.draw_interface(world.team_color, world.score)
            show_cv_image(screen, board_img)

            # Save logs
            image_logger.append(debug_board_img)

            clock.tick(2)  # FPS

    except KeyboardInterrupt:
        print("User interrupted the program.")

    finally:
        pygame.quit()
        print("Cleaning up...")


if __name__ == "__main__":
    main()
