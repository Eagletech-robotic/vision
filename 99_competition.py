import pygame
import cv2 as cv

from lib import board, common, camera
from models.analyser import generate_world
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

    try:
        persistent_state = PersistentState()

        running = True
        debug_mode = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:  # Esc or Q to quit
                        running = False
                    if event.key == pygame.K_d:
                        debug_mode = not debug_mode

            capture_1 = stream_1.capture()
            capture_2 = stream_2.capture()
            world, persistent_state = generate_world(capture_1, capture_2, persistent_state)

            if debug_mode:
                board_img = board.draw_interface_debug(capture_1, capture_2, world)
            else:
                board_img = board.draw_interface(world.team_color, world.score)
            show_cv_image(screen, board_img)

            clock.tick(5)  # 5 FPS ≈ 200ms per frame

    except KeyboardInterrupt:
        print("User interrupted the program.")
    finally:
        pygame.quit()
        print("Cleaning up...")


if __name__ == "__main__":
    main()
