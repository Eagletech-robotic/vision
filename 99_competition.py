import pygame
import cv2 as cv

from lib import board


def cv_image_to_pygame(cv_image):
    cv_image_rgb = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
    pygame_surface = pygame.surfarray.make_surface(cv_image_rgb.swapaxes(0, 1))
    return pygame_surface


def main():
    pygame.init()

    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    pygame.display.set_caption("EagleTech Score")

    team_color, score = "blue", 86
    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:  # Esc or Q to quit
                        running = False

            score += 2

            cv_img = board.draw_interface(team_color, score, width=1920, height=1080)
            pygame_img = cv_image_to_pygame(cv_img)
            screen.blit(pygame_img, (0, 0))
            pygame.display.flip()

            clock.tick(5)  # 5 FPS ≈ 200ms per frame

    except KeyboardInterrupt:
        print("User interrupted the program.")
    finally:
        pygame.quit()
        print("Cleaning up...")


if __name__ == "__main__":
    main()
