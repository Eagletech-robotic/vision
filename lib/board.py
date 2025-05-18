import cv2 as cv
import numpy as np
import os

IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080


def load_logo(width, height):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "../assets/logo.png")
    logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)
    logo = cv.resize(logo, (width, height))
    return logo


def draw_interface(team_color, score):
    img = _draw_common_elements()

    # ------------
    # Header
    # ------------
    # Insert logo with transparency
    logo = load_logo(200, 200)
    roi = img[80:80 + logo.shape[0], 450:450 + logo.shape[1]]
    alpha_mask = logo[:, :, 3:4] / 255.0
    roi[:] = roi * (1 - alpha_mask) + logo[:, :, :3] * alpha_mask

    cv.putText(img, "EagleTech Robotics", (670, 180),
               cv.FONT_HERSHEY_DUPLEX, 2, (70, 46, 12), 5)

    # ------------
    # Score
    # ------------
    score_box_width, score_box_height = 500, 400
    score_box_x = int(IMAGE_WIDTH / 2 - score_box_width / 2)
    score_box_y = 400

    # Inner box
    cv.rectangle(img,
                 (score_box_x, score_box_y),
                 (score_box_x + score_box_width, score_box_y + score_box_height),
                 (51, 51, 51), -1)

    if team_color is None:
        _put_text_centered(img, "attente", int(IMAGE_WIDTH / 2), score_box_y + int(score_box_height / 2 - 45),
                           font_scale=2.0, color=(255, 255, 255), thickness=3)
        _put_text_centered(img, "detection", int(IMAGE_WIDTH / 2), score_box_y + int(score_box_height / 2 + 45),
                           font_scale=2.0, color=(255, 255, 255), thickness=3)

    else:
        # Border
        border_color = (5, 188, 251) if team_color == "yellow" else (244, 133, 66)  # BGR
        border_thickness = 30
        half_thickness = int(border_thickness / 2)
        cv.rectangle(img,
                     (score_box_x - half_thickness, score_box_y - half_thickness),
                     (score_box_x + score_box_width + half_thickness, score_box_y + score_box_height + half_thickness),
                     border_color, border_thickness)

        # Score counter
        _put_text_centered(img, str(score), int(IMAGE_WIDTH / 2), score_box_y + int(score_box_height / 2),
                           font_scale=7.0, color=(255, 255, 255), thickness=15)

    return img


def draw_interface_debug(capture_1, capture_2, world, log_lines):
    img = _draw_common_elements()

    mini_width = 800
    mini_height = (mini_width * IMAGE_HEIGHT) // IMAGE_WIDTH

    def insert_capture(image, x, y):
        image = cv.resize(image, (mini_width, mini_height))
        img[y:y + image.shape[0], x:x + image.shape[1]] = image

    insert_capture(capture_1.debug_image(), 100, 50)
    insert_capture(capture_2.debug_image(), IMAGE_WIDTH - mini_width - 100, 50)
    insert_capture(world.debug_image(log_lines), (IMAGE_WIDTH - mini_width) // 2, 80 + mini_height)

    return img


def _draw_common_elements():
    img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8) * 220

    # ------------
    # Footer
    # ------------
    cv.rectangle(img, (0, IMAGE_HEIGHT - 70), (IMAGE_WIDTH, IMAGE_HEIGHT), (26, 26, 26), -1)
    _put_text_centered(img, "Analyse Video en Temps Reel - Systeme EagleTech Vision", int(IMAGE_WIDTH / 2),
                       IMAGE_HEIGHT - 35,
                       font_scale=0.8, color=(255, 255, 255), thickness=2)

    return img


def _put_text_centered(img, text, x, y, font=cv.FONT_HERSHEY_DUPLEX, font_scale=1.0, color=(255, 255, 255),
                       thickness=2):
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(x - text_size[0] / 2)
    text_y = int(y + text_size[1] / 2)
    cv.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
