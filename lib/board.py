import cv2 as cv
import numpy as np
import os


def load_logo(width, height):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(script_dir, "../assets/logo.png")
    logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)
    logo = cv.resize(logo, (width, height))
    return logo


def draw_interface(team_color, score, width=1920, height=1080):
    img = np.ones((height, width, 3), np.uint8) * 220

    def put_text_centered(text, x, y, font=cv.FONT_HERSHEY_DUPLEX, font_scale=1.0, color=(255, 255, 255), thickness=2):
        text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int(x - text_size[0] / 2)
        text_y = int(y + text_size[1] / 2)
        cv.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

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
    score_box_x = int(width / 2 - score_box_width / 2)
    score_box_y = 400

    # Inner box
    cv.rectangle(img,
                 (score_box_x, score_box_y),
                 (score_box_x + score_box_width, score_box_y + score_box_height),
                 (51, 51, 51), -1)

    if team_color is None:
        put_text_centered("attente", int(width / 2), score_box_y + int(score_box_height / 2 - 45),
                          font_scale=2.0, color=(255, 255, 255), thickness=3)
        put_text_centered("detection", int(width / 2), score_box_y + int(score_box_height / 2 + 45),
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
        put_text_centered(str(score), int(width / 2), score_box_y + int(score_box_height / 2),
                          font_scale=7.0, color=(255, 255, 255), thickness=15)

    # ------------
    # Footer
    # ------------
    cv.rectangle(img, (0, height - 70), (width, height), (26, 26, 26), -1)
    put_text_centered("Analyse Video en Temps Reel - Systeme EagleTech Vision", int(width / 2), height - 35,
                      font_scale=0.8, color=(255, 255, 255), thickness=2)

    return img
