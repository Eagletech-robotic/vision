import cv2 as cv

from lib import board, common


def main():
    common.init_fullscreen_window("EagleTech Score")

    team_color, score = "blue", 86

    try:
        while True:
            score += 2

            img = board.draw_interface(team_color, score, width=1920, height=1080)
            cv.imshow("EagleTech Score", img)

            c = cv.waitKey(200)
            if c == ord("q"):
                break

    except KeyboardInterrupt:
        print("User interrupted the program.")
    finally:
        print("Cleaning up...")


if __name__ == "__main__":
    main()
