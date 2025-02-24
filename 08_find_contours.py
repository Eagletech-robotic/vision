import cv2 as cv
import numpy as np

from lib import common, camera, detection, vision


def main():
    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)
    camera_matrix, dist_coeffs = camera.load_calibration(camera_index)

    while True:
        _, image = cap.read()

        edges = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.GaussianBlur(edges, (5, 5), 0)
        # edges = cv.Canny(edges, 50, 150)
        # kernel = np.ones((5, 5), np.uint8)
        # edges = cv.dilate(edges, kernel, iterations=1)
        # edges = cv.erode(edges, kernel, iterations=1)

        ret, thresh = cv.threshold(edges, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv.contourArea(contour)

            # Keep only long, rectangular shapes
            if aspect_ratio > 2.5 and area > 5000:
                filtered_contours.append(contour)

        image_with_contours = image.copy()
        cv.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

        print(f"Nb contours detected: {len(contours)}")
        common.show_in_window("image", image)
        common.show_in_window("edges", edges)
        common.show_in_window("image_with_contours", image_with_contours)

        c = cv.waitKey(100)
        if c == ord("q"):
            break


if __name__ == "__main__":
    main()
