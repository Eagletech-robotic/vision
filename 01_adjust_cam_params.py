# ----------------
# Adjust camera parameters using trackbars, and save them to a yaml file.
# The camera view is displayed with detected ArUco markers to help optimize the parameters.
# ----------------

import cv2 as cv
from lib import common, camera, detection


def add_trackbars(window_name, properties, cap):
    def on_trackbar(prop, values):
        def value_updated(index):
            value = values[index]
            camera.monitor_property_changes(cap)(lambda: cap.set(prop, value))

        return value_updated

    common.init_window(window_name)
    for prop in properties:
        values = camera.detect_acceptable_values(cap, prop, -100, 500)
        name = camera.ALL_CAMERA_PROPERTIES[prop]
        if not name:
            print(f"Unknown property {prop}. Exiting.")
            exit()
        print(f"Acceptable values for {name}: {values}")
        if len(values) > 1:
            value = int(cap.get(prop))
            index = values.index(value)
            cv.createTrackbar(name, "image", index, len(values) - 1, on_trackbar(prop, values))


@common.measure_time
def process_image(image, aruco_detector):
    corners, ids, rejected = aruco_detector.detectMarkers(image)
    print("Number of tags detected: ", len(corners) if corners is not None else 0)
    cv.aruco.drawDetectedMarkers(image, corners, ids)
    common.show_in_window("image", image)


FRAMES_PER_SEC = 5


def main():
    common.run_hw_diagnostics()
    aruco_detector = detection.build_aruco_detector()

    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)

    camera.print_properties(cap, all=True)
    camera.load_properties(cap, camera_index)
    add_trackbars("image", [
        cv.CAP_PROP_AUTO_EXPOSURE,
        cv.CAP_PROP_AUTOFOCUS,
        cv.CAP_PROP_FPS,
        cv.CAP_PROP_BRIGHTNESS,
        cv.CAP_PROP_CONTRAST,
        cv.CAP_PROP_SATURATION,
        cv.CAP_PROP_HUE,
        cv.CAP_PROP_GAIN,
        cv.CAP_PROP_EXPOSURE,
        cv.CAP_PROP_FOCUS,
    ], cap)

    print("\nPress 'q' to quit, 's' to save parameters.\n")
    while True:
        ret, frame = cap.read()
        process_image(frame, aruco_detector)
        c = cv.waitKey(1000 // FRAMES_PER_SEC)
        if c == ord("q"):
            break
        elif c == ord("s"):
            camera.save_properties(cap, camera_index)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
