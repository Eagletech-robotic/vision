import cv2 as cv
from lib import common, camera, detection


def add_trackbars(window_name, properties, cap):
    def on_trackbar(prop):
        def value_updated(value):
            camera.monitor_property_changes(cap)(lambda: cap.set(prop, value))

        return value_updated

    common.init_window(window_name)
    for prop in properties:
        name = camera.CAMERA_PROPERTIES[prop]
        if not name:
            print(f"Unknown property {prop}. Exiting.")
            exit()
        cv.createTrackbar(name, "image", int(cap.get(prop)), 255, on_trackbar(prop))


def process_image(image):
    # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detection.aruco_detector.detectMarkers(image)
    print("Number of tags detected: ", len(corners) if corners is not None else 0)
    cv.aruco.drawDetectedMarkers(image, corners, ids)
    common.show_in_window("image", image)


camera_index = camera.pick_camera()
cap = cv.VideoCapture(camera_index)

camera.load_properties(cap, camera_index)
add_trackbars("image", [
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
camera_fps = int(cap.get(cv.CAP_PROP_FPS))
while True:
    ret, frame = cap.read()
    process_image(frame)
    c = cv.waitKey(1000 // camera_fps)
    if c == ord("q"):
        break
    elif c == ord("s"):
        camera.save_properties(cap, camera_index)

cap.release()
cv.destroyAllWindows()
