import cv2 as cv
from lib import common, camera, detection


def set_properties(cap):
    @camera.monitor_property_changes(cap)
    def func():
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 3264)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2448)
        cap.set(cv.CAP_PROP_FPS, 40)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_FOCUS, 255)
        camera.print_camera_properties(cap)


def add_trackbars(window_name, property_names, cap):
    def on_trackbar(prop):
        def value_updated(value):
            camera.monitor_property_changes(cap)(lambda: cap.set(prop, value))

        return value_updated

    common.init_window(window_name)
    for name in property_names:
        prop = camera.CAMERA_PROPERTIES[name]
        if not prop:
            print(f"No property found with name {name}. Exiting.")
            exit()
        cv.createTrackbar(name, "image", 100, 255, on_trackbar(prop))


def process_image(image):
    # gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detection.aruco_detector.detectMarkers(image)
    print("Number of tags detected: ", len(corners) if corners is not None else 0)
    cv.aruco.drawDetectedMarkers(image, corners, ids)
    common.show_in_window("image", image)


camera_index = camera.pick_camera()
cap = cv.VideoCapture(camera_index)

set_properties(cap)
add_trackbars("image", ["Brightness", "Contrast", "Saturation", "Hue", "Gain", "Focus"], cap)

print("\nPress 'q' to quit...\n")
camera_fps = int(cap.get(cv.CAP_PROP_FPS))
while True:
    ret, frame = cap.read()
    process_image(frame)
    c = cv.waitKey(1000 // camera_fps)
    if c == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
