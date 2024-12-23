import cv2
from collections import defaultdict

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)


def process_frame(frame, frame_count):
    if frame_count % 30 == 0:  # Check every 30 frames
        current_values, changes = monitor_dynamic_properties(cap)

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(frame)

    print("Processing frame, number of tags found: ", len(corners) if corners is not None else 0)

    if ids is not None:
        # Draw markers
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw IDs
        for i, corner in enumerate(corners):
            center = corner[0].mean(axis=0).astype(int)
            cv2.putText(frame, str(ids[i][0]),
                        tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    sized_down_frame = cv2.resize(frame, (1632, 1224))
    cv2.imshow('frame', sized_down_frame)


def print_camera_properties(cap):
    # Get backend API being used
    backend = cap.getBackendName()
    print(f"Backend API: {backend}")

    # Try to get camera properties
    print("\nCamera Properties:")
    # name of the camera
    print(f"Zoom: {cap.get(cv2.CAP_PROP_ZOOM)}")
    print(f"Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Format: {cap.get(cv2.CAP_PROP_FORMAT)}")
    print(f"Mode: {cap.get(cv2.CAP_PROP_MODE)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Hue: {cap.get(cv2.CAP_PROP_HUE)}")
    print(f"Gain: {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print(f"Auto Focus: {cap.get(cv2.CAP_PROP_AUTOFOCUS)}")
    print(f"Focus: {cap.get(cv2.CAP_PROP_FOCUS)}")


def monitor_dynamic_properties(cap):
    # Dictionary to track property changes
    props = {
        'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
        'CONTRAST': cv2.CAP_PROP_CONTRAST,
        'SATURATION': cv2.CAP_PROP_SATURATION,
        'HUE': cv2.CAP_PROP_HUE,
        'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
        'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE,
    }

    # Track previous values
    prev_values = {name: cap.get(prop) for name, prop in props.items()}
    changes = defaultdict(int)

    # Get current values and compare
    current_values = {name: cap.get(prop) for name, prop in props.items()}

    for name in props:
        if current_values[name] != prev_values[name]:
            changes[name] += 1
            print(f"{name}: {prev_values[name]} -> {current_values[name]}")

    return current_values, changes


#  List all available cameras
available_cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()
print("Available cameras: ", available_cameras)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_ZOOM, 50)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
cap.set(cv2.CAP_PROP_FPS, 40)
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 255)

print_camera_properties(cap)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't read frame. Exiting ...")
        break

    process_frame(frame, frame_count)
    frame_count += 1

    cv2.waitKey(35)

cap.release()
cv2.destroyAllWindows()
