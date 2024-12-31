import cv2 as cv


def list_available_cameras():
    saved_log_level = cv.getLogLevel()
    cv.setLogLevel(0)  # Suppress errors when camera is not available

    try:
        def test_camera(index):
            cap = cv.VideoCapture(index)
            if cap.isOpened():
                backend = cap.getBackendName()
                cap.release()
                return {"index": index, "backend": backend}
            return None

        return [cam for cam in map(test_camera, range(10)) if cam]

    finally:
        cv.setLogLevel(saved_log_level)


def pick_camera():
    available_cameras = list_available_cameras()
    print("Available cameras: ", available_cameras)
    if len(available_cameras) > 1:
        return int(input("Enter the camera index then press enter:"))
    elif len(available_cameras) == 1:
        return available_cameras[0]["index"]
    else:
        print("No cameras found")
        exit()


CAMERA_PROPERTIES = {
    "Zoom": cv.CAP_PROP_ZOOM,
    "Width": cv.CAP_PROP_FRAME_WIDTH,
    "Height": cv.CAP_PROP_FRAME_HEIGHT,
    "FPS": cv.CAP_PROP_FPS,
    "Format": cv.CAP_PROP_FORMAT,
    "Mode": cv.CAP_PROP_MODE,
    "Brightness": cv.CAP_PROP_BRIGHTNESS,
    "Contrast": cv.CAP_PROP_CONTRAST,
    "Saturation": cv.CAP_PROP_SATURATION,
    "Hue": cv.CAP_PROP_HUE,
    "Gain": cv.CAP_PROP_GAIN,
    "Exposure": cv.CAP_PROP_EXPOSURE,
    "Auto Exposure": cv.CAP_PROP_AUTO_EXPOSURE,
    "Auto Focus": cv.CAP_PROP_AUTOFOCUS,
    "Focus": cv.CAP_PROP_FOCUS,
}


def print_camera_properties(cap):
    print("Camera properties:")
    for name, prop in CAMERA_PROPERTIES.items():
        print(f" - {name}: {cap.get(prop)}")


def monitor_property_changes(cap):
    def decorator(func):
        def get_properties():
            return {name: cap.get(prop) for name, prop in CAMERA_PROPERTIES.items()}

        prev_properties = get_properties()
        func()
        current_properties = get_properties()

        changes = [
            f"- {name}: {prev_properties[name]} -> {current_properties[name]}"
            for name in current_properties
            if current_properties[name] != prev_properties[name]
        ]
        if len(changes) > 0:
            print("Property changes:")
            for change in changes:
                print(change)

    return decorator
