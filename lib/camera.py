import cv2 as cv
import subprocess
import yaml


def camera_name(camera_index):
    cmd = f"udevadm info --name=/dev/video{camera_index}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to execute command: {cmd}")

    lines = result.stdout.split("\n")
    read_value = lambda key: next(s for s in lines if f"E: {key}" in s).split("=")[-1]

    return read_value("ID_MODEL")


def list_available_cameras():
    saved_log_level = cv.getLogLevel()
    cv.setLogLevel(0)  # Suppress errors when camera is not available

    try:
        def test_camera(index):
            cap = cv.VideoCapture(index)
            if cap.isOpened():
                cap.release()
                return {"index": index, "name": camera_name(index)}
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


def capture(camera_index):
    cap = cv.VideoCapture(camera_index)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    return cap


CAMERA_PROPERTIES = {
    cv.CAP_PROP_ZOOM: "Zoom",
    cv.CAP_PROP_FRAME_WIDTH: "Width",
    cv.CAP_PROP_FRAME_HEIGHT: "Height",
    cv.CAP_PROP_FPS: "FPS",
    cv.CAP_PROP_FORMAT: "Format",
    cv.CAP_PROP_MODE: "Mode",
    cv.CAP_PROP_BRIGHTNESS: "Brightness",
    cv.CAP_PROP_CONTRAST: "Contrast",
    cv.CAP_PROP_SATURATION: "Saturation",
    cv.CAP_PROP_HUE: "Hue",
    cv.CAP_PROP_GAIN: "Gain",
    cv.CAP_PROP_EXPOSURE: "Exposure",
    cv.CAP_PROP_AUTO_EXPOSURE: "Auto Exposure",
    cv.CAP_PROP_AUTOFOCUS: "Auto Focus",
    cv.CAP_PROP_FOCUS: "Focus",
}


def monitor_property_changes(cap):
    def decorator(func):
        def get_properties():
            return {name: cap.get(prop) for prop, name in CAMERA_PROPERTIES.items()}

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


def print_camera_properties(cap):
    print("Camera properties:")
    for prop, name in CAMERA_PROPERTIES.items():
        print(f" - {name}: {cap.get(prop)}")


def camera_settings_path(camera_index):
    name = camera_name(camera_index)
    return f"camera-settings/{name}.yaml"


def load_properties(cap, camera_index):
    @monitor_property_changes(cap)
    def func():
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0)

        with open(camera_settings_path(camera_index), 'r') as file:
            settings = yaml.safe_load(file)
        for name, value in settings["properties"].items():
            prop = next(p for p, n in CAMERA_PROPERTIES.items() if n == name)
            cap.set(prop, value)


def save_properties(cap, camera_index):
    @monitor_property_changes(cap)
    def func():
        with open(camera_settings_path(camera_index), 'w') as file:
            settings = {
                "properties": {name: cap.get(prop) for prop, name in CAMERA_PROPERTIES.items()}
            }
            yaml.dump(settings, file)
