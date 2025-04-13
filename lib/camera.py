import cv2 as cv
import subprocess

import numpy as np
import yaml

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
    cv.CAP_PROP_AUTO_EXPOSURE: "Auto Exposure",
    cv.CAP_PROP_EXPOSURE: "Exposure",
    cv.CAP_PROP_AUTOFOCUS: "Auto Focus",
    cv.CAP_PROP_FOCUS: "Focus",
}

ALL_CAMERA_PROPERTIES = {
    cv.CAP_PROP_POS_MSEC: "CAP_PROP_POS_MSEC",
    cv.CAP_PROP_POS_FRAMES: "CAP_PROP_POS_FRAMES",
    cv.CAP_PROP_POS_AVI_RATIO: "CAP_PROP_POS_AVI_RATIO",
    cv.CAP_PROP_FRAME_WIDTH: "CAP_PROP_FRAME_WIDTH",
    cv.CAP_PROP_FRAME_HEIGHT: "CAP_PROP_FRAME_HEIGHT",
    cv.CAP_PROP_FPS: "CAP_PROP_FPS",
    cv.CAP_PROP_FOURCC: "CAP_PROP_FOURCC",
    cv.CAP_PROP_FRAME_COUNT: "CAP_PROP_FRAME_COUNT",
    cv.CAP_PROP_FORMAT: "CAP_PROP_FORMAT",
    cv.CAP_PROP_MODE: "CAP_PROP_MODE",
    cv.CAP_PROP_BRIGHTNESS: "CAP_PROP_BRIGHTNESS",
    cv.CAP_PROP_CONTRAST: "CAP_PROP_CONTRAST",
    cv.CAP_PROP_SATURATION: "CAP_PROP_SATURATION",
    cv.CAP_PROP_HUE: "CAP_PROP_HUE",
    cv.CAP_PROP_GAIN: "CAP_PROP_GAIN",
    cv.CAP_PROP_EXPOSURE: "CAP_PROP_EXPOSURE",
    cv.CAP_PROP_CONVERT_RGB: "CAP_PROP_CONVERT_RGB",
    cv.CAP_PROP_WHITE_BALANCE_BLUE_U: "CAP_PROP_WHITE_BALANCE_BLUE_U",
    cv.CAP_PROP_RECTIFICATION: "CAP_PROP_RECTIFICATION",
    cv.CAP_PROP_MONOCHROME: "CAP_PROP_MONOCHROME",
    cv.CAP_PROP_SHARPNESS: "CAP_PROP_SHARPNESS",
    cv.CAP_PROP_AUTO_EXPOSURE: "CAP_PROP_AUTO_EXPOSURE",
    cv.CAP_PROP_GAMMA: "CAP_PROP_GAMMA",
    cv.CAP_PROP_TEMPERATURE: "CAP_PROP_TEMPERATURE",
    cv.CAP_PROP_TRIGGER: "CAP_PROP_TRIGGER",
    cv.CAP_PROP_TRIGGER_DELAY: "CAP_PROP_TRIGGER_DELAY",
    cv.CAP_PROP_WHITE_BALANCE_RED_V: "CAP_PROP_WHITE_BALANCE_RED_V",
    cv.CAP_PROP_ZOOM: "CAP_PROP_ZOOM",
    cv.CAP_PROP_FOCUS: "CAP_PROP_FOCUS",
    cv.CAP_PROP_GUID: "CAP_PROP_GUID",
    cv.CAP_PROP_ISO_SPEED: "CAP_PROP_ISO_SPEED",
    cv.CAP_PROP_BACKLIGHT: "CAP_PROP_BACKLIGHT",
    cv.CAP_PROP_PAN: "CAP_PROP_PAN",
    cv.CAP_PROP_TILT: "CAP_PROP_TILT",
    cv.CAP_PROP_ROLL: "CAP_PROP_ROLL",
    cv.CAP_PROP_IRIS: "CAP_PROP_IRIS",
    cv.CAP_PROP_SETTINGS: "CAP_PROP_SETTINGS",
    cv.CAP_PROP_BUFFERSIZE: "CAP_PROP_BUFFERSIZE",
    cv.CAP_PROP_AUTOFOCUS: "CAP_PROP_AUTOFOCUS",
    cv.CAP_PROP_SAR_NUM: "CAP_PROP_SAR_NUM",
    cv.CAP_PROP_SAR_DEN: "CAP_PROP_SAR_DEN",
    cv.CAP_PROP_BACKEND: "CAP_PROP_BACKEND",
    cv.CAP_PROP_CHANNEL: "CAP_PROP_CHANNEL",
    cv.CAP_PROP_AUTO_WB: "CAP_PROP_AUTO_WB",
    cv.CAP_PROP_WB_TEMPERATURE: "CAP_PROP_WB_TEMPERATURE",
    cv.CAP_PROP_CODEC_PIXEL_FORMAT: "CAP_PROP_CODEC_PIXEL_FORMAT",
    cv.CAP_PROP_BITRATE: "CAP_PROP_BITRATE",
    cv.CAP_PROP_ORIENTATION_META: "CAP_PROP_ORIENTATION_META",
    cv.CAP_PROP_ORIENTATION_AUTO: "CAP_PROP_ORIENTATION_AUTO",
    cv.CAP_PROP_HW_ACCELERATION: "CAP_PROP_HW_ACCELERATION",
    cv.CAP_PROP_HW_DEVICE: "CAP_PROP_HW_DEVICE",
    cv.CAP_PROP_HW_ACCELERATION_USE_OPENCL: "CAP_PROP_HW_ACCELERATION_USE_OPENCL",
    cv.CAP_PROP_OPEN_TIMEOUT_MSEC: "CAP_PROP_OPEN_TIMEOUT_MSEC",
    cv.CAP_PROP_READ_TIMEOUT_MSEC: "CAP_PROP_READ_TIMEOUT_MSEC",
    cv.CAP_PROP_STREAM_OPEN_TIME_USEC: "CAP_PROP_STREAM_OPEN_TIME_USEC",
    cv.CAP_PROP_VIDEO_TOTAL_CHANNELS: "CAP_PROP_VIDEO_TOTAL_CHANNELS",
    cv.CAP_PROP_VIDEO_STREAM: "CAP_PROP_VIDEO_STREAM",
    cv.CAP_PROP_AUDIO_STREAM: "CAP_PROP_AUDIO_STREAM",
    cv.CAP_PROP_AUDIO_POS: "CAP_PROP_AUDIO_POS",
    cv.CAP_PROP_AUDIO_SHIFT_NSEC: "CAP_PROP_AUDIO_SHIFT_NSEC",
    cv.CAP_PROP_AUDIO_DATA_DEPTH: "CAP_PROP_AUDIO_DATA_DEPTH",
    cv.CAP_PROP_AUDIO_SAMPLES_PER_SECOND: "CAP_PROP_AUDIO_SAMPLES_PER_SECOND",
    cv.CAP_PROP_AUDIO_BASE_INDEX: "CAP_PROP_AUDIO_BASE_INDEX",
    cv.CAP_PROP_AUDIO_TOTAL_CHANNELS: "CAP_PROP_AUDIO_TOTAL_CHANNELS",
    cv.CAP_PROP_AUDIO_TOTAL_STREAMS: "CAP_PROP_AUDIO_TOTAL_STREAMS",
    cv.CAP_PROP_AUDIO_SYNCHRONIZE: "CAP_PROP_AUDIO_SYNCHRONIZE",
    cv.CAP_PROP_LRF_HAS_KEY_FRAME: "CAP_PROP_LRF_HAS_KEY_FRAME",
    cv.CAP_PROP_CODEC_EXTRADATA_INDEX: "CAP_PROP_CODEC_EXTRADATA_INDEX",
    cv.CAP_PROP_FRAME_TYPE: "CAP_PROP_FRAME_TYPE",
    cv.CAP_PROP_N_THREADS: "CAP_PROP_N_THREADS",
    cv.CAP_PROP_PTS: "CAP_PROP_PTS",
    cv.CAP_PROP_DTS_DELAY: "CAP_PROP_DTS_DELAY"
}


def camera_name(camera_index):
    cmd = f"udevadm info --name=/dev/video{camera_index}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to execute command: {cmd}")

    lines = result.stdout.split("\n")
    read_value = lambda key: next(s for s in lines if f"E: {key}" in s).split("=")[-1]

    return read_value("ID_MODEL") + "--" + read_value("ID_SERIAL_SHORT")


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

    if len(available_cameras) == 1:
        return available_cameras[0]["index"]

    while True:
        try:
            camera_index = int(input("Enter the camera index then press enter:"))
            if any(cam["index"] == camera_index for cam in available_cameras):
                return camera_index
        except ValueError:
            pass


def capture(camera_index):
    cap = cv.VideoCapture(camera_index)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    return cap


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


def detect_acceptable_values(cap, prop, min, max):
    acceptable_values = []
    prev_value = cap.get(prop)
    for value in range(min, max + 1):
        cap.set(prop, value)
        if cap.get(prop) == value:
            acceptable_values.append(value)
    cap.set(prop, prev_value)
    return acceptable_values


def print_properties(cap, all=False):
    print("Camera properties:")
    for prop, name in (ALL_CAMERA_PROPERTIES if all else CAMERA_PROPERTIES).items():
        print(f" - {name}: {cap.get(prop)}")


def camera_settings_path(camera_index, reset=False):
    name = camera_name(camera_index)
    return f"camera-settings/{name}{'.reset' if reset else ''}.yaml"


def reset_properties(cap, camera_index):
    try:
        with open(camera_settings_path(camera_index, reset=True), 'r') as file:
            settings = yaml.safe_load(file)
        for name, value in settings["properties"].items():
            prop = next(p for p, n in CAMERA_PROPERTIES.items() if n == name)
            cap.set(prop, value)
        print("Properties have been reset to default values.")
    except FileNotFoundError:
        print(f"Reset file not found: {camera_settings_path(camera_index, reset=True)}")


def load_properties(cap, camera_index):
    @monitor_property_changes(cap)
    def func():
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        try:
            with open(camera_settings_path(camera_index), 'r') as file:
                settings = yaml.safe_load(file)
            for name, value in settings["properties"].items():
                prop = next(p for p, n in CAMERA_PROPERTIES.items() if n == name)
                cap.set(prop, value)
        except FileNotFoundError:
            return


def load_calibration(camera_index=None, camera_name=None):
    try:
        path = f"camera-settings/{camera_name}.yaml" if camera_index is None else camera_settings_path(camera_index)
        with open(path, 'r') as file:
            settings = yaml.safe_load(file)
        camera_matrix = np.array(settings["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(settings["dist_coeffs"], dtype=np.float32)
        return camera_matrix, dist_coeffs
    except FileNotFoundError:
        return None, None


def save_properties(cap, camera_index):
    @monitor_property_changes(cap)
    def func():
        properties = {name: cap.get(prop) for prop, name in CAMERA_PROPERTIES.items()}
        _update_yaml(camera_settings_path(camera_index), {"properties": properties})


def save_calibration(camera_matrix, dist_coeffs, camera_index):
    updates = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist()
    }
    _update_yaml(camera_settings_path(camera_index), updates)


def _update_yaml(path, updates):
    try:
        with open(path, 'r') as file:
            settings = yaml.safe_load(file) or {}
    except FileNotFoundError:
        settings = {}

    settings.update(updates)

    with open(path, 'w') as file:
        yaml.safe_dump(settings, file)
