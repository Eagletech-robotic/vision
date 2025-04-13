import cv2 as cv


def run_hw_diagnostics():
    print(f"CUDA enabled devices: {cv.cuda.getCudaEnabledDeviceCount()}")
    print(f"OpenCL enabled devices: {cv.ocl.haveOpenCL()}")

    device = cv.ocl.Device.getDefault()
    print(f"OpenCL device: {device.name()}")


# Example use:
#   marker_corners, marker_ids, _ = common.measure_time(
#       lambda:
#       cv.aruco.detectMarkers(image, dictionary, parameters=detector_params),
#       name="detectMarkers")()
def measure_time(func, name=None):
    def new_function(*args, **kwds):
        start = cv.getTickCount()
        result = func(*args, **kwds)
        end = cv.getTickCount()
        print("%s took %.6f seconds" % (name or func.__name__, (end - start) / cv.getTickFrequency()))
        return result

    return new_function


def init_window(name):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 1280, 720)


def init_fullscreen_window(name):
    cv.namedWindow(name, cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO | cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(name, 1920, 1080)


def show_in_window(name, image):
    init_window(name)
    cv.imshow(name, image)


def print_fields(params):
    for attr in dir(params):
        if not attr.startswith('_'):  # Skip private attributes
            value = getattr(params, attr)
            if not callable(value):  # Skip methods
                print(f"{attr}: {value}")


def put_text_with_background(img, text, position, font=cv.FONT_HERSHEY_SIMPLEX,
                             font_scale=1, text_color=(255, 255, 255),
                             bg_color=(0, 0, 0), thickness=2, padding=5):
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
    x, y = position
    bg_rect = (
        x - padding,
        y - text_height - padding,
        text_width + (padding * 2),
        text_height + baseline + (padding * 2)
    )

    cv.rectangle(img, (bg_rect[0], bg_rect[1]), (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), bg_color, -1)
    cv.putText(img, text, (x, y), font, font_scale, text_color, thickness)
