import cv2 as cv


def measure_time(func):
    def new_function(*args, **kwds):
        start = cv.getTickCount()
        result = func(*args, **kwds)
        end = cv.getTickCount()
        print("%s took %.6f seconds" % (func.__name__, (end - start) / cv.getTickFrequency()))
        return result

    return new_function


def init_window(name):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 1280, 720)

def show_in_window(name, image):
    init_window(name)
    cv.imshow(name, image)


def print_fields(params):
    for attr in dir(params):
        if not attr.startswith('_'):  # Skip private attributes
            value = getattr(params, attr)
            if not callable(value):  # Skip methods
                print(f"{attr}: {value}")

