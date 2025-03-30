# ----------------
# Reset camera settings by applying the values in the xxx.reset.yaml file.
# ----------------

from lib import camera


def main():
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.reset_properties(cap, camera_index)


if __name__ == "__main__":
    main()
