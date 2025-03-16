# ----------------
# Capture one image each second, save as numbered images.
# Usage: python 11_capture.py output_folder
# ----------------

import cv2 as cv
import sys
import os

from lib import common, camera


def main():
    if sys.argv[1:]:
        output_folder = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} output_folder")
        exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Select and initialize camera
    common.run_hw_diagnostics()
    camera_index = camera.pick_camera()
    cap = camera.capture(camera_index)
    camera.load_properties(cap, camera_index)

    print(f"Saving images to {output_folder}. Press 'q' to quit.")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            common.show_in_window("image", frame)

            # Save frame as image
            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv.imwrite(filename, frame)
            print(f"Saved {filename}")

            frame_count += 1

            key = cv.waitKey(1000)
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()
        print(f"Saved {frame_count} images to {output_folder}")


if __name__ == "__main__":
    main()
