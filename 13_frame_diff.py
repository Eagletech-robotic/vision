# ----------------
# Compute the diff between frames.
# Usage: python 13_frame_diff.py frame_folder
# ----------------

import sys
from lib import detection, vision, camera, frames, common
import cv2 as cv


def main():
    if sys.argv[1:]:
        frame_folder = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} frame_folder")
        exit(1)

    # Load frames
    images = frames.import_from_folder(frame_folder)
    image_indices = sorted(images.keys())

    for image_index in image_indices[1:]:
        def preprocess(src):
            dest = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            dest = cv.GaussianBlur(dest, (5, 5,), 0)
            return dest

        current = preprocess(images[image_index])
        previous = preprocess(images[image_index - 1])
        abs_diff = cv.absdiff(current, previous)

        common.show_in_window("image", abs_diff)

        c = cv.waitKey(1000)
        if c == ord('q'):
            break


if __name__ == "__main__":
    main()
