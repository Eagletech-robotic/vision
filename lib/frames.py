import os
import cv2 as cv

def import_from_folder(frame_folder):
    """
    :return: Dictionary of images with frame number as key.
    """
    images = {}
    for filename in os.listdir(frame_folder):
        if filename.endswith(".jpg"):
            frame_nb = int(filename.split("_")[1].split(".")[0])
            images[frame_nb] = cv.imread(os.path.join(frame_folder, filename))
    print(f"Loaded {len(images)} images from {frame_folder}")
    return images
