# ----------------
# Analyse the frames in the given folder.
# Usage: python 12_frame_analysis.py frame_folder
# ----------------

import cv2 as cv
import sys
import os
from lib import detection, vision, camera


def main():
    if sys.argv[1:]:
        frame_folder = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} frame_folder")
        exit(1)

    # Select and initialize camera
    camera_matrix, dist_coeffs = camera.load_calibration(camera_name="W4DS--SN0001")

    # Load all images in the folder
    images = {}
    for filename in os.listdir(frame_folder):
        if filename.endswith(".jpg"):
            frame_nb = int(filename.split("_")[1].split(".")[0])
            images[frame_nb] = cv.imread(os.path.join(frame_folder, filename))
    print(f"Loaded {len(images)} images from {frame_folder}")

    # Analyse images
    aruco_detector = detection.build_aruco_detector()

    for frame_nb in sorted(images.keys()):
        print(f"Frame {frame_nb}")
        image = images[frame_nb]

        # Aruco detection
        corners, ids, _rejected = aruco_detector.detectMarkers(image)

        # Pose estimation
        ret, rvec, tvec = \
            vision.estimate_pose(corners, ids, vision.MarkerPositions, camera_matrix, dist_coeffs)

        if ret:
            camera_pos = vision.get_camera_position(rvec, tvec)
            print(f"  Camera Position (mm): X={camera_pos[0]:.1f}, Y={camera_pos[1]:.1f}, Z={camera_pos[2]:.1f}")

        for id, corner in zip(ids, corners):
            marker_id = id[0]
            center = corner[0].mean(axis=0)
            z_world = 0.0 if marker_id == 7 else vision.z_world(marker_id) # HACK: Marker 7 is on the ground
            world_point = vision.image_to_world_point(center, z_world, rvec, tvec, camera_matrix, dist_coeffs)
            print(f"  Tag {marker_id}: {world_point}")


if __name__ == "__main__":
    main()
