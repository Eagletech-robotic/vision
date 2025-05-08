# ----------------
# Find Aruco tags in frames.
# Usage: python 12_frame_aruco.py frame_folder
# ----------------

import sys
from lib import detection, vision, camera, frames


def main():
    if sys.argv[1:]:
        frame_folder = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} frame_folder")
        exit(1)

    # Load frames
    images = frames.import_from_folder(frame_folder)

    # Initialize camera and detector
    camera_matrix, dist_coeffs = camera.load_calibration(camera_name="W4DS--SN0001")
    aruco_detector = detection.build_aruco_detector()

    for frame_nb in sorted(images.keys()):
        print(f"Frame {frame_nb}")
        image = images[frame_nb]

        # Aruco detection
        corners, ids, _rejected = aruco_detector.detectMarkers(image)

        # Pose estimation
        ret, rvec, tvec = \
            vision.estimate_pose(corners, ids, vision.FIELD_MARKERS, camera_matrix, dist_coeffs)

        if ret:
            camera_pos = vision.get_camera_position(rvec, tvec)
            print(f"  Camera Position: X={camera_pos[0]:.3f}, Y={camera_pos[1]:.3f}, Z={camera_pos[2]:.3f}")

        for id, corner in zip(ids, corners):
            marker_id = id[0]
            center = corner[0].mean(axis=0)
            z_world = 0.0 if marker_id == 7 else vision.z_world(marker_id)  # HACK: Marker 7 is on the ground
            world_point = vision.image_to_world_point(center, z_world, rvec, tvec, camera_matrix, dist_coeffs)
            print(f"  Tag {marker_id}: {world_point}")


if __name__ == "__main__":
    main()
