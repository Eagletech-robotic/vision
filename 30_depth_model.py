import cv2 as cv

from lib import common, camera
from lib.depth_estimator import DepthEstimator

camera_index = camera.pick_camera()
cap = camera.capture(camera_index)
camera.load_properties(cap, camera_index)

depth_estimator = DepthEstimator(model_size='small', device='cpu')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    depth_map = depth_estimator.estimate_depth(frame)
    depth_colored = depth_estimator.colorize_depth(depth_map)
    common.show_in_window("Depth model", depth_colored)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
