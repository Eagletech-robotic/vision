import torch

import cv2 as cv

from lib import common, camera
from lib.depth_estimator import DepthEstimator

use_xpu = True # Set to True Intel Arc

if use_xpu:
    torch.xpu.set_device(0)

camera_index = camera.pick_camera()
cap = camera.capture(camera_index)
camera.load_properties(cap, camera_index)

if use_xpu:
    depth_estimator = DepthEstimator(model_size='medium', device='xpu')
else:
    depth_estimator = DepthEstimator(model_size='medium', device='cpu')

while True:
    cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv.resize(frame, (1280, 800))
    depth_map = depth_estimator.estimate_depth(frame)
    depth_colored = depth_estimator.colorize_depth(depth_map)
    common.show_in_window("Depth model", depth_colored)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
