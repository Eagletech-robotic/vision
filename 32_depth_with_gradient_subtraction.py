import numpy as np
import cv2 as cv
import torch
import intel_extension_for_pytorch as ipex  # before torch
from lib import common, camera
from lib.depth_estimator import DepthEstimator

# --- CONFIG: pick two field‐corner points for your flat field gradient — EDIT THESE
P0 = (600,  80)    # e.g. top‐left corner
P1 = (700, 750)   # e.g. bottom‐right corner
PTS = [P0, P1]

# init camera + depth estimator
camera_index    = camera.pick_camera()
cap             = camera.capture(camera_index)
camera.load_properties(cap, camera_index)
torch.xpu.set_device(0)
depth_estimator = DepthEstimator(model_size='medium', device='xpu')

# create two named windows once
cv.namedWindow("Depth – objects only", cv.WINDOW_NORMAL)
cv.namedWindow("Frame + picks",       cv.WINDOW_NORMAL)

while True:
    cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (1280, 800))
    depth_map = depth_estimator.estimate_depth(frame)

    # ---- inside the main frame loop, right after you get depth_map -------
    if isinstance(depth_map, torch.Tensor):
        dm = depth_map.detach().cpu().numpy()
    else:
        dm = depth_map  # (H, W) float32

    # 1. pre-compute the t(x,y) weight map (once per resolution)
    h, w = dm.shape
    if 't_map' not in globals():
        x0, y0 = P0
        vx, vy = P1[0] - x0, P1[1] - y0
        denom = float(vx * vx + vy * vy)
        xs, ys = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        t_map = ((xs - x0) * vx + (ys - y0) * vy) / denom  # shape (H, W)
        one_map = np.ones_like(t_map, dtype=np.float32)

    # 2. build the 2-column design matrix A and RHS b (all flattened)
    A0 = one_map.ravel()  # column for z0
    A1 = t_map.ravel()  # column for Δ
    A = np.vstack((A0, A1)).T  # (N, 2)
    b = dm.ravel()  # (N,)

    # 3. least-squares solution  (z0, Δ)  in one NumPy call
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    z0, dZ = coef  # Δ = dZ = z1 - z0

    # 4. build & subtract the plane
    plane = z0 + t_map * dZ
    resid = dm - plane
    resid[resid < 0] = 0  # keep only objects rising above the floor

    # 5. median residual = “how flat did we make the floor?”
    flatness = float(np.median(np.abs(resid)))
    print(f"median residual = {flatness:.4f}")

    # 6. colour-map residual & show as before
    res_norm = cv.normalize(resid, None, 0, 255, cv.NORM_MINMAX)
    res_u8 = res_norm.astype(np.uint8)
    res_col = cv.applyColorMap(res_u8, cv.COLORMAP_TURBO)
    cv.imshow("Depth – objects only", res_col)

    # annotate picks
    disp = frame.copy()
    for (x, y) in (P0, P1):
        cv.drawMarker(disp, (x, y), (0, 255, 255),
                      markerType=cv.MARKER_CROSS, markerSize=20, thickness=2)
    cv.imshow("Frame + picks", disp)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
