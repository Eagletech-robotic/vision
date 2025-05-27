import os
import numpy as np
import cv2 as cv
from pathlib import Path
import torch

from lib.depth_estimator import DepthEstimator


def generate_depth_colored(img, depth_estimator):
    """Generate colorized depth map from input image."""
    depth_map = depth_estimator.estimate_depth(img)
    depth_colored = depth_estimator.colorize_depth(depth_map)
    return depth_colored, depth_map


def generate_depth_with_gradient_subtraction(depth_map, p0=(600, 80), p1=(700, 750)):
    """Apply gradient subtraction to depth map to remove floor plane."""
    # Convert depth map to numpy if needed
    if isinstance(depth_map, torch.Tensor):
        dm = depth_map.detach().cpu().numpy()
    else:
        dm = depth_map

    # Build the t(x,y) weight map
    h, w = dm.shape
    x0, y0 = p0
    vx, vy = p1[0] - x0, p1[1] - y0
    denom = float(vx * vx + vy * vy)
    xs, ys = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    t_map = ((xs - x0) * vx + (ys - y0) * vy) / denom
    one_map = np.ones_like(t_map, dtype=np.float32)

    # Build design matrix and solve least squares
    A0 = one_map.ravel()
    A1 = t_map.ravel()
    A = np.vstack((A0, A1)).T
    b = dm.ravel()
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    z0, dZ = coef

    # Subtract the plane
    plane = z0 + t_map * dZ
    resid = dm - plane
    resid[resid < 0] = 0

    # Colorize the gradient-subtracted result
    res_norm = cv.normalize(resid, None, 0, 255, cv.NORM_MINMAX)
    res_u8 = res_norm.astype(np.uint8)
    res_colored = cv.applyColorMap(res_u8, cv.COLORMAP_TURBO)
    
    return res_colored


def main():
    input_dir = Path("depth-model/in")
    output_dir = Path("depth-model/out")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    depth_estimator = DepthEstimator(model_size='medium', device='cpu')
    
    for image_path in input_dir.glob("*.jpg"):
        print(f"Processing {image_path.name}...")
        
        img = cv.imread(str(image_path))
        if img is None:
            print(f"Failed to read {image_path.name}")
            continue
        
        # Generate depth and gradient-subtracted depth
        depth_colored, depth_map = generate_depth_colored(img, depth_estimator)
        gradient_subtracted = generate_depth_with_gradient_subtraction(depth_map)
        
        # Create 2x2 grid
        h_img, w_img = img.shape[:2]
        
        # Resize all images to match original size
        depth_colored_resized = cv.resize(depth_colored, (w_img, h_img))
        gradient_subtracted_resized = cv.resize(gradient_subtracted, (w_img, h_img))
        blank = np.zeros_like(img)
        
        # Build grid: original | depth | gradient-subtracted | empty
        top_row = np.hstack([img, depth_colored_resized])
        bottom_row = np.hstack([gradient_subtracted_resized, blank])
        grid = np.vstack([top_row, bottom_row])
        
        output_path = output_dir / image_path.name
        cv.imwrite(str(output_path), grid)
        print(f"Saved to {output_path}")
    
    print("Done processing all images")


if __name__ == "__main__":
    main()

