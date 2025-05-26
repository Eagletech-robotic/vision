#!/usr/bin/env python3
"""Extract depth model training images from log files."""

import os
import random
import cv2
from pathlib import Path

# Configuration
NUM_IMAGES = 4
INPUT_DIR = Path(__file__).parent / "logs"
OUTPUT_DIR = Path(__file__).parent / "depth-model/in"

# Crop margins
TOP_MARGIN = 50
BOTTOM_OFFSET = 40
SIDE_MARGIN = 100
CENTER_OVERLAP = 60


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all jpg images
    all_images = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_images.append(Path(root) / file)

    if not all_images:
        print("No images found!")
        return

    # Select random images
    selected = random.sample(all_images, min(NUM_IMAGES, len(all_images)))

    # Process each image
    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Randomly choose side
        side = random.choice(['left', 'right'])

        # Crop image
        if side == 'left':
            cropped = img[TOP_MARGIN:h // 2 - BOTTOM_OFFSET, SIDE_MARGIN:w // 2 - CENTER_OVERLAP]
            suffix = '-1'
        else:
            cropped = img[TOP_MARGIN:h // 2 - BOTTOM_OFFSET, w // 2 + CENTER_OVERLAP:w - SIDE_MARGIN]
            suffix = '-2'

        # Save
        output_file = OUTPUT_DIR / f"{img_path.stem}{suffix}.jpg"
        cv2.imwrite(str(output_file), cropped)
        print(f"Saved: {output_file.name}")


if __name__ == '__main__':
    main()
