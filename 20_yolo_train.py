import torch
from ultralytics import YOLO
from pathlib import Path

# Before running this script, have the following folders configured in your ~/.config/Ultralytics/settings.json:
#   "datasets_dir": "/home/antoine/Desktop/robot/vision/yolo-datasets",
#   "weights_dir": "/home/antoine/Desktop/robot/vision/yolo-runs/weights",
#   "runs_dir": "/home/antoine/Desktop/robot/vision/yolo-runs",

torch.set_num_threads(8)

name = "20250323_01"

model = YOLO("yolo11m.pt").to("cpu")
print(f"Model loaded on device: {model.device}")

results = model.train(
    data=Path(f"./yolo-datasets/{name}/data.yaml").resolve(),
    epochs=20,
    imgsz=640,
    batch=8,  # 3: max value on NVIDIA GTX 1650
    name=name,
    exist_ok=True,
)
