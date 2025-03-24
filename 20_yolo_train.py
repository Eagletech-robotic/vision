from ultralytics import YOLO

model = YOLO("yolo11m.pt").to("cpu")
print(f"Model loaded on device: {model.device}")

results = model.train(data="yolo-datasets/20250323_01/data.yaml",
                      epochs=10,
                      imgsz=640,
                      batch=16,
                      project="yolo-runs/train",
                      name="20250323_01",
                      exist_ok=True)
