import cv2 as cv

from ultralytics import YOLO
from lib import common, camera

name = "detect/20250323_01"
# name = "obb/20250323_01-obb"

model = YOLO(f"./yolo-runs/{name}/weights/best.pt")
print(f"Model loaded on device: {model.device}")

camera_index = camera.pick_camera()
cap = camera.capture(camera_index)
camera.load_properties(cap, camera_index)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()
    common.show_in_window("Tracking", annotated_frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
