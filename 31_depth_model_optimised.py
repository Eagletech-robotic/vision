import cv2 as cv
import threading, queue
from lib import common, camera
from lib.depth_estimator import DepthEstimator

# — tunables —
FRAME_SKIP = 2  # run depth every Nth frame
INPUT_SIZE = (640, 480)  # model feed resolution
FIELD_ROI = (80, 0, 560, 480)  # x1, y1, x2, y2 of the flat field


# ———————————

class AsyncCamera:
    """Non‑blocking VideoCapture that always returns the latest frame."""

    def __init__(self, index, size):
        self.cap = camera.capture(index)
        camera.load_properties(self.cap, index)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, size[1])
        self.q = queue.Queue(maxsize=1)
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            cap.grab() # Evict any stale images from the one-image buffer (CAP_PROP_BUFFERSIZE=1)
            ok, frame = self.cap.read()
            if not ok:
                self.running = False
                break
            if self.q.full():
                _ = self.q.get_nowait()  # drop stale frame
            self.q.put_nowait(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.running = False
        self.cap.release()


camera_index = camera.pick_camera()
cam = AsyncCamera(camera_index, INPUT_SIZE)
depth_est = DepthEstimator(model_size='small', device='cpu')

frame_id, last_depth_color = 0, None
while True:
    frame = cam.read()
    if frame is None:
        print("Camera offline");
        break

    # ➊ restrict work to the robot field
    x1, y1, x2, y2 = FIELD_ROI
    roi = frame[y1:y2, x1:x2]
    roi_small = cv.resize(roi, INPUT_SIZE, interpolation=cv.INTER_AREA)

    # ➋ run depth only every FRAME_SKIP frames
    if frame_id % FRAME_SKIP == 0:
        depth = depth_est.estimate_depth(roi_small)
        last_depth_color = depth_est.colorize_depth(depth)
    frame_id += 1

    # ➌ show (upsample depth back to ROI size)
    if last_depth_color is not None:
        depth_vis = cv.resize(last_depth_color, (roi.shape[1], roi.shape[0]),
                              interpolation=cv.INTER_NEAREST)
        display = frame.copy()
        display[y1:y2, x1:x2] = depth_vis
        common.show_in_window("Depth view", display)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
