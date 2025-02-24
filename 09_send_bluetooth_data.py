from lib import ble_robot
import time

while True:
    new_data = "".join(chr(65 + (i % 26)) for i in range(ble_robot.FRAME_LENGTH - 2)) + "\r\n"
    ble_robot.update_frame(new_data.encode())
    print(f"Updated frame: {new_data}")
    time.sleep(1)  # Simulating updates from main thread
