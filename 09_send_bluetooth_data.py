from lib import ble_robot
import time

ble_robot.start_ble_thread(ble_robot.MacAddress.TEST_BOARD, nb_frames_per_second=0.1)

data = "".join(chr(65 + (i % 26)) for i in range(ble_robot.FRAME_LENGTH - 2)) + "\r\n"
ble_robot.update_frame(data.encode())

while True:
    time.sleep(1)
