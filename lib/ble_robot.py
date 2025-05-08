import asyncio
import threading
import traceback
import subprocess

from bleak import BleakClient

# To check reception of the data:
# - Plug the FTDI module - which is connected to the HM10 module - into the USB port.
# - Run: `minicom -b 9600 -D /dev/ttyUSB0`
#
# ble-serial creates a virtual serial port over BLE:
# - Scan for BLE devices:
#   `./.venv/bin/ble-scan`
# - Connect to a BLE device:
#   `./.venv/bin/ble-serial -d 68:5E:1C:26:76:7C`
#   This creates a symlink in /tmp/ttyBLE. You can then run `minicom -b 9600 -D /tmp/ttyBLE` and type in the terminal.
#   The data will be sent to the BLE device and received via the other minicom.

# Replace with your robot's BLE address and characteristic UUID
ROBOT_BLE_ADDRESS = "68:5E:1C:31:9E:4B"
TEST_BOARD_BLE_ADDRESS = "68:5E:1C:26:76:7C"
BLE_ADDRESS = ROBOT_BLE_ADDRESS # Change to TEST_BOARD_BLE_ADDRESS for the board

CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
FRAME_LENGTH = 130  # Length of the frame to be sent (in bytes)
TIMEOUT_MS = 15.0  # Timeout for BLE operations in milliseconds

# Shared frame and lock for thread safety
frame = bytearray(FRAME_LENGTH)  # Default frame of FRAME_LENGTH octets
frame_lock = threading.Lock()


async def is_device_connected():
    """Checks if the BLE device is currently connected to the laptop."""
    try:
        output = subprocess.check_output(["bluetoothctl", "info", BLE_ADDRESS], text=True)
        return "Connected: yes" in output
    except subprocess.CalledProcessError:
        return False  # Device not found


async def disconnect_device():
    """Forces the BLE device to disconnect if it's already connected."""
    try:
        subprocess.run(["bluetoothctl", "disconnect", BLE_ADDRESS], check=True)
        print("Forced device disconnection.")
        await asyncio.sleep(2)  # Give it a moment to disconnect
    except subprocess.CalledProcessError:
        print("Failed to disconnect the device.")


async def send_frame():
    """Asynchronous function to send the frame periodically."""
    print("Connecting to the robot...")

    if await is_device_connected():
        print("Device is already connected. Disconnecting...")
        await disconnect_device()

    client = BleakClient(BLE_ADDRESS, timeout=TIMEOUT_MS)

    try:
        await client.connect()
        print("Connected to the robot")

        for service in client.services:
            print("Service:", service, " - UUID:", service.uuid)
            for char in service.characteristics:
                print("Characteristic:", char, " - UUID:", char.uuid)

        print("Starting transmission...")
        while True:
            with frame_lock:
                data_to_send = frame[:]  # Copy the current frame safely

            try:
                await client.write_gatt_char(CHARACTERISTIC_UUID, data_to_send, response=False)
                print("Frame sent")
            except Exception as e:
                print(f"Failed to send frame: {e}")

            await asyncio.sleep(0.2)  # 5 times per second

    except asyncio.CancelledError:
        print("Task was cancelled")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    finally:
        await client.disconnect()
        print("Disconnected from the robot")


def start_ble_thread():
    """Starts the asyncio event loop in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_frame())


def update_frame(new_frame: bytes):
    """Updates the frame while ensuring thread safety."""
    if len(new_frame) != FRAME_LENGTH:
        raise ValueError(f"Frame must be exactly {FRAME_LENGTH} bytes, got {len(new_frame)}")
    with frame_lock:
        frame[:] = new_frame


# Start the BLE sender in a background thread
ble_thread = threading.Thread(target=start_ble_thread, daemon=True)
ble_thread.start()
