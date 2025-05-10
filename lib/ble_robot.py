import asyncio
import threading
import traceback
import subprocess
import re

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
BLE_ADDRESS = ROBOT_BLE_ADDRESS  # Change to TEST_BOARD_BLE_ADDRESS for the board

CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
FRAME_LENGTH = 130  # Length of the frame to be sent (in bytes)
NB_FRAMES_PER_SECOND = 5  # Number of frames sent to the robot per second
TIMEOUT = 15.0  # Timeout for BLE operations (in milliseconds)

# Shared frame and lock for thread safety
frame = bytearray(FRAME_LENGTH)  # Default frame of FRAME_LENGTH octets
frame_lock = threading.Lock()

# The buffer of received frames. Printed on screen as soon as we receive "\n".
rx_buffer = ""


def _clean_ansi_and_control(text: str) -> str:
    """Strip ANSI sequences and non‑printables (keep \\n and \\r)"""
    text = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
    text = re.sub(r'\x1b\[H', '', text)
    text = re.sub(r'\x1b\[2J', '', text)
    text = re.sub(r'\x1b\[\d+;\d+H', '', text)
    text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', text)
    return ''.join(c for c in text if ord(c) >= 32 or c in '\n\r')


def _on_packet_received(_, data: bytearray):
    global rx_buffer
    try:
        chunk = data.decode("ascii", errors="replace")
        rx_buffer += _clean_ansi_and_control(chunk)

        # Cap buffer size
        if len(rx_buffer) > 4096:
            rx_buffer = rx_buffer[-4096:]

        # Print all chars received before the last \n, if any
        if '\n' in rx_buffer:
            parts = rx_buffer.split('\n')

            for line in parts[:-1]:
                print(f"[RX] {line}", flush=True)
            rx_buffer = parts[-1]
    except Exception:
        print(f"[RX-hex] {data.hex()}\n")


async def _is_device_connected():
    """Checks if the BLE device is currently connected to the laptop."""
    try:
        output = subprocess.check_output(["bluetoothctl", "info", BLE_ADDRESS], text=True)
        return "Connected: yes" in output
    except subprocess.CalledProcessError:
        return False  # Device not found


async def _disconnect_device():
    """Forces the BLE device to disconnect if it's already connected."""
    try:
        subprocess.run(["bluetoothctl", "disconnect", BLE_ADDRESS], check=True)
        print("Forced device disconnection.")
        await asyncio.sleep(2)  # Give it a moment to disconnect
    except subprocess.CalledProcessError:
        print("Failed to disconnect the device.")


async def _send_frame():
    """Asynchronous function to send the frame periodically."""
    while True:
        client = None  # ensure defined for finally‑block
        try:
            print("Connecting to the robot...")

            if await _is_device_connected():
                print("Device is already connected. Disconnecting...")
                await _disconnect_device()

            client = BleakClient(BLE_ADDRESS, timeout=TIMEOUT_MS / 1000.0)
            await client.connect()
            await client.get_services()

            for service in client.services:
                print("Service:", service, " - UUID:", service.uuid)
                for char in service.characteristics:
                    print("Characteristic:", char, " - UUID:", char.uuid)
            print("Connected to the robot")

            rx_char = client.services.get_characteristic(CHARACTERISTIC_UUID)
            # BlueZ always says 23, so fall back to 20
            chunk_size = getattr(rx_char, "max_write_without_response_size", None) or 20

            # Listen for incoming data
            await client.start_notify(rx_char, _on_packet_received)

            print("Starting transmission...")
            # Send the most recent frame periodically until the connection fails
            while True:
                with frame_lock:
                    payload = bytes(frame)  # snapshot

                try:
                    for off in range(0, len(payload), chunk_size):
                        await client.write_gatt_char(rx_char, payload[off:off + chunk_size], response=False)
                    print("Frame sent")

                except Exception as e:
                    print(f"Failed to send frame: {e}")
                    raise e

                await asyncio.sleep(1.0 / NB_FRAMES_PER_SECOND)

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    pass
            print("Disconnected from the robot - will reconnect")


def _start_ble_thread():
    """Starts the asyncio event loop in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_send_frame())


def update_frame(new_frame: bytes):
    """Updates the frame while ensuring thread safety."""
    if len(new_frame) != FRAME_LENGTH:
        raise ValueError(f"Frame must be exactly {FRAME_LENGTH} bytes, got {len(new_frame)}")
    with frame_lock:
        frame[:] = new_frame


# Start the BLE sender in a background thread
ble_thread = threading.Thread(target=_start_ble_thread, daemon=True)
ble_thread.start()
