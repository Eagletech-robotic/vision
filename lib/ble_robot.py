import asyncio
import bleak
import threading
import traceback
import subprocess
import re
from datetime import datetime
from collections import deque


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

# --------------------------------------------------------------------------- #
# Public constants                                                             #
# --------------------------------------------------------------------------- #

class MacAddress:
    ROBOT = "68:5E:1C:31:9E:4B"
    TEST_BOARD = "68:5E:1C:26:76:7C"


CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
FRAME_LENGTH = 111  # Length of the frame to be sent (in bytes)
BLEAK_TIMEOUT = 10.0  # Timeout for BLE operations (in seconds)

# --------------------------------------------------------------------------- #
# Internal state (shared between threads)                                      #
# --------------------------------------------------------------------------- #

asyncio_loop: asyncio.AbstractEventLoop | None = None
ble_client: bleak.BleakClient | None = None
rx_char = None  # resolved GATT characteristic
chunk_size = 20  # updated after connect

_pending_frame: bytes | None = None  # newest frame waiting to be sent
_sending = False  # True while a BLE write is running
_state_lock = threading.Lock()

_stdout_buffer = ""
_external_buffer = deque(maxlen=1024)
_external_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _clean_ansi_and_control(text: str) -> str:
    """Strip ANSI sequences and non-printables (keep LF/CR)."""
    text = re.sub(r'\x1b\[[0-9;]*[mK]', '', text)
    text = re.sub(r'\x1b\[H', '', text)
    text = re.sub(r'\x1b\[2J', '', text)
    text = re.sub(r'\x1b\[\d+;\d+H', '', text)
    text = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', text)
    return ''.join(c for c in text if ord(c) >= 32 or c in '\n\r')


def _timestamped(tag: str, msg: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"[{ts}] [{tag}] {msg}"


# --------------------------------------------------------------------------- #
# RX path — called from BLE loop thread                                        #
# --------------------------------------------------------------------------- #

def _on_packet_received(_, data: bytearray):
    global _stdout_buffer

    try:
        chunk = data.decode("ascii", errors="replace")
        _stdout_buffer += _clean_ansi_and_control(chunk)

        # Cap buffer size
        if len(_stdout_buffer) > 4096:
            _stdout_buffer = _stdout_buffer[-4096:]

        # Print all chars received before the last \n, if any
        if '\n' in _stdout_buffer:
            parts = _stdout_buffer.split('\n')
            for line in parts[:-1]:
                line_with_time = _timestamped("RX", line)
                print(line_with_time, flush=True)
                with _external_lock:
                    _external_buffer.append(line_with_time)
            _stdout_buffer = parts[-1]

    except Exception:
        line_with_time = _timestamped("RX-hex", data.hex())
        print(line_with_time, flush=True)
        with _external_lock:
            _external_buffer.append(line_with_time)


# --------------------------------------------------------------------------- #
# Connection manager — runs in its own thread                                  #
# --------------------------------------------------------------------------- #

async def _is_device_connected(ble_address: str) -> bool:
    try:
        output = subprocess.check_output(["bluetoothctl", "info", ble_address], text=True)
        return "Connected: yes" in output
    except subprocess.CalledProcessError:
        return False  # Device not found


async def _disconnect_device(ble_address: str):
    try:
        subprocess.run(["bluetoothctl", "disconnect", ble_address], check=True)
        print("Forced device disconnection.")
        await asyncio.sleep(2)
    except subprocess.CalledProcessError:
        print("Failed to disconnect the device.")


async def _connection_manager(ble_address: str):
    """
    Keep the BLE link alive and dispatch notifications.
    """
    global ble_client, rx_char, chunk_size
    while True:
        client = None
        try:
            print("Connecting to the robot…")

            if await _is_device_connected(ble_address):
                print("Device already connected – disconnecting first.")
                await _disconnect_device(ble_address)

            client = bleak.BleakClient(ble_address, timeout=BLEAK_TIMEOUT)
            await client.connect()
            await client.get_services()

            rx_char = client.services.get_characteristic(CHARACTERISTIC_UUID)
            # BlueZ always says 23, so fall back to 20
            chunk_size = getattr(rx_char, "max_write_without_response_size", None) or 20

            # Listen for incoming data
            await client.start_notify(rx_char, _on_packet_received)
            print(f"Connected. Chunk size = {chunk_size} B.")

            ble_client = client
            while True:
                await asyncio.sleep(3600)  # stay alive, nothing else to do

        except bleak.exc.BleakDeviceNotFoundError as e:
            print(e)

        except Exception as e:
            print("BLE error:", e)
            traceback.print_exc()

        finally:
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    pass
            ble_client = None
            rx_char = None
            print("Disconnected – reconnecting soon.")
            await asyncio.sleep(2)


def _ble_thread(ble_address: str):
    """
    Thread entry: create a new asyncio loop and run the connection manager.
    """
    global asyncio_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio_loop = loop
    loop.run_until_complete(_connection_manager(ble_address))


# --------------------------------------------------------------------------- #
# TX helpers (run inside asyncio loop)                                         #
# --------------------------------------------------------------------------- #

async def _send_payload_async(payload: bytes):
    for off in range(0, len(payload), chunk_size):
        await ble_client.write_gatt_char(rx_char, payload[off:off + chunk_size], response=False)


def _schedule_next_send():
    """
    Called with _state_lock held. If a pending frame exists and no send is
    running, schedule the async write in the BLE loop.
    """
    global _pending_frame, _sending

    if _sending or _pending_frame is None or asyncio_loop is None \
            or ble_client is None or rx_char is None:
        return

    payload = _pending_frame
    _pending_frame = None
    _sending = True

    future = asyncio.run_coroutine_threadsafe(_send_payload_async(payload), asyncio_loop)

    def _on_done(fut):
        global _sending
        print("SENT SUCCESS")
        exc = fut.exception()
        if exc:
            print("Failed to send frame:", exc)
        with _state_lock:
            _sending = False
            _schedule_next_send()  # maybe another frame arrived meanwhile

    future.add_done_callback(_on_done)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def start_ble_thread(ble_address: str):
    """
    Kick off the BLE thread (rx only).
    """
    th = threading.Thread(target=_ble_thread, args=(ble_address,), daemon=True)
    th.start()


def send_frame(frame: bytes):
    """
    Enqueue a frame for transmission. Non-blocking; at most one write is active
    at any time, and only the newest enqueued frame is kept.
    """
    if len(frame) != FRAME_LENGTH:
        raise ValueError(f"Frame must be {FRAME_LENGTH} bytes, got {len(frame)}")

    global _pending_frame
    with _state_lock:
        _pending_frame = frame[:]  # copy to decouple from caller
        _schedule_next_send()


def read_buffer():
    """
    Return and clear text received from the robot.
    """
    with _external_lock:
        lines = list(_external_buffer)
        _external_buffer.clear()
    return lines
