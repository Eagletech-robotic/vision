# eagle_encode.py
from typing import List, Tuple
import math

# ----------------------------------------------------------------------
# Constants that match the C++ / TS implementations
# ----------------------------------------------------------------------
PAYLOAD_LEN = 128  # payload only (no starter / checksum)
FRAME_LEN = 130  # 0xFF + payload + checksum
MAX_OBJECTS = 60


# ----------------------------------------------------------------------
# Bit‑level helper
# ----------------------------------------------------------------------
def _push_bits(bit_list: List[int], value: int, n_bits: int) -> None:
    """Append the `n_bits` least‑significant bits of value to bit_list (LSB‑first)."""
    for i in range(n_bits):
        bit_list.append((value >> i) & 1)


def _bits_to_bytes(bits: List[int]) -> bytes:
    """Pack a list of bits (LSB‑first) into a bytes object (LSB‑first inside each byte)."""
    out = bytearray((len(bits) + 7) // 8)
    for idx, bit in enumerate(bits):
        if bit:
            out[idx >> 3] |= 1 << (idx & 7)
    # pad up to 128 bytes
    if len(out) < PAYLOAD_LEN:
        out.extend(b"\x00" * (PAYLOAD_LEN - len(out)))
    return bytes(out)


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def build_payload(
        robot_colour: str,  # "blue" or "yellow"
        robot_detected: bool,
        robot_pose: Tuple[float, float, float],  # x, y, theta_rad
        opponent_detected: bool,
        opponent_pose: Tuple[float, float, float],  # x, y, theta_rad
        bleachers: List[Tuple[float, float, float]],  # list[(x, y, theta_rad)]
) -> bytes:
    """
    Build the 128‑byte Eagle payload (no starter / checksum).

    All positions are in **metres**, orientations in **radians**.
    Only bleachers are encoded as objects (type 0).
    """
    bits: List[int] = []
    to_cm = lambda m: round(m * 100)  # metres → integer cm
    to_deg = lambda r: round(r * 180 / math.pi) % 360  # rad → deg

    # ───────── header ────────────────────────────────────────────────
    # Packet layout (little-endian bit order)
    #   bit 0      : robot_colour (0=blue,1=yellow)
    #   bit 1      : robot_detected
    #   bits 2-10  : robot_x (cm)
    #   bits 11-18 : robot_y (cm)
    #   bits 19-27 : robot_orientation (deg + 180)
    #   bit 28     : opponent_detected
    #   bits 29-37 : opponent_x (cm)
    #   bits 38-45 : opponent_y (cm)
    #   bits 46-54 : opponent_orientation (deg + 180)
    #   bits 55-60 : object_count (0-60)
    #   bits 61-63 : padding (0)
    # -----------------------------------------------------------------

    _push_bits(bits, 1 if robot_colour == "yellow" else 0, 1)
    _push_bits(bits, 1 if robot_detected else 0, 1)

    _push_bits(bits, to_cm(robot_pose[0]), 9)
    _push_bits(bits, to_cm(robot_pose[1]), 8)
    _push_bits(bits, to_deg(robot_pose[2]) & 0x1FF, 9)

    _push_bits(bits, 1 if opponent_detected else 0, 1)
    _push_bits(bits, to_cm(opponent_pose[0]), 9)
    _push_bits(bits, to_cm(opponent_pose[1]), 8)
    _push_bits(bits, to_deg(opponent_pose[2]) & 0x1FF, 9)

    # Truncate list before encoding the count so that the encoded value is consistent
    bleachers = bleachers[:MAX_OBJECTS]  # cap at 60
    _push_bits(bits, len(bleachers), 6)  # object_count
    _push_bits(bits, 0, 3)  # padding bits 61-63

    # ───────── objects ───────────────────────────────────────────────
    for x, y, theta_rad in bleachers:
        _push_bits(bits, 0, 2)  # type 0 = Bleacher
        raw_x = round(to_cm(x) * 63 / 300) & 0x3F
        raw_y = round(to_cm(y) * 31 / 200) & 0x1F
        _push_bits(bits, raw_x, 6)
        _push_bits(bits, raw_y, 5)
        _push_bits(bits, round((to_deg(theta_rad) % 180) / 30) & 0x7, 3)

    return _bits_to_bytes(bits)  # exactly 128 bytes


def frame_payload(payload: bytes) -> bytes:
    """
    Prefix 0xFF and append 8‑bit checksum (sum(payload) & 0xFF).
    Raises if payload is not exactly 128 bytes.
    """
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"payload must be {PAYLOAD_LEN} bytes")

    checksum = sum(payload) & 0xFF
    return b"\xFF" + payload + bytes((checksum,))


# ----------------------------------------------------------------------
# Pretty‑printer / decoder  (debug & logging aid)
# ----------------------------------------------------------------------
def _bits_from_payload(payload: bytes) -> List[int]:
    """Expand 128‑byte payload into an LSB‑first bit list."""
    bits: List[int] = []
    for byte in payload:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


def _pop(bits: List[int], n_bits: int) -> int:
    """Pop n_bits (LSB‑first) from the front of the list and return the value."""
    v = 0
    for i in range(n_bits):
        v |= bits.pop(0) << i
    return v


def frame_to_human(frame: bytes) -> str:
    """
    Decode a 130‑byte Eagle frame and return a human‑friendly multi‑line string.
    Raises ValueError if frame is malformed (wrong length, bad checksum, no 0xFF).
    """
    if len(frame) != FRAME_LEN:
        raise ValueError(f"expected {FRAME_LEN} bytes, got {len(frame)}")
    if frame[0] != 0xFF:
        raise ValueError("frame must start with 0xFF")
    if (sum(frame[1:-1]) & 0xFF) != frame[-1]:
        raise ValueError("checksum mismatch")

    payload = frame[1:-1]
    bits = _bits_from_payload(payload)

    colour = "yellow" if _pop(bits, 1) else "blue"
    robot_detected = bool(_pop(bits, 1))

    robot_x_cm = _pop(bits, 9)
    robot_y_cm = _pop(bits, 8)
    robot_theta_deg = _pop(bits, 9)

    opponent_detected = bool(_pop(bits, 1))

    opp_x_cm = _pop(bits, 9)
    opp_y_cm = _pop(bits, 8)
    opp_theta_deg = _pop(bits, 9)

    obj_count = _pop(bits, 6)
    _pop(bits, 3)  # padding

    objects = []
    for _ in range(obj_count):
        o_type_raw = _pop(bits, 2)
        raw_x = _pop(bits, 6)
        raw_y = _pop(bits, 5)
        raw_theta = _pop(bits, 3)
        x_cm = round(raw_x * 300 / 63)
        y_cm = round(raw_y * 200 / 31)
        theta = raw_theta * 30
        o_type = ["Bleacher", "Plank", "Can"][o_type_raw]
        objects.append((o_type, x_cm, y_cm, theta))

    # -------- build pretty string -----------------------------------
    lines = [
        "⇢ Eagle frame (human readable)",
        f"  Colour                : {colour}",
        f"  Robot detected        : {robot_detected}",
        f"  Robot   (cm,deg)      : x={robot_x_cm}, y={robot_y_cm}, theta={robot_theta_deg}",
        f"  Opponent detected     : {opponent_detected}",
        f"  Opponent(cm,deg)      : x={opp_x_cm}, y={opp_y_cm}, theta={opp_theta_deg}",
        f"  Objects ({obj_count})",
    ]
    for i, (typ, x, y, th) in enumerate(objects):
        lines.append(f"    {i:02d}  {typ:<8} x={x:3d}  y={y:3d}  θ={th:3d}")

    return "\n".join(lines)
