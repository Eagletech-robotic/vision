# eagle_encode.py
from typing import List, Tuple
import math

# ----------------------------------------------------------------------
# Constants that match the C++ / TS implementations
# ----------------------------------------------------------------------
PAYLOAD_LEN = 7  # payload only (no starter / checksum)
FRAME_LEN = PAYLOAD_LEN + 2  # 0xFF + payload + checksum


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
    # pad up to fixed payload length (zeros)
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
) -> bytes:
    """
    Build the 128‑byte Eagle payload (no starter / checksum).
    All positions are in **metres**, orientations in **radians**.
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
    #   bit 55     : padding (0)
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

    return _bits_to_bytes(bits)


def frame_payload(payload: bytes) -> bytes:
    """
    Prefix 0xFF and append 8‑bit checksum (sum(payload) & 0xFF).
    """
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"payload must be {PAYLOAD_LEN} bytes")

    checksum = sum(payload) & 0xFF
    return b"\xFF" + payload + bytes((checksum,))


# ----------------------------------------------------------------------
# Pretty‑printer / decoder  (debug & logging aid)
# ----------------------------------------------------------------------
def _bits_from_payload(payload: bytes) -> List[int]:
    """Expand payload into an LSB‑first bit list."""
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

    # -------- build pretty string -----------------------------------
    lines = [
        "⇢ Eagle frame (human readable)",
        f"  Colour                : {colour}",
        f"  Robot detected        : {robot_detected}",
        f"  Robot   (cm,deg)      : x={robot_x_cm}, y={robot_y_cm}, theta={robot_theta_deg}",
        f"  Opponent detected     : {opponent_detected}",
        f"  Opponent(cm,deg)      : x={opp_x_cm}, y={opp_y_cm}, theta={opp_theta_deg}",
    ]

    return "\n".join(lines)
