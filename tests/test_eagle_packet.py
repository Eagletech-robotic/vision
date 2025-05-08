import math
import os
import sys

# Ensure project root is on PYTHONPATH so that `lib` can be imported when the
# tests are executed from any working directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.eagle_packet import build_payload


def test_manual_bit_pattern_one_object():
    # Payload built manually bit-by-bit in C++ reference test (test_eagle_packet.cpp).
    # Header (8 bytes) + one object (2 bytes) → 10-byte payload.
    reference_payload = bytes(
        [
            0b00101010,
            0b10100000,
            0b10010000,
            0b10110110,
            0b10000000,
            0b10000001,
            0b10010110,
            0b00000000,
            0b00001100,
            0b01000101,
        ]
    ) + b"\x00" * (128 - 10)

    # Build the same payload with the Python encoder
    generated_payload = build_payload(
        robot_colour="blue",
        robot_pose=(0.10, 0.20, math.radians(210)),  # 10 cm, 20 cm, 210°
        opponent_pose=(0.05, 0.06, math.radians(90)),  # 5 cm, 6 cm, 90°
        bleachers=[(0.14, 0.32, math.radians(60))],  # one bleacher 14 cm, 32 cm, 60°
        robot_detected=True,
        opponent_detected=True,
    )

    # The encoder should produce exactly the same 128-byte payload.
    assert generated_payload == reference_payload, (
        "Encoded payload does not match the reference bit pattern from "
        "test_eagle_packet.cpp"
    ) 