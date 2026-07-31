"""Stand-in classifier for bring-up, selected by SMARTBIN_FAKE_VISION.

Swap-in for vision.py so the trigger -> rules -> Bridge -> lights path can be
validated before a real classifier exists. Delete once Task 1 lands.
"""

import os
import random

import vision

# The expected category on each line is written out by hand rather than looked
# up from disposal_rules.yaml on purpose: it is an independent oracle, so a
# wrong mapping in the rules engine shows up as a mismatch instead of agreeing
# with itself. Colour and tone come from the table in sketch.ino.
_SEQUENCE = [
    ("plastic_bottle", 0.93, "recycle",   "blue, 880 Hz"),
    ("food_scraps",    0.88, "compost",   "green, 660 Hz"),
    ("styrofoam",      0.79, "trash",     "grey, 440 Hz"),
    ("battery",        0.95, "hazardous", "red, 1320 Hz"),
    ("phone",          0.84, "ewaste",    "violet, 1100 Hz"),
    ("plastic_bottle", 0.20, "unknown",   "amber, 220 Hz - below the 0.55 floor"),
    ("banana",         0.99, "unknown",   "amber, 220 Hz - label not in the vocabulary"),
]

_index = 0


def _next_case():
    global _index
    if os.environ.get("SMARTBIN_FAKE_VISION") == "random":
        return random.choice(_SEQUENCE)
    case = _SEQUENCE[_index % len(_SEQUENCE)]
    _index += 1
    return case


def capture():
    # Still exercises the real camera so bring-up covers it, but a missing
    # webcam must not stop the fake labels from flowing.
    try:
        return vision.capture()
    except Exception as exc:
        print(f"[fake_vision] camera unavailable ({exc}), continuing without a frame")
        return None


def classify(frame):
    label, confidence, expected, signal = _next_case()
    saw = "frame" if frame is not None else "no frame"
    print(f"[fake_vision] {saw}: {label} {confidence:.2f} -> expect {expected} ({signal})")
    return label, confidence
