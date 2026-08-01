"""Stand-in classifier for bring-up, selected by SMARTBIN_FAKE_VISION.

Swap-in for vision.py so the trigger -> rules -> Bridge -> lights path can be
exercised without the model, the camera, or a bin full of props.
"""

import os
import random

import vision

# Labels are real wastenet classes so the rules engine is genuinely exercised.
# The expected category on each line is written by hand rather than looked up
# from disposal_rules.yaml on purpose: it is an independent oracle, so a wrong
# mapping shows up as a mismatch instead of agreeing with itself. Colour and
# tone come from the table in sketch.ino.
_SEQUENCE = [
    ("cardboard",  0.93, "recyclable",     "blue, 880 Hz"),
    ("plastic",    0.88, "recyclable",     "blue, 880 Hz"),
    ("biological", 0.79, "non-recyclable", "white, 440 Hz"),
    ("shoes",      0.71, "non-recyclable", "white, 440 Hz"),
    ("battery",    0.95, "hazardous",      "red, 1320 Hz"),
    ("cardboard",  0.20, "unknown",        "yellow, 220 Hz - below the 0.65 floor"),
    ("banana",     0.99, "unknown",        "yellow, 220 Hz - label not in the vocabulary"),
]

_index = 0


def _next_case():
    global _index
    if os.environ.get("SMARTBIN_FAKE_VISION") == "random":
        return random.choice(_SEQUENCE)
    case = _SEQUENCE[_index % len(_SEQUENCE)]
    _index += 1
    return case


def capture(count=None):
    # Still exercises the real camera so bring-up covers it, but a missing
    # webcam must not stop the fake labels from flowing.
    try:
        return vision.capture(count)
    except Exception as exc:
        print(f"[fake_vision] camera unavailable ({exc}), continuing without frames")
        return []


def classify(frames):
    label, confidence, expected, signal = _next_case()
    saw = f"{len(frames)} frame(s)" if frames else "no frames"
    print(f"[fake_vision] {saw}: {label} {confidence:.2f} -> expect {expected} ({signal})")
    return label, confidence
