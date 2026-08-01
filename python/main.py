import os
import queue
from datetime import datetime
from pathlib import Path

import cv2
from arduino.app_utils import App, Bridge

from rules import Rules
from store import EventStore

# The real classifier is the default now. Set SMARTBIN_FAKE_VISION=1 to swap in
# the stand-in, which is still the quickest way to exercise all six categories
# without hunting for objects the model recognises.
USE_FAKE_VISION = os.environ.get("SMARTBIN_FAKE_VISION", "0") != "0"

if USE_FAKE_VISION:
    import fake_vision as vision

    print("[bin] using the stand-in classifier")
else:
    import vision

# Every trigger writes its frame here so the camera's real view can be inspected
# after the fact, and so the vision owner gets labelled sample images for free.
CAPTURE_DIR = Path(
    os.environ.get("SMARTBIN_CAPTURES", Path(__file__).with_name("captures"))
)

_rules = Rules()
_store = EventStore()

# on_trigger has to return instantly, so it only hands the request to loop().
# Depth of one: while a classification is in flight further trips are dropped
# rather than queued, since a stale result is worse than none.
_triggers = queue.Queue(maxsize=1)


def on_trigger(distance_mm=0):
    # The MCU sends how far the item was when it fired, which sizes the crop.
    try:
        _triggers.put_nowait(int(distance_mm))
    except queue.Full:
        pass
    return True


Bridge.provide("on_trigger", on_trigger)


def _save_capture(frames, label, category):
    """Keep one frame from the burst. The rest differ only by a few milliseconds
    and are not worth the disk."""
    if not frames:
        return None

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = CAPTURE_DIR / f"{stamp}_{label}_{category}.jpg"

    cv2.imwrite(str(path), frames[0])
    cv2.imwrite(str(CAPTURE_DIR / "latest.jpg"), frames[0])

    # The full frame with the crop window drawn, so the camera can be aimed and
    # the parallax offsets tuned by eye rather than by guesswork.
    overlay = vision.debug_overlay() if hasattr(vision, "debug_overlay") else None
    if overlay is not None:
        cv2.imwrite(str(CAPTURE_DIR / "latest_full.jpg"), overlay)

    return path


def _handle_trigger(distance_mm):
    # The backend owns the no-frame case, so a missing camera is its policy.
    frames = vision.capture(distance_mm=distance_mm)
    label, confidence = vision.classify(frames)
    category = _rules.resolve(label, confidence)

    saved = _save_capture(frames, label, category)
    _store.log(label, confidence, category)

    where = f"saved {saved.name}" if saved else "no frames captured"
    print(f"[bin] {len(frames)} frame(s): {label} ({confidence:.2f}) -> {category}, {where}")

    Bridge.call("set_feedback", category)


_announced = False


def loop():
    global _announced
    if not _announced:
        # The MCU boots long before this container, so it stays in degraded mode
        # until we announce ourselves.
        Bridge.notify("mpu_ready")
        _announced = True
        print("[bin] announced mpu_ready, waiting for triggers")

    try:
        distance_mm = _triggers.get(timeout=0.5)
    except queue.Empty:
        return

    _handle_trigger(distance_mm)


App.run(user_loop=loop)
