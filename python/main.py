import os
import queue
import time
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

# Order is a contract with sketch.ino: the MCU indexes straight into its STYLES
# table with what we send, so these must stay in the same order as that array.
# Anything unrecognised maps to the last entry, unknown.
_MCU_CATEGORIES = ("recycle", "compost", "trash", "hazardous", "ewaste", "unknown")


def CATEGORY_INDEX(category):
    try:
        return _MCU_CATEGORIES.index(category)
    except ValueError:
        return len(_MCU_CATEGORIES) - 1

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

# Set when the MCU reports the item has gone. The camera read happens in the
# loop rather than here, because V4L2 permits a single opener and reading from
# a Bridge handler thread races the trigger path for it.
_want_background = False


def scene_clear():
    global _want_background
    _want_background = True
    return True


Bridge.provide("scene_clear", scene_clear)


def host_log(line):
    # The MCU's own Monitor output only reaches App Lab's serial tab, which is
    # invisible from the Linux logs. This is the way back.
    print(line)
    return True


Bridge.provide("host_log", host_log)


def _log(line):
    # Stays in the Python console. Forwarding it to the serial monitor was
    # tried and removed: a String parameter never reaches an MCU handler.
    print(line)


if not USE_FAKE_VISION:
    from classification import classifier

    classifier.set_logger(_log)


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

    # The full frame with the detection box drawn on it. Without this there is
    # no way to tell a bad classification from a crop that missed the object.
    if getattr(vision, "last_frame", None) is not None:
        import detector

        annotated = detector.draw_box(
            vision.last_frame, vision.last_box, f"{label} {category}"
        )
        cv2.imwrite(str(CAPTURE_DIR / "latest_boxed.jpg"), annotated)

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

    # Send the category as an index, not as its name. A String parameter never
    # reached the MCU handler - every call timed out, so the sketch fell back to
    # its own timeout and showed unknown even when the classifier was certain.
    # Zero-argument handlers marshalled fine, which is why the mpu_ready
    # handshake worked and masked this for so long.
    #
    # Short timeout because the MCU gives up after 4s anyway, and the try
    # because an unreachable MCU must cost one result rather than the whole
    # Linux side.
    try:
        Bridge.call("set_feedback", CATEGORY_INDEX(category), timeout=2)
    except Exception as exc:
        print(f"[bin] could not reach the MCU ({exc})")


# Re-announced rather than sent once. The MCU boots long before this container
# and stays in degraded mode until it hears from us - but it also reboots on
# every sketch flash, and a one-shot announcement is lost when that happens
# after we have already spoken. The symptom is the bin insisting Linux is not
# up while Linux is running perfectly. Cheap to repeat, so repeat.
ANNOUNCE_EVERY_S = 5.0
_last_announce = 0.0


def loop():
    global _last_announce, _want_background

    now = time.monotonic()
    if now - _last_announce >= ANNOUNCE_EVERY_S:
        Bridge.notify("mpu_ready")
        if _last_announce == 0.0:
            print("[bin] announced mpu_ready, waiting for triggers")
        _last_announce = now

    try:
        distance_mm = _triggers.get(timeout=0.5)
    except queue.Empty:
        # Nothing to classify, so this is the safe moment to refresh the empty
        # scene the detector compares against.
        if _want_background:
            _want_background = False
            if vision.grab_background():
                print("[bin] background reference updated")
        return

    _handle_trigger(distance_mm)


App.run(user_loop=loop)
