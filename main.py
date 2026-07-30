import os
import queue

from arduino.app_utils import App, Bridge

from rules import Rules
from store import EventStore

if os.environ.get("SMARTBIN_FAKE_VISION"):
    import fake_vision as vision

    print("[bin] SMARTBIN_FAKE_VISION set, using the stand-in classifier")
else:
    import vision

_rules = Rules()
_store = EventStore()

# on_trigger has to return instantly, so it only hands the request to loop().
# Depth of one: while a classification is in flight further trips are dropped
# rather than queued, since a stale result is worse than none.
_triggers = queue.Queue(maxsize=1)


def on_trigger():
    try:
        _triggers.put_nowait(True)
    except queue.Full:
        pass
    return True


Bridge.provide("on_trigger", on_trigger)


def _handle_trigger():
    # The backend owns the no-frame case, so a missing camera is its policy.
    label, confidence = vision.classify(vision.capture())
    category = _rules.resolve(label, confidence)

    _store.log(label, confidence, category)
    print(f"[bin] {label} ({confidence:.2f}) -> {category}")

    Bridge.call("set_feedback", category)


_announced = False


def loop():
    global _announced
    if not _announced:
        # The MCU boots long before this container, so it stays in degraded mode
        # until we announce ourselves.
        Bridge.notify("mpu_ready")
        _announced = True

    try:
        _triggers.get(timeout=0.5)
    except queue.Empty:
        return

    _handle_trigger()


App.run(user_loop=loop)
