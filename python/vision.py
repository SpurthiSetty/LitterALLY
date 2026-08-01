import glob
import os
import re

import cv2

UNKNOWN_LABEL = "unknown"

# A freshly opened USB webcam returns black frames until auto-exposure settles,
# and V4L2 hands back whatever is sitting in its buffer, which between triggers
# is stale. Both cost frames rather than accuracy, so discard generously.
_WARMUP_FRAMES = 10
_FLUSH_FRAMES = 5

# Frames per trigger. Override with SMARTBIN_BURST; the classifier combines them
# into one verdict.
BURST_FRAMES = int(os.environ.get("SMARTBIN_BURST", "3"))

_camera = None


def _candidate_indices():
    # Enumerate what actually exists rather than hardcoding: the webcam was
    # /dev/video2 one boot and /dev/video0 the next, so USB enumeration order
    # cannot be relied on. The other nodes are ISP/metadata devices that open
    # fine but never yield a frame, which the read test below filters out.
    found = sorted(
        int(match.group(1))
        for match in (re.search(r"(\d+)$", path) for path in glob.glob("/dev/video*"))
        if match
    )
    preferred = os.environ.get("SMARTBIN_CAMERA_INDEX")
    if preferred is not None:
        index = int(preferred)
        return [index] + [i for i in found if i != index]
    return found


def _ensure_camera():
    global _camera
    if _camera is not None and _camera.isOpened():
        return _camera

    for index in _candidate_indices():
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            ok, frame = candidate.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                for _ in range(_WARMUP_FRAMES):
                    candidate.read()
                print(f"[vision] webcam ready at index {index}, {width}x{height}")
                _camera = candidate
                return _camera
        candidate.release()

    print("[vision] no usable webcam found")
    return None


def capture(count=None):
    """Grab a short burst of frames and return them as a list.

    Several frames of the same item beat one, because a single frame can catch
    motion blur, a reflection, or an awkward angle. They are taken back to back
    so they are all of the same object, and classified together downstream.
    Returns [] when no camera is available.
    """
    if count is None:
        count = BURST_FRAMES

    camera = _ensure_camera()
    if camera is None:
        return []

    # Drop whatever the driver has buffered so the burst reflects now, not the
    # last time anyone looked.
    for _ in range(_FLUSH_FRAMES):
        camera.grab()

    frames = []
    for _ in range(max(1, count)):
        ok, frame = camera.read()
        if ok and frame is not None:
            frames.append(frame)

    return frames


def classify(frames):
    """Return (label, confidence) for a burst. The label -> category step is not ours.

    frames may be empty when the camera is missing or every read failed.
    """
    if not frames:
        return UNKNOWN_LABEL, 0.0

    try:
        from classification import classify_burst
    except Exception as exc:
        # A missing runtime or model must not take the bin down: the MCU shows
        # unknown and everything else keeps working.
        print(f"[vision] classifier unavailable ({exc})")
        return UNKNOWN_LABEL, 0.0

    try:
        return classify_burst(frames)
    except Exception as exc:
        print(f"[vision] classification failed ({exc})")
        return UNKNOWN_LABEL, 0.0
