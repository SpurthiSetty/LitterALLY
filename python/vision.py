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

# Distance-proportional crop. Apparent size goes as 1/distance, so an item at
# the far edge of the window covers roughly an eighth of the width it does at
# the near edge. CROP_REF_MM is the distance at which an item is taken to fill
# the frame; beyond it the crop tightens by the same ratio.
#
# The floor exists because the arithmetic alone would crop to ~12% at 250 mm,
# which is 77x58 pixels upscaled to 224x224 - mostly interpolation artefacts.
CROP_REF_MM = float(os.environ.get("SMARTBIN_CROP_REF_MM", "30"))
CROP_MIN_FRACTION = float(os.environ.get("SMARTBIN_CROP_MIN", "0.25"))

# Parallax. The distance sensor and the webcam sit at different points on the
# board, so an item on the sensor's axis lands off-axis in the camera, and the
# offset grows as the item comes closer - it goes as 1/distance, same as
# apparent size. These are the offsets at CROP_REF_MM, as a fraction of frame
# width and height; positive x is right, positive y is down. Tune them against
# the debug overlay, which draws the crop window on the full frame.
CROP_SHIFT_X = float(os.environ.get("SMARTBIN_CROP_SHIFT_X", "0"))
CROP_SHIFT_Y = float(os.environ.get("SMARTBIN_CROP_SHIFT_Y", "0"))

_camera = None
_last_full = None
_last_box = None


def crop_for_distance(frame, distance_mm):
    """Crop a square that tracks where the item should appear, and how big.

    Square rather than rectangular on purpose: the model wants 224x224, and
    resizing a 4:3 frame straight to square stretches everything by a third.
    """
    global _last_full, _last_box

    height, width = frame.shape[:2]

    if not distance_mm or distance_mm <= 0:
        fraction = 1.0
        parallax = 1.0
    else:
        ratio = CROP_REF_MM / float(distance_mm)
        fraction = max(CROP_MIN_FRACTION, min(1.0, ratio))
        # Not clamped to the crop floor: the offset is a separate physical
        # effect and keeps shrinking after the crop size has bottomed out.
        parallax = min(1.0, ratio)

    side = max(32, int(min(height, width) * fraction))

    centre_x = width / 2.0 + CROP_SHIFT_X * width * parallax
    centre_y = height / 2.0 + CROP_SHIFT_Y * height * parallax

    left = int(round(centre_x - side / 2.0))
    top = int(round(centre_y - side / 2.0))
    left = max(0, min(width - side, left))
    top = max(0, min(height - side, top))

    _last_full = frame
    _last_box = (left, top, side)

    return frame[top:top + side, left:left + side]


def debug_overlay():
    """The full frame with the crop window drawn on it, for aiming the camera."""
    if _last_full is None or _last_box is None:
        return None

    annotated = _last_full.copy()
    left, top, side = _last_box
    cv2.rectangle(annotated, (left, top), (left + side, top + side), (0, 255, 0), 3)
    return annotated


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


def capture(count=None, distance_mm=None):
    """Grab a short burst of frames, cropped for the reported distance.

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
            frames.append(crop_for_distance(frame, distance_mm))

    if frames:
        side = frames[0].shape[0]
        print(f"[vision] {len(frames)} frame(s) at {distance_mm} mm, cropped to {side}x{side}")

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
