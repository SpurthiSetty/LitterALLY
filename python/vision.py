import glob
import os
import re

import cv2

UNKNOWN_LABEL = "unknown"

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
                print(f"[vision] webcam ready at index {index}, {width}x{height}")
                _camera = candidate
                return _camera
        candidate.release()

    print("[vision] no usable webcam found")
    return None


def release():
    # V4L2 allows a single opener, so the handle has to be freed explicitly
    # rather than left to garbage collection.
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


def capture():
    camera = _ensure_camera()
    if camera is None:
        return None
    ok, frame = camera.read()
    return frame if ok else None


def classify(frame):
    # Task 1 seam: replace with the real classifier. Returning a zero-confidence
    # label keeps the trigger path exercisable end to end until then, because the
    # rules engine resolves anything under the confidence floor to "unknown".
    #
    # frame may be None when the camera is missing or a read failed, so the real
    # classifier has to handle that case rather than assume an array.
    if frame is None:
        return UNKNOWN_LABEL, 0.0
    return UNKNOWN_LABEL, 0.0
