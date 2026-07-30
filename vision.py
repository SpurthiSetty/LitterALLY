import cv2

UNKNOWN_LABEL = "unknown"

# The USB webcam lands on /dev/video2 on this board; the rest are fallbacks.
_CAMERA_INDICES = (2, 0, 1, 4)

_camera = None


def _ensure_camera():
    global _camera
    if _camera is not None and _camera.isOpened():
        return _camera

    for index in _CAMERA_INDICES:
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            ok, frame = candidate.read()
            if ok and frame is not None:
                print(f"[vision] USB webcam ready at index {index}")
                _camera = candidate
                return _camera
        candidate.release()

    print("[vision] no USB webcam found")
    return None


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
    return UNKNOWN_LABEL, 0.0
