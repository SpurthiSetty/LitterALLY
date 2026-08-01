"""Find the presented item, so the classifier sees it rather than the room.

A classifier labels the whole frame and pools features across all of it, so an
AA battery covering 7% of the picture lost to the white counter behind it and
came back "paper" at 0.76. The same battery filling a phone screen scored 0.99.
Cropping to the item is what closes that gap.

YOLOX is a generic COCO detector, which cuts both ways. It knows bottle, banana,
apple, cup, book, cell phone and laptop - much of what a bin sees - but has no
class for a battery or a sheet of cardboard. When it finds nothing usable the
caller falls back to the whole frame, which is no worse than before.
"""

import os

import cv2

# COCO classes that are the room rather than something being held out. Without
# this the crop lands on the fridge: on a real capture the two detections were
# "refrigerator" at 51% and "person" at 43%, and neither is the item.
_SCENERY = {
    "person", "refrigerator", "oven", "microwave", "sink", "toaster",
    "chair", "couch", "bed", "dining table", "toilet", "tv", "clock",
    "potted plant", "vase",
}

# Detections below this are ignored. Lower than the classifier's floor on
# purpose: a weak box that happens to be right still improves the crop, and a
# wrong one only costs us the fallback we would have taken anyway.
MIN_CONFIDENCE = float(os.environ.get("SMARTBIN_DETECT_CONF", "0.20"))

# Grown around the box before cropping. Detectors clip tight to the object and
# classifiers were trained on images with some margin around the subject.
BOX_PADDING = float(os.environ.get("SMARTBIN_DETECT_PAD", "0.15"))

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from arduino.app_bricks.object_detection import ObjectDetection

        _detector = ObjectDetection(confidence=MIN_CONFIDENCE)
    return _detector


def find_item(frame):
    """Best guess at the presented item as (x1, y1, x2, y2), or None.

    None means "no opinion" - the caller should use the whole frame.
    """
    try:
        detector = _get_detector()
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        results = detector.detect(buffer.tobytes(), image_type="jpg")
    except Exception as exc:
        print(f"[detector] unavailable ({exc})")
        return None

    candidates = []
    for entry in (results or {}).get("detection", []):
        name = str(entry.get("class_name", "")).lower()
        if name in _SCENERY:
            continue
        box = entry.get("bounding_box_xyxy")
        if not box or len(box) != 4:
            continue
        # Confidence arrives as a percentage in a string, e.g. "51.59".
        try:
            score = float(entry.get("confidence", 0)) / 100.0
        except (TypeError, ValueError):
            score = 0.0
        candidates.append((score, name, [float(v) for v in box]))

    if not candidates:
        return None

    score, name, box = max(candidates, key=lambda c: c[0])
    print(f"[detector] {name} {score:.2f}")
    return _pad(box, frame.shape[1], frame.shape[0])


def _pad(box, width, height):
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * BOX_PADDING
    dy = (y2 - y1) * BOX_PADDING
    x1 = int(max(0, x1 - dx))
    y1 = int(max(0, y1 - dy))
    x2 = int(min(width, x2 + dx))
    y2 = int(min(height, y2 + dy))

    # A degenerate box is worse than none: cropping to it would hand the
    # classifier a few pixels of nothing.
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return (x1, y1, x2, y2)


def crop_to(frame, box):
    if box is None:
        return frame
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def draw_box(frame, box, label=""):
    """Full frame with the box drawn, so what was cropped is inspectable."""
    annotated = frame.copy()
    if box is None:
        cv2.putText(annotated, "no detection - full frame", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        return annotated

    x1, y1, x2, y2 = box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if label:
        cv2.putText(annotated, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated
