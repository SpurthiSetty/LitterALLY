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
import numpy as np

# yolox - the COCO detector. Knows bottle, banana, cup, cell phone; knows
#         nothing about a battery, a sheet of cardboard or a piece of cloth.
# diff  - difference against a reference frame of the empty scene. Needs no
#         classes at all, which is the point: it finds whatever was not there
#         before. The default.
# off   - classify the whole frame.
#
# Defaults to off. Both alternatives were measured against real captures and
# neither earns its place yet. yolox finds the room - person, chair,
# refrigerator - and almost never the held item, because COCO has no class for
# a battery or a sheet of cardboard. diff finds the largest thing that moved,
# which is always the hand: one capture boxed the hand and wrapper while the
# battery sat sharp and centred just below the crop, and the hand duly
# classified as "biological". Cropping wrong is worse than not cropping, and
# uncropped the same battery scored 0.97.
MODE = os.environ.get("SMARTBIN_DETECT", "off").lower()

# How different a pixel must be to count as changed. Low enough to catch a dark
# battery against a dark counter, high enough to ignore sensor noise and the
# slow drift of daylight.
DIFF_THRESHOLD = int(os.environ.get("SMARTBIN_DIFF_THRESHOLD", "28"))

# Ignore changed regions smaller than this share of the frame - a few hundred
# noisy pixels are not an object.
DIFF_MIN_AREA = float(os.environ.get("SMARTBIN_DIFF_MIN_AREA", "0.004"))

# A changed region touching the frame edge is something reaching in - an arm, a
# sleeve, a shifted chair - not something being held out to be looked at. An
# item presented to a camera sits wholly inside the picture. This is what
# separates the battery from the hand above it: taking the largest region
# always chose the hand, which then classified as "biological" while the
# battery sat sharp and centred just below the crop.
BORDER_MARGIN = int(os.environ.get("SMARTBIN_DIFF_BORDER", "8"))

# A change covering most of the view is the light changing or the camera
# moving, not an object. Cropping to it is a no-op anyway.
DIFF_MAX_AREA = float(os.environ.get("SMARTBIN_DIFF_MAX_AREA", "0.80"))

_background = None

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


def set_background(frame):
    """Remember what the scene looks like with nothing in it.

    Called when the MCU reports the item has been withdrawn, which is the only
    moment we can be sure the view is empty. Stored blurred and in grey: the
    comparison cares about shapes appearing, not about sensor noise or a
    one-pixel shift.
    """
    global _background
    if frame is None:
        return
    _background = _prepare(frame)


def has_background():
    return _background is not None


# Skin in YCrCb. The chroma range is broad on purpose - it has to hold across
# skin tones, and a false positive only costs a smaller crop, while a false
# negative puts the hand back in the picture. Tan cardboard sits near this
# range, which is the known cost of the approach.
REMOVE_SKIN = os.environ.get("SMARTBIN_DIFF_SKIN", "1") != "0"
_SKIN_LOW = np.array([0, 133, 77], dtype=np.uint8)
_SKIN_HIGH = np.array([255, 180, 135], dtype=np.uint8)


def _skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, _SKIN_LOW, _SKIN_HIGH)
    # Grow it slightly so the rim of the hand goes too, rather than leaving a
    # halo that reconnects the hand to the item.
    return cv2.dilate(skin, np.ones((7, 7), np.uint8), iterations=2)


def _prepare(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (21, 21), 0)


def _find_by_difference(frame):
    """Box around whatever was not in the empty scene.

    This is the one approach that does not need to know what the object is,
    which is why it works for a battery when a COCO detector cannot: it is not
    recognising a battery, only noticing that something is there now.
    """
    if _background is None:
        return None

    current = _prepare(frame)
    if current.shape != _background.shape:
        return None

    delta = cv2.absdiff(_background, current)
    _, mask = cv2.threshold(delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Take the hand out of the change. Fingers touch the item, so after any
    # dilation the two are one connected region and no contour rule can tell
    # them apart - the largest region is always hand-plus-item, and the item is
    # the smaller half. Skin colour is the one property that separates them.
    if REMOVE_SKIN:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(_skin_mask(frame)))

    # Open to drop speckle, then dilate so an object broken into separate
    # bright and dark patches is bridged into one region. Kept modest: heavy
    # dilation re-merges whatever the skin mask just separated.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = frame.shape[:2]
    frame_area = float(width * height)

    interior = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < DIFF_MIN_AREA * frame_area or w * h > DIFF_MAX_AREA * frame_area:
            continue
        if (x <= BORDER_MARGIN or y <= BORDER_MARGIN
                or x + w >= width - BORDER_MARGIN
                or y + h >= height - BORDER_MARGIN):
            continue
        interior.append((w * h, (x, y, x + w, y + h)))

    if not interior:
        # Better to hand over the whole frame than to crop to the arm. A wrong
        # crop removes the subject entirely; no crop is merely the old
        # behaviour.
        print("[detector] nothing fully inside the frame, using it all")
        return None

    area, box = max(interior, key=lambda item: item[0])
    print(f"[detector] item {box[2] - box[0]}x{box[3] - box[1]} "
          f"({100.0 * area / frame_area:.0f}% of frame), "
          f"{len(contours)} changed regions, {len(interior)} inside")
    return _pad(list(box), width, height)


def find_item(frame):
    """Best guess at the presented item as (x1, y1, x2, y2), or None.

    None means "no opinion" - the caller should use the whole frame.
    """
    if MODE in ("0", "off", "none"):
        return None
    if MODE == "diff":
        return _find_by_difference(frame)
    return _find_yolox(frame)


def _find_yolox(frame):
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
