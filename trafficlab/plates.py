"""Localize light rectangular plates and read Latin alphanumerics on the CPU."""

import logging
import re
from collections import Counter

import cv2

from .config import pixels

LOG = logging.getLogger(__name__)


def read_plate(frame, box):
    import pytesseract

    h, w = frame.shape[:2]
    left, top, right, bottom = pixels(box, w, h)
    crop = frame[max(0, top) : min(bottom, h), max(left, 0) : min(w, right)]
    if not crop.size:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # CLAHE improves local contrast for unevenly illuminated plates.
    gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, pw, ph = cv2.boundingRect(contour)
        if pw >= 28 and ph >= 8 and 1.8 <= pw / ph <= 7.5 and 0.025 < pw * ph / gray.size < 0.65:
            candidates.append((pw * ph, x, y, pw, ph))
    for _, x, y, pw, ph in sorted(candidates, reverse=True)[:2]:
        plate = gray[y : y + ph, x : x + pw]
        plate = cv2.resize(plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plate = cv2.copyMakeBorder(plate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        try:
            result = pytesseract.image_to_data(
                plate,
                lang="eng",
                timeout=3,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                output_type=pytesseract.Output.DICT,
            )
        except RuntimeError:
            LOG.warning("OCR timed out or failed on a candidate crop")
            continue
        words = [
            (word, float(conf)) for word, conf in zip(result["text"], result["conf"]) if word.strip()
        ]
        value = "".join(word.strip() for word, _ in words)
        if re.fullmatch(r"[A-Z0-9]{4,12}", value) and min(conf for _, conf in words) >= 40:
            return value
    return None


def consensus(observations):
    readings = Counter(value for value in observations if value)
    if readings:
        value, count = readings.most_common(1)[0]
        if count >= 2 and count > sum(readings.values()) - count:
            return value
    return None
