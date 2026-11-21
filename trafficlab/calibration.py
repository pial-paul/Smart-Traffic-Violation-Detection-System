"""Inspect the exact image geometry used by the analyzer before importing a clip."""

from pathlib import Path
import math

import cv2

from .config import Camera, pixels


def preview_camera(video, config, index, out):
    camera = Camera.load(config)
    if index < 0 or not Path(video).is_file():
        raise ValueError("Supply a local recording and a nonnegative frame number")
    if out.suffix.lower() != ".png":
        raise ValueError("Choose a .png output path")
    cap = cv2.VideoCapture(str(video))
    try:
        if index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = cap.read()
        if not ok:
            raise ValueError("Requested frame could not be decoded")
    finally:
        cap.release()
    h, w = image.shape[:2]
    for label, roi, color in (
        ("ROAD", camera.road_roi, (240, 184, 78)),
        ("SIGNAL", camera.signal_roi, (90, 217, 240)),
    ):
        left, top, right, bottom = pixels(roi, w, h)
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), color, 2)
        cv2.putText(
            image,
            label,
            (left + 4, max(18, top + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    (ax, ay), (bx, by) = camera.stop_line

    def point(x, y):
        return round(x * (w - 1)), round(y * (h - 1))

    cv2.line(image, point(ax, ay), point(bx, by), (113, 243, 145), 3, cv2.LINE_AA)
    middle_x, middle_y = (ax + bx) / 2, (ay + by) / 2
    length = math.hypot(bx - ax, by - ay)
    nx, ny = (ay - by) / length, (bx - ax) / length
    cv2.arrowedLine(
        image,
        point(middle_x - 0.07 * nx, middle_y - 0.07 * ny),
        point(middle_x + 0.07 * nx, middle_y + 0.07 * ny),
        (113, 243, 145),
        3,
        cv2.LINE_AA,
        tipLength=0.3,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), image):
        raise OSError("Could not save calibration preview")
