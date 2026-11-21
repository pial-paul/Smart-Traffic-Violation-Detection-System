"""Streaming fixed-camera analysis for constant-frame-rate video files."""

import hashlib
import math
import time
from collections import Counter, deque
from datetime import timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import pixels
from .detection import CocoDetector, MotionDetector, detect_frame
from .plates import consensus, read_plate
from .rules import Crossing, Signal
from .tracking import Tracker


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(frames, camera):
    panels = []
    for frame, box, label in frames:
        h, w = frame.shape[:2]
        scale = min(1, 800 / w)
        image = cv2.resize(frame, (round(w * scale), max(1, round(h * scale))))
        h, w = image.shape[:2]
        left, top, right, bottom = pixels(box, w, h)
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), (26, 193, 245), 2)
        pts = [(round(x * (w - 1)), round(y * (h - 1))) for x, y in camera.stop_line]
        cv2.line(image, *pts, (136, 235, 135), 2, cv2.LINE_AA)
        image = cv2.copyMakeBorder(image, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(24, 27, 24))
        cv2.putText(
            image,
            label,
            (9, h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(0.45, w / 1050),
            (248, 250, 248),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    ok, data = cv2.imencode(".jpg", np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError("Could not encode evidence")
    return data.tobytes()


def analyze(video, config, detector, start, on_event, ocr=True):
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("Recording start requires an explicit UTC offset")
    if detector not in {"motion", "coco"}:
        raise ValueError("Choose motion or coco")
    video = Path(video)
    if not video.is_file():
        raise ValueError("Input must be a local video file")
    begin = time.perf_counter()
    if ocr:
        import pytesseract

        if "eng" not in pytesseract.get_languages(config=""):
            raise ValueError("Install Tesseract English data or pass --no-ocr")
    model = MotionDetector() if detector == "motion" else CocoDetector()
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("Video must decode with a valid constant frame rate (up to 240 fps)")
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        gap = max(2, min(8, round(0.25 * fps)))
        tracker, signal, crossing = Tracker(gap), Signal(), Crossing(config.stop_line)
        history = deque(maxlen=gap + 1)
        counts = Counter()
        index = events = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            state = signal.update(frame, config.signal_roi, index)
            counts[state] += 1
            detections = detect_frame(model, frame, config.road_roi)
            observed = tracker.update(detections, index)
            history.append((index, frame, {t.id: t.det.box for t in observed}))
            crossing.forget(tracker.active)
            for track in observed:
                prev = crossing.observe(track, index, signal.red_since, gap, frame)
                if prev is None:
                    continue
                samples = [(i, img, boxes[track.id]) for i, img, boxes in history if track.id in boxes]
                plate = None
                if ocr:
                    plate = consensus(read_plate(img, box) for _, img, box in samples[-3:])
                occurred = (start + timedelta(seconds=index / fps)).astimezone(timezone.utc)
                event = dict(
                    kind="red_light_crossing",
                    frame=index,
                    track_id=track.id,
                    occurred_at=occurred.isoformat(timespec="milliseconds"),
                    camera=config.camera,
                    x=float(track.det.foot[0]),
                    y=float(track.det.foot[1]),
                    plate=plate,
                    detector=detector,
                    label=track.det.label,
                    signal=state,
                    approach_frame=prev.frame,
                )
                shots = []
                for tag, i, img, box in (
                    ("APPROACH", prev.frame, prev.image, prev.box),
                    ("CROSSING", index, frame, track.det.box),
                ):
                    shot_time = (start + timedelta(seconds=i / fps)).astimezone(timezone.utc)
                    label = (
                        f"{tag}   frame {i}   track {track.id}   {shot_time:%H:%M:%S.%f}"[:-3] + " UTC"
                    )
                    shots.append((img, box, label))
                on_event(event, evidence(shots, config))
                events += 1
            index += 1
        if index == 0 or (declared > 0 and index < declared):
            raise ValueError(f"Incomplete video decode ({index} of {declared} declared frames)")
        elapsed = time.perf_counter() - begin
        return dict(
            frames=index,
            events=events,
            fps=fps,
            seconds=elapsed,
            processing_fps=index / elapsed,
            signal_frames=dict(counts),
        )
    finally:
        cap.release()
