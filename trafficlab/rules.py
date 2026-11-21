"""Conservative signal and directed finite-segment crossing rules."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .config import pixels


class Signal:
    def __init__(self, min_frames=3):
        self.min_frames = min_frames
        self.raw = None
        self.count = 0
        self.red_since = None

    def update(self, frame, roi, index):
        h, w = frame.shape[:2]
        left, top, right, bottom = pixels(roi, w, h)
        hue, sat, val = cv2.split(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV))
        visible = (sat >= 75) & (val >= 70)
        masks = {
            "red": (hue <= 10) | (hue >= 170),
            "green": (hue >= 36) & (hue <= 90),
            "amber": (hue >= 12) & (hue <= 35),
        }
        votes = sorted(((np.mean(mask & visible), name) for name, mask in masks.items()), reverse=True)
        raw = votes[0][1] if votes[0][0] >= 0.025 and votes[0][0] > 2 * votes[1][0] else "unknown"
        self.count = self.count + 1 if self.raw == raw else 1
        self.raw = raw
        state = raw if self.count >= self.min_frames else "unknown"
        if state != "red":
            self.red_since = None
        elif self.red_since is None:
            self.red_since = index
        return state


@dataclass
class Position:
    frame: int
    foot: tuple
    box: tuple
    last_seen: int
    image: object


class Crossing:
    def __init__(self, line, band=0.008):
        self.a, self.b = line
        self.dx, self.dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        self.length = math.hypot(self.dx, self.dy)
        self.band = band
        self.approach = {}
        self.emitted = set()

    def forget(self, ids):
        self.approach = {k: v for k, v in self.approach.items() if k in ids}
        self.emitted.intersection_update(ids)

    def observe(self, track, index, red_since, max_gap, image=None):
        point = track.det.foot

        def side(p):
            return (self.dx * (p[1] - self.a[1]) - self.dy * (p[0] - self.a[0])) / self.length

        distance = side(point)
        if track.id in self.emitted:
            return None
        prev = self.approach.get(track.id)
        if prev and index - prev.last_seen > max_gap:
            del self.approach[track.id]
            prev = None
        if distance < -self.band:
            self.approach[track.id] = Position(index, tuple(point), track.det.box, index, image)
            return None
        if distance <= self.band:
            if prev:
                prev.last_seen = index
            return None
        prev = self.approach.pop(track.id, None)
        if not prev or track.hits < 3 or red_since is None or prev.frame < red_since:
            return None
        fraction = -side(prev.foot) / (distance - side(prev.foot))
        x = prev.foot[0] + fraction * (point[0] - prev.foot[0])
        y = prev.foot[1] + fraction * (point[1] - prev.foot[1])
        along = ((x - self.a[0]) * self.dx + (y - self.a[1]) * self.dy) / (self.length**2)
        if 0 <= along <= 1:
            self.emitted.add(track.id)
            return prev
        return None
