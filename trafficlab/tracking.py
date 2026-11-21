"""Global one-to-one association with a damped constant-velocity predictor."""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    det: object
    last: int
    hits: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


class Tracker:
    def __init__(self, max_age=5):
        self.active = {}
        self.next_id = 1
        self.max_age = max_age

    def update(self, detections, index):
        self.active = {k: v for k, v in self.active.items() if index - v.last <= self.max_age + 1}
        tracks = list(self.active.values())
        used = set()
        if tracks and detections:
            cost = np.full((len(tracks), len(detections)), 1000.0)
            for i, track in enumerate(tracks):
                gap = index - track.last
                expected = track.det.foot + track.velocity * gap
                old_w = track.det.box[2] - track.det.box[0]
                for j, det in enumerate(detections):
                    dist = np.linalg.norm(det.foot - expected)
                    width_ratio = (det.box[2] - det.box[0]) / max(old_w, 1e-6)
                    if dist < min(0.16, 0.05 + gap * 0.015) and 0.48 < width_ratio < 2.1:
                        cost[i, j] = dist + 0.025 * abs(np.log(width_ratio))
            rr, cc = linear_sum_assignment(cost)
            for i, j in zip(rr, cc):
                if cost[i, j] >= 1000:
                    continue
                track = tracks[i]
                det = detections[j]
                velocity = (det.foot - track.det.foot) / (index - track.last)
                track.velocity = 0.6 * velocity + 0.4 * track.velocity
                track.det, track.last = det, index
                track.hits += 1
                used.add(j)
        for j, det in enumerate(detections):
            if j not in used:
                self.active[self.next_id] = Track(self.next_id, det, index)
                self.next_id += 1
        return [track for track in self.active.values() if track.last == index]
