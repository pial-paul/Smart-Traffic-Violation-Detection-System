import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from trafficlab.config import Camera
from trafficlab.detection import Detection
from trafficlab.plates import consensus
from trafficlab.rules import Crossing, Signal
from trafficlab.tracking import Track, Tracker


def vehicle(x, y):
    return Detection((x - 0.05, y - 0.12, x + 0.05, y), "car", 0.9)


class VisionTests(unittest.TestCase):
    def test_red_must_stabilize_and_unknown_clears_it_immediately(self):
        signal = Signal()
        red = np.full((10, 10, 3), (0, 0, 170), dtype=np.uint8)
        roi = (0, 0, 1, 1)
        self.assertEqual([signal.update(red, roi, i) for i in range(3)], ["unknown", "unknown", "red"])
        self.assertEqual(signal.red_since, 2)
        mixed = red.copy()
        mixed[5:] = (0, 180, 0)
        self.assertEqual(signal.update(mixed, roi, 3), "unknown")
        self.assertIsNone(signal.red_since)
        self.assertEqual(signal.update(red // 4, roi, 4), "unknown")

    def cross(self, observations, x=0.5):
        rule = Crossing(((0.2, 0.5), (0.8, 0.5)))
        hits = []
        for idx, y, since in observations:
            result = rule.observe(Track(1, vehicle(x, y), idx, hits=4), idx, since, 3)
            if result:
                hits.append(idx)
        return hits

    def test_cross_once_and_respect_finite_endpoints(self):
        path = [(1, 0.4, 0), (2, 0.55, 0), (3, 0.4, 0), (4, 0.55, 0)]
        self.assertEqual(self.cross(path), [2])
        self.assertEqual(self.cross(path, x=0.82), [])

    def test_slow_observed_transition_remains_eligible(self):
        path = [(1, 0.48, 0)] + [(i, 0.5, 0) for i in range(2, 20)] + [(20, 0.52, 0)]
        self.assertEqual(self.cross(path), [20])

    def test_direction_gap_red_onset_and_jitter(self):
        for path in (
            [(1, 0.55, 0), (2, 0.45, 0)],
            [(1, 0.45, 0), (9, 0.55, 0)],
            [(1, 0.45, None), (2, 0.55, 2)],
            [(1, 0.49, 0), (2, 0.505, 0)],
        ):
            self.assertEqual(self.cross(path), [])

    def test_assignment_survives_order_changes_and_short_occlusion(self):
        tracker = Tracker(max_age=2)
        first = tracker.update([vehicle(0.3, 0.3), vehicle(0.7, 0.3)], 0)
        second = tracker.update([vehicle(0.69, 0.32), vehicle(0.31, 0.32)], 1)
        self.assertEqual([t.id for t in first], [1, 2])
        self.assertLess(second[0].det.foot[0], second[1].det.foot[0])
        tracker.update([], 2)
        self.assertEqual(tracker.update([vehicle(0.33, 0.35)], 3)[0].id, 1)
        self.assertGreater(tracker.update([vehicle(0.33, 0.35)], 8)[0].id, 2)

    def test_plate_read_requires_agreement(self):
        self.assertIsNone(consensus(["ABC123", None, "ABC128"]))
        self.assertEqual(consensus(["ABC123", "ABC123", None]), "ABC123")

    def test_invalid_calibration(self):
        base = dict(
            camera="lab",
            signal_roi=[0, 0, 0.2, 0.2],
            road_roi=[0, 0.2, 1, 1],
            stop_line=[[0, 0.5], [1, 0.5]],
        )
        with TemporaryDirectory() as d:
            path = Path(d) / "camera.json"
            for fields in (
                {"camera": True},
                {"road_roi": [0, 0, 0, 1]},
                {"stop_line": [[0, 0.5], [0, 0.5]]},
                {"signal_roi": [0, 0, float("nan"), 1]},
            ):
                path.write_text(json.dumps({**base, **fields}))
                with self.assertRaises(ValueError):
                    Camera.load(path)
