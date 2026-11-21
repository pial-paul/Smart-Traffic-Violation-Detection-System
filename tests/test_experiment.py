import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from trafficlab.experiment import (
    CAMERA,
    SCENES,
    annotations,
    background,
    render_frame,
    score_events,
    sha256,
    validate_scene,
)


def event(x, frame=10, plate="LAB317"):
    return {"frame": frame, "x": x, "y": 0.6, "plate": plate}


class ExperimentTests(unittest.TestCase):
    def test_partial_and_duplicate_predictions(self):
        score = score_events([event(0.3), event(0.7)], [event(0.3), event(0.3, plate=None)])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 1))

    def test_match_does_not_require_readable_plate(self):
        score = score_events([event(0.3)], [event(0.3, plate=None)])
        self.assertEqual((score["tp"], score["plate_exact"], score["plate_read"]), (1, 0, 0))

    def test_missing_ground_truth_plate_excluded_from_ocr_denominator(self):
        score = score_events([event(0.3, plate=None)], [event(0.3, plate=None)])
        self.assertEqual((score["tp"], score["plate_eligible"], score["plate_exact"]), (1, 0, 0))

    def test_existing_inputs_require_matching_bytes_labels_and_timezone(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "input.avi"
            video.write_bytes(b"fixture")
            scene = {
                "frames": 30,
                "expected_events": [event(0.3)],
                "recording_start": "2025-01-01T00:00:00Z",
                "file": video.name,
                "sha256": sha256(video),
            }
            self.assertEqual(validate_scene(scene, root)[0], video)
            for change in (
                {"recording_start": "2025-01-01"},
                {"sha256": "stale"},
                {"frames": 5},
                {"expected_events": [event(float("nan"))]},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_scene({**scene, **change}, root)

    def test_maximum_cardinality_before_error(self):
        # A greedy nearest match for the first truth would strand the second truth.
        score = score_events([event(0.20), event(0.26)], [event(0.22), event(0.14)])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (2, 0, 0))

    def test_independent_space_and_time_limits(self):
        for prediction in (event(0.3, frame=15), event(0.38, frame=10)):
            with self.subTest(prediction=prediction):
                score = score_events([event(0.3)], [prediction])
                self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 1, 1))

    def test_empty_predictions_and_empty_ground_truth(self):
        for truth, observed, expected in [
            ([], [], (0, 0)),
            ([event(0.3)], [], (0, 1)),
            ([], [event(0.3)], (1, 0)),
        ]:
            score = score_events(truth, observed)
            self.assertEqual((score["fp"], score["fn"]), expected)

    def test_render_and_annotations(self):
        self.assertEqual([len(annotations(scene)[0]) for scene in SCENES], [1, 2, 1, 1, 0, 0, 0, 1])
        scene = next(s for s in SCENES if s.dim)
        base = background()
        self.assertTrue(np.array_equal(render_frame(base, scene, 8), render_frame(base, scene, 8)))
        self.assertFalse(np.array_equal(render_frame(base, scene, 8), render_frame(base, scene, 9)))

    def test_sample_config_matches_generated_camera(self):
        path = Path(__file__).resolve().parents[1] / "demos" / "camera.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), CAMERA)
