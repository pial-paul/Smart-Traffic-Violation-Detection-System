from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import cv2

from trafficlab.__main__ import ingest, recording_time
from trafficlab.experiment import (
    CAMERA,
    FPS,
    FRAMES,
    HEIGHT,
    SCENES,
    WIDTH,
    background,
    dump_json,
    render_frame,
)
from trafficlab.store import Store


class IngestionTests(unittest.TestCase):
    def test_video_to_durable_case_and_safe_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            config, video = out / "camera.json", out / "scene.avi"
            dump_json(config, CAMERA)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
            self.assertTrue(writer.isOpened())
            try:
                base = background()
                for frame in range(FRAMES):
                    writer.write(render_frame(base, SCENES[0], frame))
            finally:
                writer.release()
            args = Namespace(
                config=config,
                video=video,
                db=out / "events.db",
                detector="motion",
                start=recording_time("2025-01-01T06:00:00+06:00"),
                no_ocr=True,
            )
            first = ingest(args)
            self.assertEqual(first["events"], 1)
            self.assertFalse(first["reused"])
            self.assertTrue(ingest(args)["reused"])
            store = Store(args.db)
            case = store.get(1)
            self.assertTrue(case["occurred_at"].startswith("2025-01-01T00:00:"))
            self.assertLess(case["approach_frame"], case["frame"])
            self.assertEqual(store.jpeg(1)[:2], b"\xff\xd8")
            args.start = recording_time("2025-01-02T00:00:00Z")
            with self.assertRaisesRegex(ValueError, "different settings"):
                ingest(args)
