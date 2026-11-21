"""Command-line entry points for offline ingestion and the operator console."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from urllib.error import URLError

from . import __version__

LOG = logging.getLogger("trafficlab")


def recording_time(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError
        return stamp.astimezone(timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Supply an ISO 8601 timestamp with a UTC offset") from exc


def ingest(args):
    from .config import Camera
    from .pipeline import analyze, file_digest
    from .store import Conflict, Store

    camera = Camera.load(args.config)
    store = Store(args.db)
    source = file_digest(args.video)
    provenance = dict(
        camera=asdict(camera),
        detector=args.detector,
        ocr=not args.no_ocr,
        start=args.start.isoformat(),
        version=__version__,
    )
    provenance = json.loads(json.dumps(provenance))

    def previous():
        entry = store.existing(camera.camera, source)
        if entry:
            run_id, original, summary = entry
            if original != provenance:
                raise ValueError(
                    "Recording already analyzed with different settings. Use a separate experiment DB."
                )
            return dict(run_id=run_id, reused=True, **summary)

    result = previous()
    if result:
        return result
    with tempfile.TemporaryDirectory(prefix="trafficlab-", dir=store.path.parent) as directory:
        path = Path(directory)

        def stage(event, jpeg):
            name = f"{event['frame']:012d}-{event['track_id']:08d}"
            (path / (name + ".jpg")).write_bytes(jpeg)
            (path / (name + ".json")).write_text(json.dumps(event), encoding="utf-8")

        summary = analyze(args.video, camera, args.detector, args.start, stage, not args.no_ocr)
        if file_digest(args.video) != source:
            raise ValueError("Recording changed during processing")
        records = (
            (json.loads(meta.read_text(encoding="utf-8")), meta.with_suffix(".jpg").read_bytes())
            for meta in sorted(path.glob("*.json"))
        )
        try:
            run_id = store.save(camera.camera, source, provenance, summary, records)
        except Conflict:
            return previous()
    return dict(run_id=run_id, reused=False, **summary)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trafficlab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    process = commands.add_parser("analyze", help="Analyze a constant-frame-rate video")
    process.add_argument("video", type=Path)
    process.add_argument("--config", type=Path, required=True)
    process.add_argument(
        "--start", type=recording_time, required=True, help="Recording start time with UTC offset"
    )
    process.add_argument("--detector", choices=["coco", "motion"], default="coco")
    process.add_argument("--no-ocr", action="store_true")
    process.add_argument("--db", type=Path, default=Path("instance/traffic.sqlite"))
    preview = commands.add_parser(
        "calibrate", help="Draw the road, signal and crossing direction on a video frame"
    )
    preview.add_argument("video", type=Path)
    preview.add_argument("--config", type=Path, required=True)
    preview.add_argument("--frame", type=int, default=0, help="Zero-based video frame")
    preview.add_argument("--out", type=Path, required=True, help="Output PNG")
    serve = commands.add_parser("serve", help="Run the review console with Waitress")
    serve.add_argument("--db", type=Path, default=Path("instance/traffic.sqlite"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--secure-cookie", action="store_true", help="Require HTTPS for session cookies")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        if args.command == "analyze":
            print(json.dumps(ingest(args), indent=2))
        elif args.command == "calibrate":
            from .calibration import preview_camera

            preview_camera(args.video, args.config, args.frame, args.out)
            print(args.out)
        else:
            from waitress import serve as run_server
            from .web import create_app

            app = create_app(
                args.db,
                os.environ.get("TRAFFIC_TOKEN"),
                os.environ.get("TRAFFIC_SECRET"),
                args.secure_cookie,
            )
            run_server(app, host=args.host, port=args.port)
    except (ValueError, OSError, RuntimeError, ImportError, sqlite3.Error, URLError) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
