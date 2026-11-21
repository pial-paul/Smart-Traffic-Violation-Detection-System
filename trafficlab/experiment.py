"""Generate repeatable fixed-camera footage and score red-light events end to end."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


WIDTH, HEIGHT, FPS, FRAMES = 768, 432, 12, 144
LINE_Y = 250
SEED = 4815
START = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
FRAME_TOLERANCE = 4
POSITION_TOLERANCE = 0.07
CAMERA = {
    "camera": "junction-lab",
    "road_roi": [0.175, 0.0, 0.84, 1.0],
    "signal_roi": [0.056, 0.112, 0.127, 0.235],
    "stop_line": [[0.18, LINE_Y / HEIGHT], [0.835, LINE_Y / HEIGHT]],
}


@dataclass(frozen=True)
class Car:
    plate: str
    x: int
    spawn: int
    speed: int = 5
    stop: bool = False
    color: tuple = (157, 103, 42)  # BGR

    def bottom(self, frame):
        if frame < self.spawn:
            return None
        if self.speed < 0:
            return HEIGHT + 75 + (frame - self.spawn) * self.speed
        y = (frame - self.spawn) * self.speed
        return min(y, LINE_Y - 23) if self.stop else y


@dataclass(frozen=True)
class Scene:
    name: str
    condition: str
    cars: tuple
    phases: tuple = ((0, "red"),)
    dim: bool = False
    obstruction: tuple = ()

    def light(self, frame):
        return next(state for first, state in reversed(self.phases) if first <= frame)


SCENES = (
    Scene(
        "daylight",
        "Two lanes. First car crosses red and second crosses green.",
        (Car("LAB317", 270, 20), Car("KIT429", 515, 56, color=(53, 94, 167))),
        ((0, "red"), (93, "green")),
    ),
    Scene(
        "parallel",
        "Two simultaneous forward red crossings in different lanes.",
        (Car("LAB317", 270, 22), Car("KIT429", 515, 22, color=(53, 94, 167))),
    ),
    Scene(
        "night",
        "Dim illumination, sensor noise and one red crossing.",
        (Car("LAB317", 270, 24),),
        dim=True,
    ),
    Scene(
        "occlusion",
        "A fixed rectangular occluder partially covers the car before crossing.",
        (Car("KIT429", 515, 22, color=(53, 94, 167)),),
        obstruction=(467, 123, 525, 169),
    ),
    Scene(
        "stop-reverse",
        "One car stops upstream. Another moves upstream over the line.",
        (Car("LAB317", 270, 22, stop=True), Car("KIT429", 515, 25, speed=-5)),
    ),
    Scene(
        "ambiguous",
        "A conflicting red and green lamp is visible during crossing.",
        (Car("LAB317", 270, 24),),
        ((0, "conflict"),),
    ),
    Scene("amber", "Car crosses while the signal is amber.", (Car("KIT429", 515, 23),), ((0, "amber"),)),
    Scene(
        "hidden-crossing",
        "An overhead structure completely hides the vehicle at the line.",
        (Car("KIT429", 515, 22, color=(53, 94, 167)),),
        obstruction=(441, 134, 589, 295),
    ),
)


def dump_json(path, data):
    # Fixed LF bytes keep annotation digests portable across operating systems.
    path.write_bytes((json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def background():
    rng = np.random.default_rng(SEED)
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = (51, 64, 63)
    for x in (21, 101, 679, 746):
        for y in range(-18, HEIGHT, 62):
            cv2.circle(image, (x, y), 20, (46, 71, 56), -1, cv2.LINE_AA)
            cv2.circle(image, (x - 3, y - 3), 14, (51, 83, 63), -1, cv2.LINE_AA)
    image[:, 133:646] = (60, 60, 57)
    image[:, 133:140] = image[:, 639:646] = (145, 148, 141)
    for y in range(-6, HEIGHT, 63):
        cv2.rectangle(image, (385, y), (390, y + 26), (172, 177, 167), -1)
    cv2.line(image, (139, LINE_Y), (639, LINE_Y), (210, 213, 203), 4)
    grain = rng.integers(-5, 6, size=(HEIGHT, WIDTH, 1), dtype=np.int16)
    return np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def draw_car(image, car, bottom):
    x, y = car.x, bottom
    hull = np.array(
        [
            (x - 51, y - 10),
            (x - 51, y - 60),
            (x - 40, y - 70),
            (x + 40, y - 70),
            (x + 51, y - 60),
            (x + 51, y - 10),
            (x + 42, y),
            (x - 42, y),
        ],
        np.int32,
    )
    cv2.fillConvexPoly(image, hull, car.color, cv2.LINE_AA)
    glass = np.array([(x - 38, y - 63), (x + 38, y - 63), (x + 32, y - 45), (x - 32, y - 45)], np.int32)
    cv2.fillConvexPoly(image, glass, (50, 54, 56), cv2.LINE_AA)
    cv2.line(image, (x - 31, y - 59), (x + 29, y - 59), (99, 101, 96), 2)
    for offset in (-42, 39):
        cv2.rectangle(image, (x + offset, y - 23), (x + offset + 4, y - 15), (70, 72, 231), -1)
    cv2.rectangle(image, (x - 36, y - 28), (x + 36, y - 7), (233, 239, 232), -1)
    size = cv2.getTextSize(car.plate, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    cv2.putText(
        image,
        car.plate,
        (x - size[0] // 2, y - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (12, 19, 21),
        1,
        cv2.LINE_AA,
    )


def render_frame(base, scene, frame):
    image = base.copy()
    for car in scene.cars:
        bottom = car.bottom(frame)
        if bottom is not None and 0 <= bottom <= HEIGHT + 70:
            draw_car(image, car, bottom)
    if scene.obstruction:
        left, top, right, bottom = scene.obstruction
        cv2.rectangle(image, (left, top), (right, bottom), (77, 78, 74), -1)
        cv2.line(image, (left + 1, top + 12), (right - 1, top + 12), (149, 148, 138), 2)
    cv2.rectangle(image, (37, 39), (102, 112), (29, 31, 30), -1)
    state = scene.light(frame)
    colors = {"red": (30, 32, 252), "green": (77, 240, 38), "amber": (18, 185, 255)}
    if state == "conflict":
        cv2.circle(image, (57, 74), 11, colors["red"], -1, cv2.LINE_AA)
        cv2.circle(image, (84, 74), 11, colors["green"], -1, cv2.LINE_AA)
    else:
        cv2.circle(image, (70, 74), 20, colors[state], -1, cv2.LINE_AA)
    if scene.dim:
        rng = np.random.default_rng(SEED + frame)
        noise = rng.normal(0, 2.4, size=(HEIGHT, WIDTH, 1)).astype(np.float32)
        image = np.clip(image * 0.46 + noise, 0, 255).astype(np.uint8)
    return image


def annotations(scene):
    """Ground truth comes from configured trajectories before rasterization."""
    events, frames = [], []
    for frame in range(FRAMES):
        visible = []
        for index, car in enumerate(scene.cars, start=1):
            y = car.bottom(frame)
            if y is None or not (0 <= y < HEIGHT + 70):
                continue
            visible.append(
                {
                    "id": index,
                    "plate": car.plate,
                    "bbox_xyxy": [car.x - 51, max(0, y - 70), car.x + 52, min(HEIGHT, y + 1)],
                }
            )
            prev = car.bottom(frame - 1)
            if prev is not None and prev < LINE_Y <= y and scene.light(frame) == "red":
                events.append(
                    {
                        "frame": frame,
                        "x": car.x / WIDTH,
                        "y": y / HEIGHT,
                        "plate": car.plate,
                        "vehicle_id": index,
                    }
                )
        frames.append({"frame": frame, "signal": scene.light(frame), "vehicles": visible})
    return events, frames


def generate_dataset(out: Path):
    out = Path(out)
    clips = out / "clips"
    labels = out / "labels"
    clips.mkdir(parents=True, exist_ok=True)
    labels.mkdir(exist_ok=True)
    dump_json(out / "camera.json", CAMERA)
    manifest = {
        "version": 1,
        "seed": SEED,
        "generated": True,
        "fps": FPS,
        "provenance": "Procedurally generated junction-lab v1; fixed trajectories and seed.",
        "camera_sha256": sha256(out / "camera.json"),
        "size": [WIDTH, HEIGHT],
        "codec": "MJPG",
        "scenes": [],
    }
    base = background()
    for scene in SCENES:
        video = clips / (scene.name + ".avi")
        temp = clips / (scene.name + ".writing.avi")
        writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the Motion JPEG writer")
        try:
            for frame in range(FRAMES):
                writer.write(render_frame(base, scene, frame))
        finally:
            writer.release()
        temp.replace(video)
        expected, frames = annotations(scene)
        label = labels / (scene.name + ".json")
        dump_json(label, frames)
        manifest["scenes"].append(
            {
                "name": scene.name,
                "condition": scene.condition,
                "recording_start": START.isoformat(),
                "file": video.relative_to(out).as_posix(),
                "sha256": sha256(video),
                "annotations": label.relative_to(out).as_posix(),
                "annotation_sha256": sha256(label),
                "frames": FRAMES,
                "expected_events": expected,
            }
        )
    dump_json(out / "manifest.json", manifest)
    return manifest


def score_events(
    expected, observed, frame_tolerance=FRAME_TOLERANCE, position_tolerance=POSITION_TOLERANCE
):
    """Maximize valid matches first, then minimize normalized time and space error."""
    if frame_tolerance <= 0 or position_tolerance <= 0:
        raise ValueError("Tolerances must be positive")
    penalty = 2 * min(len(expected), len(observed)) + 1.0
    costs = np.full((len(expected), len(observed)), penalty)
    for i, truth in enumerate(expected):
        for j, prediction in enumerate(observed):
            dt = abs(truth["frame"] - prediction["frame"])
            dist = math.hypot(truth["x"] - prediction["x"], truth["y"] - prediction["y"])
            if dt <= frame_tolerance and dist <= position_tolerance:
                costs[i, j] = dt / frame_tolerance + dist / position_tolerance
    rows, cols = linear_sum_assignment(costs)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if costs[i, j] < penalty]
    labeled = [(i, j) for i, j in pairs if expected[i].get("plate")]
    return {
        "tp": len(pairs),
        "fp": len(observed) - len(pairs),
        "fn": len(expected) - len(pairs),
        "plate_eligible": len(labeled),
        "plate_read": sum(bool(observed[j].get("plate")) for _, j in labeled),
        "plate_exact": sum(observed[j].get("plate") == expected[i]["plate"] for i, j in labeled),
        "matches": [{"expected": i, "observed": j} for i, j in pairs],
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def write_preview(video, path):
    """Export the actual generated clip with elapsed media time for the README."""
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    frames = []
    try:
        for index in range(FRAMES):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Incomplete preview input at frame {index}")
            if index % 3:
                continue
            frame = cv2.resize(frame, (576, 324), interpolation=cv2.INTER_AREA)
            banner = np.full((38, 576, 3), (35, 41, 38), dtype=np.uint8)
            cv2.putText(
                banner,
                "JUNCTION LAB   /   SYNTHETIC INPUT",
                (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 238, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                banner,
                f"{index / FPS:05.2f} s",
                (490, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (184, 205, 191),
                1,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(np.vstack([banner, frame]), cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).quantize(colors=96, method=Image.Quantize.FASTOCTREE))
    finally:
        cap.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=round(3000 / FPS), loop=0, optimize=True
    )


def verified_file(root, name, digest):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != digest:
        raise ValueError(
            f"Dataset file missing, outside dataset directory or checksum mismatch: {name!r}"
        )
    return path


def validate_scene(scene, out):
    frames = scene["frames"]
    if type(frames) is not int or frames <= 0:
        raise ValueError("Expected positive integer frame count")
    start = datetime.fromisoformat(scene["recording_start"].replace("Z", "+00:00"))
    if start.utcoffset() is None:
        raise ValueError("Dataset recording_start needs a UTC offset")
    for event in scene["expected_events"]:
        if type(event["frame"]) is not int or not 0 <= event["frame"] < frames:
            raise ValueError("Expected event with valid zero-based frame number")
        if any(type(event[k]) not in (int, float) or not 0 <= event[k] <= 1 for k in ("x", "y")):
            raise ValueError("Event position must use normalized x and y")
        if event.get("plate") is not None and not isinstance(event["plate"], str):
            raise ValueError("Reference plate must be text or null")
    if "annotations" in scene:
        labels = verified_file(out, scene["annotations"], scene["annotation_sha256"])
        data = json.loads(labels.read_text(encoding="utf-8"))
        if len(data) != frames or any(row["frame"] != i for i, row in enumerate(data)):
            raise ValueError("Per-frame annotation count or index mismatch")
    return verified_file(out, scene["file"], scene["sha256"]), start


def evaluate(out, manifest, ocr, detector="motion"):
    from .config import Camera
    from .pipeline import analyze

    if (
        manifest["version"] != 1
        or not isinstance(manifest["generated"], bool)
        or not manifest["provenance"]
    ):
        raise ValueError("Require a version 1 dataset with declared provenance")
    if not manifest["scenes"] or len({s["name"] for s in manifest["scenes"]}) != len(manifest["scenes"]):
        raise ValueError("A dataset requires uniquely named scenes")
    config = Camera.load(verified_file(out, "camera.json", manifest["camera_sha256"]))
    scenes = []
    totals = dict.fromkeys(("tp", "fp", "fn", "plate_exact", "plate_read", "plate_eligible"), 0)
    total_seconds, total_frames = 0.0, 0
    for scene in manifest["scenes"]:
        observed = []
        path, start = validate_scene(scene, out)
        before = perf_counter()
        analysis = analyze(
            path, config, detector, start, lambda event, evidence: observed.append(event.copy()), ocr=ocr
        )
        seconds = perf_counter() - before
        if analysis["frames"] != scene["frames"]:
            raise ValueError(f"Decoded frame count differs from manifest for {scene['name']}")
        scored = score_events(scene["expected_events"], observed)
        for key in totals:
            totals[key] += scored[key]
        total_seconds += seconds
        total_frames += scene["frames"]
        scenes.append(
            {
                "name": scene["name"],
                "expected": scene["expected_events"],
                "observed": observed,
                "score": scored,
                "frames": analysis["frames"],
                "seconds": seconds,
                "signal_frames": analysis["signal_frames"],
            }
        )
    tp, fp, fn, labeled = (totals[k] for k in ("tp", "fp", "fn", "plate_eligible"))
    implementations = ("config", "detection", "experiment", "pipeline", "plates", "rules", "tracking")
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scipy": metadata.version("scipy"),
        "opencv_threads": cv2.getNumThreads(),
    }
    if ocr:
        import pytesseract

        environment["tesseract"] = str(pytesseract.get_tesseract_version()).splitlines()[0]
        environment["pytesseract"] = metadata.version("pytesseract")
    if detector == "coco":
        import torch

        environment.update(
            torch=metadata.version("torch"),
            torchvision=metadata.version("torchvision"),
            device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        )
    report = {
        "provenance": manifest["provenance"],
        "generated": manifest["generated"],
        "environment": environment,
        "ocr_enabled": ocr,
        "detector": detector,
        "manifest_sha256": sha256(out / "manifest.json"),
        "camera_sha256": sha256(out / "camera.json"),
        "source_sha256": {
            name + ".py": sha256(Path(__file__).with_name(name + ".py")) for name in implementations
        },
        "frame_tolerance": FRAME_TOLERANCE,
        "position_tolerance": POSITION_TOLERANCE,
        "totals": {
            **totals,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "plate_exact_per_labeled_match": ratio(totals["plate_exact"], labeled) if ocr else None,
            "plate_read_per_labeled_match": ratio(totals["plate_read"], labeled) if ocr else None,
            "seconds": total_seconds,
            "frames": total_frames,
            "processed_fps": ratio(total_frames, total_seconds),
        },
        "scenes": scenes,
    }
    dump_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for generated outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Read and verify camera.json and manifest.json already in --out",
    )
    parser.add_argument("--detector", choices=("motion", "coco"), default="motion")
    parser.add_argument("--preview", type=Path, help="Optional daylight demo GIF destination")
    parser.add_argument("--no-ocr", action="store_true", help="Skip plate reads during evaluation")
    args = parser.parse_args()
    if args.evaluate_existing:
        if args.preview:
            parser.error("--preview is available when generating the synthetic daylight clip")
        manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = generate_dataset(args.out)
    if args.preview:
        write_preview(args.out / "clips" / "daylight.avi", args.preview)
    if args.generate_only:
        print(args.out / "manifest.json")
        return
    report = evaluate(args.out, manifest, not args.no_ocr, args.detector)
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
