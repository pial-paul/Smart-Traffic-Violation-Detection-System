"""Camera geometry in normalized image coordinates."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


def coords(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("Invalid coordinate dimensions")
    if any(type(n) not in (float, int) or not math.isfinite(n) or not 0 <= n <= 1 for n in value):
        raise ValueError("Camera coordinates must be finite numbers between 0 and 1")
    return tuple(value)


@dataclass(frozen=True)
class Camera:
    camera: str
    road_roi: tuple
    signal_roi: tuple
    stop_line: tuple

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) != {"camera", "road_roi", "signal_roi", "stop_line"}:
            raise ValueError("Expected camera, road_roi, signal_roi and stop_line")
        name = data["camera"]
        if not isinstance(name, str) or not name.strip() or len(name) > 80 or not name.isprintable():
            raise ValueError("Invalid camera name")
        road, signal = (coords(data[k], 4) for k in ("road_roi", "signal_roi"))
        for x1, y1, x2, y2 in (road, signal):
            if x1 >= x2 or y1 >= y2:
                raise ValueError("ROI must be ordered left, top, right, bottom")
        if not isinstance(data["stop_line"], list) or len(data["stop_line"]) != 2:
            raise ValueError("Supply two stop-line endpoints")
        a, b = (coords(p, 2) for p in data["stop_line"])
        if math.dist(a, b) < 0.02:
            raise ValueError("Stop line too short")
        return cls(name, road, signal, (a, b))


def pixels(roi, width, height):
    x1, y1, x2, y2 = roi
    return int(x1 * width), int(y1 * height), math.ceil(x2 * width), math.ceil(y2 * height)
