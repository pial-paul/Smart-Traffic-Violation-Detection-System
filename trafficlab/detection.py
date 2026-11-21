"""Vehicle and foreground detection; boxes are full-frame normalized xyxy."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import pixels


@dataclass(frozen=True)
class Detection:
    box: tuple
    label: str
    confidence: float

    @property
    def foot(self):
        x1, _, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, y2])


class MotionDetector:
    """Offline fixed-camera baseline. Foreground blobs have no semantic class."""

    def __init__(self):
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=28, detectShadows=False
        )
        self.frames = 0

    def detect(self, image):
        mask = self.background.apply(image, learningRate=0.003)
        self.frames += 1
        if self.frames <= 3:
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        parts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        boxes = []
        for part in parts:
            x, y, width, height = cv2.boundingRect(part)
            if 0.001 <= cv2.contourArea(part) / (h * w) <= 0.45 and min(width, height) >= 8:
                boxes.append(((x, y, x + width, y + height), "foreground", 1.0))
        return boxes


class CocoDetector:
    def __init__(self):
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        self.torch = torch
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.labels = weights.meta["categories"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to(self.device).eval()

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).to(self.device).float() / 255
        with self.torch.inference_mode():
            out = self.model([tensor])[0]
        boxes = out["boxes"].cpu().tolist()
        labels = out["labels"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        return [
            (box, self.labels[label], score)
            for box, label, score in zip(boxes, labels, scores)
            if self.labels[label] in {"car", "truck", "bus", "motorcycle"} and score >= 0.55
        ]


def detect_frame(model, frame, roi):
    h, w = frame.shape[:2]
    left, top, right, bottom = pixels(roi, w, h)
    result = []
    for (x1, y1, x2, y2), label, score in model.detect(frame[top:bottom, left:right]):
        box = ((x1 + left) / w, (y1 + top) / h, (x2 + left) / w, (y2 + top) / h)
        result.append(Detection(box, label, score))
    return sorted(result, key=lambda det: det.box)
