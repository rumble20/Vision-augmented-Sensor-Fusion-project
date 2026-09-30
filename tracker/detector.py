from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    cx: float
    cy: float
    conf: float = 1.0


class SimulatedDetector:
    """Stands in for the network: noisy bbox centre of the true target,
    and nothing at all on the frames listed as dropped."""

    def __init__(self, noise_pix=3.0, dropped_frames=(), seed=0):
        self.noise = noise_pix
        self.dropped = set(dropped_frames)
        self.rng = np.random.default_rng(seed)

    def detect(self, true_xy, frame_idx):
        if frame_idx in self.dropped:
            return None
        cx, cy = np.asarray(true_xy) + self.rng.normal(0, self.noise, 2)
        return Detection(cx, cy)


class YoloDetector:
    """Wraps an ultralytics yolo model and returns the most confident box
    of the wanted class. Import is lazy so the rest runs without torch."""

    def __init__(self, weights="yolov8n.pt", target_class=0, min_conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.target_class = target_class
        self.min_conf = min_conf

    def detect(self, frame, frame_idx=None):
        result = self.model(frame, verbose=False)[0]
        best = None
        for box in result.boxes:
            conf = float(box.conf)
            if int(box.cls) != self.target_class or conf < self.min_conf:
                continue
            if best is None or conf > best.conf:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best = Detection((x1 + x2) / 2, (y1 + y2) / 2, conf)
        return best
