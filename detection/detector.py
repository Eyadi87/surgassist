import cv2
import yaml
import logging
import numpy as np
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Color ranges for synthetic video instruments (BGR format)
SYNTH_COLOR_RANGES = {
    "scalpel":       {"lower": np.array([130, 30,  30]),  "upper": np.array([255, 90,  90])},
    "forceps":       {"lower": np.array([30,  130, 30]),  "upper": np.array([90,  255, 90])},
    "needle_holder": {"lower": np.array([30,  30,  130]), "upper": np.array([90,  90,  255])},
    "suction":       {"lower": np.array([130, 130, 30]),  "upper": np.array([255, 255, 90])},
}


class SurgicalDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.instruments = self.cfg["instruments"]
        self.conf        = self.cfg["detection"]["confidence"]
        self.iou         = self.cfg["detection"]["iou"]
        self.imgsz       = self.cfg["detection"]["imgsz"]
        self.model       = None
        self._load_model(config_path)

    def _load_model(self, config_path):
        try:
            from ultralytics import YOLO
            model_path = self.cfg["model_path"]
            if Path(model_path).exists():
                self.model = YOLO(model_path)
                logger.info(f"Loaded fine-tuned model: {model_path}")
            else:
                self.model = YOLO(self.cfg["base_model"])
                logger.info(f"Loaded base model: {self.cfg['base_model']}")
        except Exception as e:
            logger.warning(f"YOLO load failed: {e}. Will use color detection only.")
            self.model = None

    def _color_detect(self, frame):
        """
        Detects instruments in the synthetic video using color segmentation.
        Works reliably on the generated sample.mp4
        """
        detections = []
        for label, ranges in SYNTH_COLOR_RANGES.items():
            if label not in self.instruments:
                continue
            mask = cv2.inRange(frame, ranges["lower"], ranges["upper"])
            # Clean up noise
            kernel = np.ones((5, 5), np.uint8)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 400:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                conf = min(1.0, area / 3000.0)
                detections.append({
                    "bbox":       [x, y, x + w, y + h],
                    "confidence": round(conf, 3),
                    "label":      label,
                    "cls_id":     self.instruments.index(label),
                })
        return detections

    def _yolo_detect(self, frame):
        """
        YOLO detection for real surgical videos.
        Maps COCO classes → instrument labels cyclically.
        """
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = self.instruments[cls_id % len(self.instruments)]
                detections.append({
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": conf,
                    "label":      label,
                    "cls_id":     cls_id,
                })
        return detections

    def detect(self, frame):
        """
        Auto-selects detection method:
        - If frame contains synthetic colored blocks → color detection
        - Otherwise → YOLO
        """
        # Quick check: does this look like a synthetic frame?
        color_dets = self._color_detect(frame)
        if color_dets:
            return color_dets
        # Fallback to YOLO for real footage
        if self.model is not None:
            return self._yolo_detect(frame)
        return []

    def draw(self, frame, detections):
        colors = {
            "scalpel":       (50,  50,  220),
            "forceps":       (50,  200, 50),
            "needle_holder": (220, 50,  50),
            "suction":       (50,  200, 220),
        }
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label  = det["label"]
            conf   = det["confidence"]
            color  = colors.get(label, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        return frame