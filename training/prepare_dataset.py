import cv2
import numpy as np
import yaml
import os
import random
from pathlib import Path

"""
Generates a synthetic YOLO-format dataset from the sample video
for demo fine-tuning. Uses color-range detection to auto-annotate
the synthetic instruments created by generate_sample.py
"""

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

instruments = cfg["instruments"]
video_path = "data/videos/sample.mp4"
output_dir = Path("data/dataset")
(output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

color_ranges = {
    "scalpel":       ([20,  20, 150], [80,  80, 255]),
    "forceps":       ([20, 150, 20],  [80,  255, 80]),
    "needle_holder": ([150, 20, 20],  [255, 80,  80]),
    "suction":       ([20, 150, 150], [80,  255, 255]),
}

cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
sample_every = 5
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if fid % sample_every != 0:
        continue

    h, w = frame.shape[:2]
    annotations = []

    for cls_id, inst in enumerate(instruments):
        lo = np.array(color_ranges[inst][0], dtype=np.uint8)
        hi = np.array(color_ranges[inst][1], dtype=np.uint8)
        mask = cv2.inRange(frame, lo, hi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 300:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if not annotations:
        continue

    split = "train" if random.random() < 0.8 else "val"
    img_path = output_dir / "images" / split / f"frame_{fid:05d}.jpg"
    lbl_path = output_dir / "labels" / split / f"frame_{fid:05d}.txt"
    cv2.imwrite(str(img_path), frame)
    with open(lbl_path, "w") as f:
        f.write("\n".join(annotations))
    saved += 1

cap.release()

data_yaml = {
    "path": str(output_dir.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": len(instruments),
    "names": instruments
}
with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print(f"Dataset prepared: {saved} frames saved to {output_dir}")