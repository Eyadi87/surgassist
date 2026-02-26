"""
Generates a synthetic surgical video with clearly colored instrument boxes.
Colors are chosen to match the detector's SYNTH_COLOR_RANGES exactly.
"""
import cv2
import numpy as np
import random
import yaml
from pathlib import Path

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

Path("data/videos").mkdir(parents=True, exist_ok=True)
output_path = "data/videos/sample.mp4"

WIDTH, HEIGHT = 640, 480
FPS           = 25
DURATION_SEC  = 90          # 90 seconds = 2250 frames — enough data
TOTAL_FRAMES  = FPS * DURATION_SEC

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))

# BGR colors — must match SYNTH_COLOR_RANGES in detector.py
COLORS = {
    "scalpel":       (50,  50,  200),   # red-ish
    "forceps":       (50,  200, 50),    # green
    "needle_holder": (200, 50,  50),    # blue
    "suction":       (50,  200, 200),   # yellow
}

instruments = cfg["instruments"]

# Initial state for each instrument
states = {
    inst: {
        "x":  random.randint(100, WIDTH - 100),
        "y":  random.randint(100, HEIGHT - 100),
        "vx": random.choice([-3, -2, 2, 3]),
        "vy": random.choice([-3, -2, 2, 3]),
        "w":  random.randint(80, 130),
        "h":  30,
        "visible": True,
        "hide_until": 0,
    }
    for inst in instruments
}

# Stagger instrument appearances — simulate realistic OR tray usage
appearance_schedule = {
    "scalpel":       0,
    "forceps":       int(TOTAL_FRAMES * 0.1),
    "needle_holder": int(TOTAL_FRAMES * 0.3),
    "suction":       int(TOTAL_FRAMES * 0.5),
}

# One instrument is "rarely used" to trigger waste detection
RARE_INSTRUMENT = "suction"

for f_idx in range(TOTAL_FRAMES):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # OR background
    cv2.rectangle(frame, (30, 30), (WIDTH - 30, HEIGHT - 30), (25, 35, 25), -1)
    cv2.rectangle(frame, (30, 30), (WIDTH - 30, HEIGHT - 30), (50, 70, 50), 2)
    cv2.putText(frame, "OPERATING ROOM - INSTRUMENT TABLE",
                (WIDTH // 2 - 190, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 80), 1)

    for inst in instruments:
        s = states[inst]

        # Respect appearance schedule
        if f_idx < appearance_schedule[inst]:
            continue

        # Rare instrument: only visible 15% of time
        if inst == RARE_INSTRUMENT:
            if f_idx % 100 < 15:
                s["visible"] = True
            else:
                s["visible"] = False
        else:
            # Normal instruments: toggle visibility every 3-10 seconds
            if f_idx % (FPS * random.randint(3, 10)) == 0:
                s["visible"] = random.random() > 0.2

        if not s["visible"]:
            continue

        # Move instrument — keep well within bounds so labels never clip edges
        s["x"] = int(np.clip(s["x"] + s["vx"], s["w"] // 2 + 60, WIDTH  - s["w"] // 2 - 60))
        s["y"] = int(np.clip(s["y"] + s["vy"], s["h"] // 2 + 60, HEIGHT - s["h"] // 2 - 60))
        if s["x"] <= s["w"] // 2 + 60 or s["x"] >= WIDTH  - s["w"] // 2 - 60:
            s["vx"] *= -1
        if s["y"] <= s["h"] // 2 + 60 or s["y"] >= HEIGHT - s["h"] // 2 - 60:
            s["vy"] *= -1

        x1 = s["x"] - s["w"] // 2
        y1 = s["y"] - s["h"] // 2
        x2 = s["x"] + s["w"] // 2
        y2 = s["y"] + s["h"] // 2

        color = COLORS[inst]
        # Draw filled rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        # White border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # Label ABOVE the box (not inside) so it never clips
        label_y = max(y1 - 6, 50)
        cv2.putText(frame, inst, (x1 + 4, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # No timestamp baked into video — dashboard draws it cleanly on top

    writer.write(frame)

writer.release()
print(f"✅ Sample video saved: {output_path}")
print(f"   {TOTAL_FRAMES} frames @ {FPS}fps = {DURATION_SEC}s")
print(f"   Rare instrument (waste demo): {RARE_INSTRUMENT}")