# sloane/pipeline/vision_to_features.py

import cv2
import os
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# === Config ===
VIDEO_PATH = "data/game72016.mp4"  # replace with actual file
MODEL_PATH = "yolov8x.pt"     # or use "yolov8x.pt" for accuracy
CONF_THRESHOLD = 0.3
SAVE_PATH = "data/features.csv"

# === Load YOLOv8 model ===
model = YOLO(MODEL_PATH)
assert model, "❌ Model load failed."

# === Read video ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🎥 Processing {VIDEO_PATH} at {fps:.2f} FPS, {frame_count} total frames...")

# === Track player centroids + dynamics ===
all_data = []
player_tracks = {}  # player_id: [ (frame, (x, y)) ]

frame_num = 0
while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
    boxes = results.boxes
    cls = results.names

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        if cls[cls_id] != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        player_id = int(box.id[0]) if box.id is not None else i  # basic ID fallback

        # Save position
        all_data.append({
            "frame": frame_num,
            "player_id": player_id,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "cx": cx,
            "cy": cy,
        })

        if player_id not in player_tracks:
            player_tracks[player_id] = []
        player_tracks[player_id].append((frame_num, cx, cy))

    frame_num += 1
    if frame_num % 100 == 0:
        print(f"🔁 Frame {frame_num}/{frame_count}...")

cap.release()

# === Create DataFrame ===
df = pd.DataFrame(all_data)
print(f"✅ Detected {len(df)} total player entries.")

# === Add speed & acceleration ===
df["speed"] = 0.0
df["acceleration"] = 0.0

for pid, track in player_tracks.items():
    prev_v = 0
    for i in range(1, len(track)):
        f1, x1, y1 = track[i - 1]
        f2, x2, y2 = track[i]

        dt = f2 - f1
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        v = dist / dt if dt > 0 else 0
        a = (v - prev_v) / dt if dt > 0 else 0

        df.loc[(df.player_id == pid) & (df.frame == f2), "speed"] = v
        df.loc[(df.player_id == pid) & (df.frame == f2), "acceleration"] = a
        prev_v = v

# === Save to CSV ===
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SAVE_PATH, index=False)
print(f"📁 Saved frame-level feature data to {SAVE_PATH}")