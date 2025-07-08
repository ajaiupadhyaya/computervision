# track_players.py

from ultralytics import YOLO
import cv2
import os
import pandas as pd

# === CONFIGURATION ===
VIDEO_PATH = "game72016.mp4"
MODEL_PATH = "yolov8n.pt"  # Use the smaller model for speed
CSV_OUTPUT_PATH = "player_detections.csv"
FRAME_SKIP = 6             # ~5 FPS
MAX_FRAMES = 1000000000000000000
# === SETUP ===
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

detections = []
frame_id = 0
processed_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or processed_frame_count >= MAX_FRAMES:
        break

    if frame_id % FRAME_SKIP == 0:
        resized = cv2.resize(frame, (1280, 720))
        results = model(resized, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confidences):
                if int(cls) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                    detections.append({
                        "frame": frame_id,
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy,
                        "confidence": conf
                    })

        processed_frame_count += 1

    frame_id += 1

cap.release()

# === SAVE TO CSV ===
df = pd.DataFrame(detections)
df.to_csv(CSV_OUTPUT_PATH, index=False)

print(f"✅ Done! Saved {len(detections)} detections from {processed_frame_count} frames.")
print(f"📄 Output: {CSV_OUTPUT_PATH}")