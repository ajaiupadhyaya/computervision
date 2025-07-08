import cv2
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

# Load model
model = YOLO('yolov8n.pt')

# Load video
video_path = 'game1_highlights.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Skip first 1:28 (88 seconds)
start_seconds = 88
start_frame = int(start_seconds * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

results = []
frame_idx = start_frame  # Start from the real frame number

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = model(frame)[0]
    for det in detections.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        if label in ['person']:  # We'll add 'sports ball' later if needed
            results.append({
                'frame': frame_idx,
                'label': label,
                'conf': conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

    frame_idx += 1

cap.release()

df = pd.DataFrame(results)
df.to_csv('game1_detections.csv', index=False)
print(f"✅ Done! Detection started at {start_seconds}s and saved to game1_detections.csv")