# segment_possessions.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

# === Load and prepare feature data ===
df = pd.read_csv("features.csv")
df = df[df["speed"] < 100].copy()
df = df.sort_values(by=["frame", "player_id"]).reset_index(drop=True)

if df.empty:
    print("❌ Error: features.csv is empty or invalid.")
    exit()

print(f"✅ Loaded {len(df)} feature rows.")

# === Configuration thresholds ===
FRAME_GAP_THRESHOLD = 30       # Allow longer gaps between frames
SPACING_CHANGE_THRESHOLD = 50000  # Loosen spacing break threshold
SPEED_DROP_THRESHOLD = 0.8     # Consider steep speed drops
POSSESSION_MIN_LENGTH = 10     # Allow short segments

# === Helper: calculate spacing area (bounding box of players) ===
def spacing_area(group):
    xs = group["cx"].values
    ys = group["cy"].values
    return (max(xs) - min(xs)) * (max(ys) - min(ys))

# === Possession segmentation ===
possessions = []
current_possession = []
current_start_frame = None
prev_frame = None
prev_spacing = None
prev_speed = None

for frame_id in sorted(df["frame"].unique()):
    frame_data = df[df["frame"] == frame_id]

    if prev_frame is not None:
        frame_gap = frame_id - prev_frame
        spacing = spacing_area(frame_data)
        avg_speed = frame_data["speed"].mean()

        spacing_diff = abs(spacing - prev_spacing) if prev_spacing is not None else 0
        speed_drop = (prev_speed - avg_speed) / prev_speed if prev_speed and prev_speed > 0 else 0

        # === Break condition ===
        if (
            frame_gap > FRAME_GAP_THRESHOLD
            or spacing_diff > SPACING_CHANGE_THRESHOLD
            or speed_drop > SPEED_DROP_THRESHOLD
        ):
            if len(current_possession) >= POSSESSION_MIN_LENGTH:
                # === Save possession segment ===
                possession_df = pd.DataFrame(current_possession)
                possessions.append({
                    "start_frame": possession_df["frame"].min(),
                    "end_frame": possession_df["frame"].max(),
                    "duration": possession_df["frame"].nunique(),
                    "avg_speed": possession_df["avg_speed"].mean(),
                    "avg_acceleration": possession_df["avg_acceleration"].mean(),
                    "bbox_area": possession_df["spacing"].mean(),
                    "frame_count": len(possession_df)
                })
                print(f"📦 Saved possession from frame {current_start_frame} to {prev_frame}.")

            current_possession = []
            current_start_frame = None

    # === Record current frame ===
    spacing = spacing_area(frame_data)
    avg_speed = frame_data["speed"].mean()
    avg_acceleration = frame_data["acceleration"].mean()

    current_possession.append({
        "frame": frame_id,
        "spacing": spacing,
        "avg_speed": avg_speed,
        "avg_acceleration": avg_acceleration
    })

    if current_start_frame is None:
        current_start_frame = frame_id

    prev_spacing = spacing
    prev_speed = avg_speed
    prev_frame = frame_id

# === Final possession (if any) ===
if len(current_possession) >= POSSESSION_MIN_LENGTH:
    possession_df = pd.DataFrame(current_possession)
    possessions.append({
        "start_frame": possession_df["frame"].min(),
        "end_frame": possession_df["frame"].max(),
        "duration": possession_df["frame"].nunique(),
        "avg_speed": possession_df["avg_speed"].mean(),
        "avg_acceleration": possession_df["avg_acceleration"].mean(),
        "bbox_area": possession_df["spacing"].mean(),
        "frame_count": len(possession_df)
    })
    print(f"📦 Saved final possession from frame {current_start_frame} to {prev_frame}.")

# === Save to CSV ===
possessions_df = pd.DataFrame(possessions)
possessions_df.to_csv("possessions.csv", index=False)

print(f"\n✅ Total possessions saved: {len(possessions_df)}")
print("📄 Output written to possessions.csv")