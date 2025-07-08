# extract_features.py

import pandas as pd
import numpy as np

# === Load detection data ===
df = pd.read_csv("player_detections.csv")

# === Sort by frame and position (placeholder for tracking) ===
df = df.sort_values(by=["frame", "cx", "cy"]).reset_index(drop=True)

# === Group by dummy player_id for now ===
# We'll just simulate tracking by grouping by detection index % N
N_FAKE_PLAYERS = 10
df["player_id"] = df.index % N_FAKE_PLAYERS

# === Initialize storage ===
feature_rows = []

# === For each player_id, compute velocity and acceleration ===
for pid, group in df.groupby("player_id"):
    group = group.sort_values("frame").reset_index(drop=True)

    for i in range(1, len(group)):
        row_prev = group.iloc[i - 1]
        row_curr = group.iloc[i]

        dt = row_curr["frame"] - row_prev["frame"]
        if dt == 0:
            continue

        dx = row_curr["cx"] - row_prev["cx"]
        dy = row_curr["cy"] - row_prev["cy"]
        vx = dx / dt
        vy = dy / dt
        speed = np.sqrt(vx**2 + vy**2)

        if i >= 2:
            prev_speed = np.sqrt(
                (group.iloc[i - 1]["cx"] - group.iloc[i - 2]["cx"])**2 +
                (group.iloc[i - 1]["cy"] - group.iloc[i - 2]["cy"])**2
            ) / (group.iloc[i - 1]["frame"] - group.iloc[i - 2]["frame"])
            acceleration = (speed - prev_speed) / dt
        else:
            acceleration = 0

        feature_rows.append({
            "frame": row_curr["frame"],
            "player_id": pid,
            "cx": row_curr["cx"],
            "cy": row_curr["cy"],
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "acceleration": acceleration,
        })

# === Save features ===
features_df = pd.DataFrame(feature_rows)
features_df.to_csv("features.csv", index=False)

print(f"✅ Saved {len(features_df)} player-movement feature rows to 'features.csv'")