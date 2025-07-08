import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load detections
df = pd.read_csv('game1_detections.csv')
df = df[(df['label'] == 'person') & (df['conf'] > 0.6)]

# Compute centroids
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2

# Group by frame
frame_groups = df.groupby('frame')

frame_stats = []
for frame, group in frame_groups:
    coords = group[['cx', 'cy']].values
    if len(coords) < 2:
        continue
    center = np.mean(coords, axis=0)
    spread = np.mean(np.linalg.norm(coords - center, axis=1))
    frame_stats.append({'frame': frame, 'spread': spread})

stats_df = pd.DataFrame(frame_stats)

# --- Possession Detection ---
# When spacing/spread shrinks or there's a time gap, start a new possession
stats_df['frame_diff'] = stats_df['frame'].diff().fillna(1)
stats_df['spread_change'] = stats_df['spread'].diff().fillna(0)

# Heuristic thresholds (can be tuned)
time_gap_thresh = 15  # frames (~0.5s)
spread_thresh = 100   # spacing collapse

# Start new possession if time jumps or spacing resets
stats_df['new_possession'] = (
    (stats_df['frame_diff'] > time_gap_thresh) |
    (stats_df['spread_change'].abs() > spread_thresh)
).astype(int)

# Assign possession IDs
possession_id = 0
possessions = []
for _, row in stats_df.iterrows():
    if row['new_possession'] == 1:
        possession_id += 1
    possessions.append(possession_id)
stats_df['possession_id'] = possessions

# --- Compute Possession Stats ---
possession_summaries = []

for pid, group in stats_df.groupby('possession_id'):
    start = group['frame'].min()
    end = group['frame'].max()
    duration_frames = end - start
    avg_spread = group['spread'].mean()

    possession_summaries.append({
        'possession_id': pid,
        'start_frame': start,
        'end_frame': end,
        'duration_frames': duration_frames,
        'avg_spread': avg_spread
    })

summary_df = pd.DataFrame(possession_summaries)
summary_df['duration_sec'] = summary_df['duration_frames'] / 30  # assuming 30fps

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(stats_df['frame'], stats_df['spread'], color='purple', lw=1.5)
for pid in summary_df['possession_id']:
    start = summary_df.loc[summary_df['possession_id'] == pid, 'start_frame'].values[0]
    ax.axvline(start, color='gray', ls='--', alpha=0.5)

ax.set_title("🏀 Player Spread Over Time — Possession Markers")
ax.set_xlabel("Frame")
ax.set_ylabel("Avg Player Spread (px)")
plt.tight_layout()
plt.savefig("possession_timeline.png")
plt.show()

# --- Export ---
summary_df.to_csv("possessions_summary.csv", index=False)
print("✅ Possession analysis complete.")
print("- possessions_summary.csv")
print("- possession_timeline.png")