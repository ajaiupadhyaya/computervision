import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from court_utils import draw_halfcourt

# --- Config ---
COURT_W = 1920
COURT_H = 1080
HOOP_X = 960
HOOP_Y = 1050

# Load detection data
df = pd.read_csv('game1_detections.csv')
df = df[df['label'] == 'person']
df = df[df['conf'] > 0.6]

# Compute player centroids
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2

# Assume every 20th frame is a potential shot
shot_frames = df['frame'].unique()[::20]
shots = []

for frame in shot_frames:
    frame_df = df[df['frame'] == frame]
    if len(frame_df) < 2:
        continue  # need at least shooter + 1 defender

    # Assume shooter is furthest from hoop
    frame_df['dist_to_hoop'] = np.sqrt((frame_df['cx'] - HOOP_X)**2 + (frame_df['cy'] - HOOP_Y)**2)
    shooter = frame_df.sort_values('dist_to_hoop', ascending=False).iloc[0]
    shooter_pos = (shooter['cx'], shooter['cy'])

    # Nearest defender (shortest Euclidean dist)
    defenders = frame_df[frame_df['cx'] != shooter['cx']]  # crude check
    if defenders.empty:
        continue

    defender_dists = np.sqrt((defenders['cx'] - shooter_pos[0])**2 + (defenders['cy'] - shooter_pos[1])**2)
    nearest_def_dist = defender_dists.min()

    # Distance to hoop (in pixels)
    shot_distance_px = shooter['dist_to_hoop']

    # Normalize to court size
    shot_distance_ft = (shot_distance_px / COURT_W) * 50  # approx
    defender_distance_ft = (nearest_def_dist / COURT_W) * 50

    # Label: simulate make/miss with a soft threshold
    # Later we’ll use real shot result (if we have ball or hoop state)
    simulated_make = int(shot_distance_ft < 20 and defender_distance_ft > 3)

    shots.append({
        'frame': frame,
        'shot_distance_ft': shot_distance_ft,
        'defender_distance_ft': defender_distance_ft,
        'make': simulated_make
    })

# Build DataFrame
shot_df = pd.DataFrame(shots)
print("📊 Raw Shot Data Sample:")
print(shot_df.head())

# --- Model Training (Logistic Regression) ---
X = shot_df[['shot_distance_ft', 'defender_distance_ft']]
y = shot_df['make']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Predict xFG%
shot_df['xFG'] = model.predict_proba(X_scaled)[:, 1]

# --- Visualize ---
fig, ax = plt.subplots(figsize=(8, 7))
draw_halfcourt(ax, title="🎯 Estimated Shot Difficulty — xFG%")
sc = ax.scatter(
    shot_df['shot_distance_ft'],
    shot_df['defender_distance_ft'],
    c=shot_df['xFG'],
    cmap='coolwarm',
    edgecolor='k',
    s=80
)
plt.colorbar(sc, ax=ax, label='xFG Probability')
ax.set_xlabel("Distance to Hoop (ft)")
ax.set_ylabel("Nearest Defender Distance (ft)")
plt.tight_layout()
plt.savefig("xfg_scatter.png")
plt.show()

# Save data
shot_df.to_csv("shot_difficulty_output.csv", index=False)
print("✅ Shot difficulty analysis complete. Outputs:")
print("- shot_difficulty_output.csv")
print("- xfg_scatter.png")