import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load detections
df = pd.read_csv('game1_detections.csv')

# Keep only confident detections
df = df[df['conf'] > 0.6]

# Group by frame
grouped = df.groupby('frame')

# Track all player centroid positions (we'll assume shooter is furthest from basket)
shots = []

for frame, group in grouped:
    centroids = []
    for _, row in group.iterrows():
        cx = (row['x1'] + row['x2']) / 2
        cy = (row['y1'] + row['y2']) / 2
        centroids.append((cx, cy))
    
    if centroids:
        # Use furthest point from bottom center (basket location)
        basket_x = 960  # assume 1280x720 resolution → center
        basket_y = 1050
        distances = [np.sqrt((cx - basket_x)**2 + (cy - basket_y)**2) for (cx, cy) in centroids]
        shooter_idx = np.argmax(distances)
        shooter_cx, shooter_cy = centroids[shooter_idx]
        shots.append((frame, shooter_cx, shooter_cy))

# Convert to DataFrame
shot_df = pd.DataFrame(shots, columns=['frame', 'x', 'y'])

# Normalize coords to a 50x47 NBA halfcourt
court_width = 1920
court_height = 720

shot_df['x_norm'] = shot_df['x'] / court_width * 50
shot_df['y_norm'] = (1 - shot_df['y'] / court_height) * 47  # invert y

# --- Plotting ---
def draw_halfcourt(ax):
    # Draw court lines
    hoop = patches.Circle((25, 5.25), radius=0.75, linewidth=2, fill=False)
    backboard = patches.Rectangle((22, 4), 6, 0.1, linewidth=2, fill=False)
    paint = patches.Rectangle((17, 0), 16, 19, linewidth=2, fill=False)
    free_throw = patches.Circle((25, 19), radius=6, linewidth=2, fill=False)
    three_pt = patches.Arc((25, 5.25), 47.5, 47.5, theta1=22, theta2=158, linewidth=2)

    ax.add_patch(hoop)
    ax.add_patch(backboard)
    ax.add_patch(paint)
    ax.add_patch(free_throw)
    ax.add_patch(three_pt)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 47)
    ax.set_aspect(1)
    ax.axis('off')
    ax.set_title('Estimated Shot Locations — 2025 NBA Finals Game 1', fontsize=14)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
draw_halfcourt(ax)
ax.scatter(shot_df['x_norm'], shot_df['y_norm'], c='red', s=30, label='Estimated Shot')
ax.legend()
plt.tight_layout()
plt.savefig('game1_shot_chart.png')
plt.show()