import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from court_utils import draw_halfcourt

# --- SETTINGS ---
COURT_WIDTH = 1920
COURT_HEIGHT = 1080
OUTPUT_RES_X = 50  # match NBA halfcourt width
OUTPUT_RES_Y = 47  # match NBA halfcourt height

# Load detections
df = pd.read_csv('game1_detections.csv')

# Filter confident person detections
df = df[(df['label'] == 'person') & (df['conf'] > 0.6)]

# Compute centroids
df['cx'] = (df['x1'] + df['x2']) / 2
df['cy'] = (df['y1'] + df['y2']) / 2

# Normalize to NBA court scale (50x47)
df['x_norm'] = df['cx'] / COURT_WIDTH * 50
df['y_norm'] = (1 - df['cy'] / COURT_HEIGHT) * 47  # invert y

coords = df[['x_norm', 'y_norm']].values

# --- Heatmap Kernel Density Estimation ---
print("📡 Performing KDE Heatmap Estimation...")
kde = gaussian_kde(coords.T, bw_method=0.3)

# Create grid
xgrid = np.linspace(0, 50, 500)
ygrid = np.linspace(0, 47, 470)
X, Y = np.meshgrid(xgrid, ygrid)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

# --- Plot Global Heatmap ---
fig, ax = plt.subplots(figsize=(8, 7))
draw_halfcourt(ax, title="🔥 Player Movement Heatmap — Game 1")
cmap = sns.color_palette("rocket", as_cmap=True)
heat = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
plt.colorbar(heat, ax=ax, label="Movement Density")
plt.tight_layout()
plt.savefig("movement_heatmap.png")
plt.show()

# --- Zone Clustering (K-Means) ---
print("🧪 Running K-Means Clustering on Position Data...")
k = 6  # number of movement zones
kmeans = KMeans(n_clusters=k, n_init=10)
df['zone'] = kmeans.fit_predict(coords)

# --- Plot Clusters Over Court ---
fig, ax = plt.subplots(figsize=(8, 7))
draw_halfcourt(ax, title="🧬 Movement Zone Clusters — Game 1")
palette = sns.color_palette("Set2", k)

for i in range(k):
    cluster_points = df[df['zone'] == i]
    ax.scatter(cluster_points['x_norm'], cluster_points['y_norm'], s=8, color=palette[i], label=f"Zone {i+1}", alpha=0.5)

ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("movement_clusters.png")
plt.show()

# --- Zone Density Table ---
zone_stats = df.groupby('zone')[['x_norm', 'y_norm']].agg(['mean', 'std', 'count'])
zone_stats.columns = ['_'.join(col) for col in zone_stats.columns]
zone_stats.reset_index(inplace=True)

print("📊 Zone Density Summary:")
print(zone_stats.to_string(index=False))

zone_stats.to_csv("movement_zone_summary.csv", index=False)
print("✅ All outputs saved: movement_heatmap.png, movement_clusters.png, movement_zone_summary.csv")