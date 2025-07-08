# cluster_possessions.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and clean possession features ===
df = pd.read_csv("possessions.csv")

# Fix any infinities or NaNs
df["avg_acceleration"] = df["avg_acceleration"].replace([np.inf, -np.inf], np.nan).fillna(0)
df["avg_speed"] = df["avg_speed"].replace([np.inf, -np.inf], np.nan).fillna(0)
df["bbox_area"] = df["bbox_area"].replace([np.inf, -np.inf], np.nan).fillna(0)

# === Features for clustering ===
features = df[["avg_speed", "avg_acceleration", "bbox_area", "duration"]].copy()

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# === Clustering ===
k = 4  # number of possession types to discover
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
df["cluster"] = kmeans.fit_predict(X_scaled)

# === Save result ===
df.to_csv("possessions_labeled.csv", index=False)
print(f"✅ Clustered into {k} types → saved to possessions_labeled.csv")

# === 2D PCA visualization ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# === Plot ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="cluster", palette="Set2", s=80)
plt.title("🏀 Possession Clusters (PCA Projection)")
plt.xlabel("PC1 – tempo + spacing")
plt.ylabel("PC2 – movement dynamics")
plt.legend(title="Possession Type", loc="upper right")
plt.tight_layout()
plt.savefig("possession_clusters.png")
plt.show()