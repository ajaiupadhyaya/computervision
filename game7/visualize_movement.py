# visualize_movement.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Load data ===
df = pd.read_csv("features.csv")
df = df[df["speed"] < 100]
df["movement_class"] = (df["speed"] > 4.0).astype(int)

# === Setup plot ===
plt.figure(figsize=(16, 8))
plt.title("🏀 Player Movement Classes Over Time", fontsize=18)
plt.xlabel("Frame", fontsize=14)
plt.ylabel("Speed", fontsize=14)

# === Color map for classes ===
colors = {0: "blue", 1: "red"}

# === Plot each player ===
for pid, group in df.groupby("player_id"):
    plt.scatter(
        group["frame"], group["speed"],
        c=group["movement_class"].map(colors),
        s=8, alpha=0.7, label=f"Player {pid}"
    )

# === Add legend and styling ===
handles = [plt.Line2D([0], [0], marker='o', color='w', label='Fast Movement', markerfacecolor='red', markersize=10),
           plt.Line2D([0], [0], marker='o', color='w', label='Slow Movement', markerfacecolor='blue', markersize=10)]
plt.legend(handles=handles, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# === Show or save ===
plt.savefig("movement_timeline.png", dpi=200)
plt.show()