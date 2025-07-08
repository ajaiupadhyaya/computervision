# train_movement_classifier.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# === Load features ===
df = pd.read_csv("features.csv")

# === Sanitize data: drop rows with NaN or infinite values ===
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# === Cap extremely large speeds/accels (outliers or bad detections) ===
df = df[df["speed"] < 100]
df = df[df["acceleration"].abs() < 100]

# === Create movement class based on speed threshold ===
df["movement_class"] = (df["speed"] > 4.0).astype(int)  # 1 = fast, 0 = slow

# === Features and target ===
X = df[["vx", "vy", "speed", "acceleration"]]
y = df["movement_class"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Train model ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === Save model ===
joblib.dump(clf, "movement_classifier.pkl")
print("✅ Model saved as 'movement_classifier.pkl'")