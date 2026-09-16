"""
train_model.py
--------------
Trains the Kore2 Battery SOH fault-detection pipeline and saves:
  - models/rf_classifier.pkl  — Random Forest
  - models/kmeans_model.pkl   — KMeans (used inside screen_fleet)
  - models/scaler.pkl         — StandardScaler fitted on training data
  - models/summary.csv        — Per-battery engineered features + labels
  - models/label_map.csv      — battery_id -> flagged mapping

Run once, then the dashboard and screen_fleet() reuse the saved artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────────────
# This file lives in backend/; data/ and models/ are siblings inside backend/.
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "Kore2_battery_performance.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

# ── 1. Load & clean ──────────────────────────────────────────────────────────
print("Loading telemetry …")
df = pd.read_csv(DATA, parse_dates=["timestamp"])
df = df.sort_values(["battery_id", "timestamp"]).reset_index(drop=True)

num_cols = ["speed_kph", "voltage_V", "current_A",
            "soc_percent", "state_of_health", "ambient_temp_C"]

df_clean = df.copy()
df_clean[num_cols] = (
    df_clean.groupby("battery_id")[num_cols]
            .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
            .reset_index(level=0, drop=True)
)

# ── 2. Engineer per-battery summary ─────────────────────────────────────────
print("Engineering features …")
rows = []
for bid, g in df_clean.groupby("battery_id"):
    ride = (g[g["state"] == "in_use"]
              .dropna(subset=["soc_percent", "timestamp"])
              .sort_values("timestamp"))
    if len(ride) < 3:
        continue
    t_hr = (ride["timestamp"] - ride["timestamp"].iloc[0]).dt.total_seconds() / 3600
    drain_rate      = -np.polyfit(t_hr, ride["soc_percent"], 1)[0]
    ride_duration   = (ride["timestamp"].iloc[-1] - ride["timestamp"].iloc[0]).total_seconds() / 60
    soc_at_shutdown = ride["soc_percent"].iloc[-1]
    rows.append(dict(
        battery_id            = bid,
        drain_rate_pct_per_hr = round(drain_rate, 2),
        ride_duration_min     = round(ride_duration, 1),
        soc_at_shutdown       = round(soc_at_shutdown, 2),
        voltage_std_inuse     = round(ride["voltage_V"].std(), 2),
        soh_pct               = round(g["state_of_health"].mean(), 2),
        temp_mean             = round(g["ambient_temp_C"].mean(), 1),
    ))

summary = pd.DataFrame(rows).sort_values("drain_rate_pct_per_hr", ascending=False).reset_index(drop=True)

# ── 3. KMeans clustering (unsupervised labelling) ────────────────────────────
print("Clustering …")
features = ["drain_rate_pct_per_hr", "ride_duration_min",
            "soc_at_shutdown", "voltage_std_inuse", "soh_pct"]

scaler = StandardScaler()
X_sum  = scaler.fit_transform(summary[features])

km = KMeans(n_clusters=2, n_init=10, random_state=42)
summary["cluster"] = km.fit_predict(X_sum)
faulty_cluster = summary.groupby("cluster")["drain_rate_pct_per_hr"].mean().idxmax()
summary["flagged"] = summary["cluster"] == faulty_cluster

flagged_ids = summary.loc[summary["flagged"], "battery_id"].tolist()
print(f"  Flagged as FAULTY: {flagged_ids}")

# ── 4. Random Forest on raw readings ─────────────────────────────────────────
print("Training Random Forest …")
label_map = summary.set_index("battery_id")["flagged"]
train_df   = df_clean.copy()
train_df["flagged"] = train_df["battery_id"].map(label_map)
model_features = ["speed_kph", "voltage_V", "current_A", "soc_percent", "ambient_temp_C"]
train_df = train_df.dropna(subset=model_features + ["flagged"])

X = train_df[model_features]
y = train_df["flagged"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=20,
    class_weight="balanced", random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"  Test accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Faulty"]))

# ── 5. Save all artifacts ─────────────────────────────────────────────────────
joblib.dump(clf,    os.path.join(MODELS, "rf_classifier.pkl"))
joblib.dump(km,     os.path.join(MODELS, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))
summary.to_csv(os.path.join(MODELS, "summary.csv"), index=False)
label_map.reset_index().rename(columns={"flagged": "is_faulty"}).to_csv(
    os.path.join(MODELS, "label_map.csv"), index=False
)

print(f"\nAll artifacts saved to  {MODELS}/")
print("  rf_classifier.pkl | kmeans_model.pkl | scaler.pkl | summary.csv | label_map.csv")
