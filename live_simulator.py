import os, sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This file lives in backend/; models/ and data/ are siblings inside
# backend/, not at the project root.
BASE   = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")

# Original 12 batteries
ORIGINAL_FAULTY  = ["KF-B101", "KF-I108", "KF-J109", "KF-L111"]
ORIGINAL_HEALTHY = ["KF-A100", "KF-C102", "KF-D103", "KF-E104",
                    "KF-F105", "KF-G106", "KF-H107", "KF-K110"]

# 8 new simulated batteries
NEW_FAULTY  = ["KF-M112", "KF-N113", "KF-O114"]
NEW_HEALTHY = ["KF-P115", "KF-Q116", "KF-R117", "KF-S118", "KF-T119"]

ALL_FAULTY   = ORIGINAL_FAULTY + NEW_FAULTY
ALL_HEALTHY  = ORIGINAL_HEALTHY + NEW_HEALTHY
ALL_BATTERIES = ALL_HEALTHY + ALL_FAULTY

# Simulate 7 days
SCAN_DAYS = [
    (datetime(2024, 1, 15) + timedelta(days=i)).strftime("%Y-%m-%d")
    for i in range(7)
]

def generate_live_reading(battery_id, seed=None, day_index=0):
    rng = np.random.default_rng(seed)
    is_faulty = battery_id in ALL_FAULTY
    # Batteries degrade slightly more each day
    degradation = day_index * 0.3
    if is_faulty:
        speed   = max(0, rng.normal(34, 3))
        voltage = rng.normal(45.0 - degradation * 0.1, 0.8)
        current = rng.normal(30 + degradation * 0.1, 2)
        soc     = max(0, min(100, rng.normal(20 - degradation * 0.2, 3)))
        soh     = rng.normal(67 - degradation * 0.2, 1)
        temp    = rng.normal(28, 1)
    else:
        speed   = max(0, rng.normal(35, 3))
        voltage = rng.normal(51.0, 0.3)
        current = rng.normal(17, 1)
        soc     = max(0, min(100, rng.normal(75, 5)))
        soh     = rng.normal(91, 1)
        temp    = rng.normal(27, 1)
    return {
        "battery_id":      battery_id,
        "speed_kph":       round(speed,   2),
        "voltage_V":       round(voltage, 2),
        "current_A":       round(current, 2),
        "soc_percent":     round(soc,     2),
        "state_of_health": round(soh,     2),
        "ambient_temp_C":  round(temp,    2),
    }

def score_live_fleet(seed=None, day_index=0, scan_date=None):
    clf      = joblib.load(os.path.join(MODELS, "rf_classifier.pkl"))
    features = ["speed_kph", "voltage_V", "current_A", "soc_percent", "ambient_temp_C"]
    rng_seed = seed if seed is not None else np.random.randint(0, 99999)
    if scan_date is None:
        scan_date = SCAN_DAYS[day_index % len(SCAN_DAYS)]
    records = []
    for i, bid in enumerate(ALL_BATTERIES):
        reading = generate_live_reading(bid, seed=rng_seed + i, day_index=day_index)
        X       = pd.DataFrame([{f: reading[f] for f in features}])
        pred    = clf.predict(X)[0]
        proba   = clf.predict_proba(X)[0]
        records.append({
            "scan_date":      scan_date,
            "battery_id":     bid,
            "speed_kph":      reading["speed_kph"],
            "voltage_V":      reading["voltage_V"],
            "current_A":      reading["current_A"],
            "soc_percent":    reading["soc_percent"],
            "soh_percent":    reading["state_of_health"],
            "ambient_temp_C": reading["ambient_temp_C"],
            "prediction":     "Faulty" if pred == 1 else "Healthy",
            "confidence":     round(float(proba[pred]), 3),
            "flagged":        bool(pred == 1),
        })
    return pd.DataFrame(records).sort_values("flagged", ascending=False).reset_index(drop=True)

def generate_all_days():
    all_records = []
    for i, day in enumerate(SCAN_DAYS):
        seed    = 1000 + i * 100
        results = score_live_fleet(seed=seed, day_index=i, scan_date=day)
        all_records.append(results)
    return pd.concat(all_records, ignore_index=True)

if __name__ == "__main__":
    print("Generating live readings for 20 batteries across 7 days ...\n")
    results = score_live_fleet(day_index=0)
    print(results[["battery_id","voltage_V","soc_percent","prediction","confidence"]].to_string(index=False))
    print(f"\nFaulty:  {results[results.flagged].battery_id.tolist()}")
    print(f"Healthy: {results[~results.flagged].battery_id.tolist()}")
    print(f"\nTotal batteries: {len(results)}")
