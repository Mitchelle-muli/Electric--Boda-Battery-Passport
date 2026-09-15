"""
screen_fleet.py
---------------
Reusable production screening function.
Takes any KOFA telemetry DataFrame, returns one row per battery with
health metrics and a `flagged` boolean.

Can be run standalone:
    python3 scripts/screen_fleet.py

Or imported by the dashboard or a scheduled job:
    from scripts.screen_fleet import screen_fleet, score_reading
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")


def screen_fleet(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Score a telemetry DataFrame and flag likely-faulty batteries.

    Parameters
    ----------
    telemetry : pd.DataFrame
        Must contain columns:
        timestamp, battery_id, state, speed_kph, voltage_V,
        current_A, soc_percent, state_of_health, ambient_temp_C

    Returns
    -------
    pd.DataFrame  — one row per battery, sorted by drain rate descending.
        Columns: battery_id, drain_rate_pct_per_hr, ride_duration_min,
                 soc_at_shutdown, voltage_std_inuse, soh_pct, cluster, flagged
    """
    tel   = telemetry.sort_values(["battery_id", "timestamp"]).copy()
    ncols = ["speed_kph", "voltage_V", "current_A",
             "soc_percent", "state_of_health", "ambient_temp_C"]

    tel[ncols] = (
        tel.groupby("battery_id")[ncols]
           .apply(lambda g: g.interpolate("linear", limit_direction="both"))
           .reset_index(level=0, drop=True)
    )

    recs = []
    for bid, g in tel.groupby("battery_id"):
        ride = (g[g["state"] == "in_use"]
                  .dropna(subset=["soc_percent"])
                  .sort_values("timestamp"))
        if len(ride) < 3:
            continue
        t_hr = (ride["timestamp"] - ride["timestamp"].iloc[0]).dt.total_seconds() / 3600
        recs.append(dict(
            battery_id            = bid,
            drain_rate_pct_per_hr = -np.polyfit(t_hr, ride["soc_percent"], 1)[0],
            ride_duration_min     = (ride["timestamp"].iloc[-1] - ride["timestamp"].iloc[0]).total_seconds() / 60,
            soc_at_shutdown       = ride["soc_percent"].iloc[-1],
            voltage_std_inuse     = ride["voltage_V"].std(),
            soh_pct               = g["state_of_health"].mean(),
        ))

    out   = pd.DataFrame(recs)
    feats = ["drain_rate_pct_per_hr", "ride_duration_min",
             "soc_at_shutdown", "voltage_std_inuse", "soh_pct"]
    Xs    = StandardScaler().fit_transform(out[feats])
    out["cluster"] = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Xs)
    bad_cluster    = out.groupby("cluster")["drain_rate_pct_per_hr"].mean().idxmax()
    out["flagged"] = out["cluster"] == bad_cluster

    return out.sort_values("drain_rate_pct_per_hr", ascending=False).reset_index(drop=True)


def score_reading(speed_kph: float, voltage_V: float, current_A: float,
                  soc_percent: float, ambient_temp_C: float) -> dict:
    """
    Score a single live sensor reading using the trained Random Forest.
    Returns {'prediction': 'Healthy'|'Faulty', 'probability': float}
    """
    clf = joblib.load(os.path.join(MODELS, "rf_classifier.pkl"))
    X   = pd.DataFrame([{
        "speed_kph": speed_kph, "voltage_V": voltage_V,
        "current_A": current_A, "soc_percent": soc_percent,
        "ambient_temp_C": ambient_temp_C
    }])
    pred  = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0][pred]
    return {"prediction": "Faulty" if pred == 1 else "Healthy", "probability": round(proba, 3)}


# ── Standalone demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA = os.path.join(BASE, "data", "Kore2_battery_performance.csv")
    df   = pd.read_csv(DATA, parse_dates=["timestamp"])
    result = screen_fleet(df)
    print("\n=== Fleet Screen Results ===")
    print(result[["battery_id", "drain_rate_pct_per_hr", "soc_at_shutdown", "soh_pct", "flagged"]]
          .to_string(index=False))
    print(f"\nFlagged for immediate action: {result.loc[result.flagged, 'battery_id'].tolist()}")

    # Single-reading demo
    print("\n=== Single Reading Score (live RF) ===")
    print(score_reading(speed_kph=38, voltage_V=45.2, current_A=30, soc_percent=22, ambient_temp_C=27))
