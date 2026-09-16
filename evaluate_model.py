"""
Kore2 Battery SOH — Model Evaluation Script
=============================================

Standalone evaluation harness for the CRISP-DM analysis in
`Kore2_Battery_SOH_CRISPDM.ipynb`.

It rebuilds the two models used in the notebook:
  1. Unsupervised  — K-Means (k=2) on engineered per-battery health features,
                      used to separate "healthy" vs "faulty" batteries.
  2. Supervised    — Random Forest classifier trained on raw per-reading
                      telemetry, using the K-Means labels as ground truth,
                      for real-time fault scoring.

...and runs a dedicated EVALUATION suite on both:
  - Clustering: silhouette score, inertia, gap statistic (largest vs.
    second-largest gap in sorted drain rate), cluster-size sanity check.
  - Confound check: does ambient temperature explain the split? (t-test)
  - Classifier: stratified train/test split, accuracy, precision/recall/F1,
    confusion matrix, ROC-AUC, 5-fold cross-validated F1, feature importances.
  - Naive baseline comparison: shows why a simple z-score rule fails,
    to justify the modelling choice.

Usage
-----
    python evaluate_kore2_battery_model.py \
        --csv Kore2_battery_performance.csv \
        --outdir ./eval_outputs

Outputs
-------
A metrics summary printed to stdout, a JSON report, and diagnostic PNG plots
written to --outdir.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for script use
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

NORMAL_COLOR = "#4C72B0"
FLAG_COLOR = "#C44E52"

NUM_COLS = ["speed_kph", "voltage_V", "current_A", "soc_percent",
            "state_of_health", "ambient_temp_C"]
CLUSTER_FEATURES = ["drain_rate_pct_per_hr", "ride_duration_min",
                     "soc_at_shutdown", "voltage_std_inuse", "soh_pct"]
MODEL_FEATURES = ["speed_kph", "voltage_V", "current_A", "soc_percent", "ambient_temp_C"]


# --------------------------------------------------------------------------
# Data loading & preparation (mirrors the notebook's Data Prep section)
# --------------------------------------------------------------------------

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["battery_id", "timestamp"]).reset_index(drop=True)

    df_clean = df.copy()
    df_clean[NUM_COLS] = (
        df_clean.groupby("battery_id")[NUM_COLS]
        .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
        .reset_index(level=0, drop=True)
    )
    remaining_na = int(df_clean[NUM_COLS].isnull().sum().sum())
    if remaining_na:
        print(f"[warn] {remaining_na} missing values remain after interpolation; "
              f"dropping those rows.")
        df_clean = df_clean.dropna(subset=NUM_COLS)
    return df_clean


def build_battery_summary(df_clean: pd.DataFrame) -> pd.DataFrame:
    """One row per battery with the engineered health metrics."""
    rows = []
    for bid, g in df_clean.groupby("battery_id"):
        ride = (g[g["state"] == "in_use"]
                .dropna(subset=["soc_percent", "timestamp"])
                .sort_values("timestamp"))
        if len(ride) < 3:
            print(f"[warn] battery {bid} has <3 in-use readings; skipping.")
            continue

        t_hr = (ride["timestamp"] - ride["timestamp"].iloc[0]).dt.total_seconds() / 3600
        drain_rate = -np.polyfit(t_hr, ride["soc_percent"], 1)[0]
        ride_min = (ride["timestamp"].iloc[-1] - ride["timestamp"].iloc[0]).total_seconds() / 60
        soc_at_shutdown = ride["soc_percent"].iloc[-1]

        rows.append(dict(
            battery_id=bid,
            drain_rate_pct_per_hr=round(drain_rate, 2),
            ride_duration_min=round(ride_min, 1),
            soc_at_shutdown=round(soc_at_shutdown, 2),
            voltage_std_inuse=round(ride["voltage_V"].std(), 2),
            soh_pct=round(g["state_of_health"].mean(), 2),
            temp_mean=round(g["ambient_temp_C"].mean(), 1),
        ))
    return pd.DataFrame(rows).sort_values("drain_rate_pct_per_hr", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def run_naive_zscore_baseline(summary: pd.DataFrame, threshold: float = 2.0) -> dict:
    """Reproduce the naive z-score check to show why it fails under contamination."""
    dr = summary["drain_rate_pct_per_hr"]
    z = (dr - dr.mean()) / dr.std()
    flagged = summary.loc[z > threshold, "battery_id"].tolist()
    return {
        "max_zscore": round(float(z.max()), 2),
        "threshold": threshold,
        "flagged_by_zscore": flagged,
        "verdict": "no batteries flagged — outliers inflate mean/std" if not flagged else "flagged some batteries",
    }


def run_kmeans(summary: pd.DataFrame, k: int = 2, random_state: int = 42):
    X = StandardScaler().fit_transform(summary[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    summary = summary.copy()
    summary["cluster"] = labels

    faulty_cluster = summary.groupby("cluster")["drain_rate_pct_per_hr"].mean().idxmax()
    summary["flagged"] = summary["cluster"] == faulty_cluster

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    return summary, km, X, sil


def train_classifier(df_clean: pd.DataFrame, label_map: pd.Series, random_state: int = 42):
    train_df = df_clean.copy()
    train_df["flagged"] = train_df["battery_id"].map(label_map)
    train_df = train_df.dropna(subset=MODEL_FEATURES + ["flagged"])

    X = train_df[MODEL_FEATURES]
    y = train_df["flagged"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20,
        class_weight="balanced", random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf, X, y, X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_clustering(summary: pd.DataFrame, sil: float) -> dict:
    sorted_dr = np.sort(summary["drain_rate_pct_per_hr"].values)
    gaps = np.diff(sorted_dr)
    largest_gap = float(gaps.max())
    second_largest_gap = float(np.sort(gaps)[-2]) if len(gaps) > 1 else float("nan")

    healthy = summary.loc[~summary["flagged"]]
    faulty = summary.loc[summary["flagged"]]

    # Confound check: does temperature explain the split?
    if len(healthy) > 1 and len(faulty) > 1:
        t_stat, p_val = stats.ttest_ind(healthy["temp_mean"], faulty["temp_mean"], equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")

    return {
        "n_healthy": int(len(healthy)),
        "n_faulty": int(len(faulty)),
        "silhouette_score": round(float(sil), 3) if sil == sil else None,
        "largest_gap_pct_per_hr": round(largest_gap, 1),
        "second_largest_gap_pct_per_hr": round(second_largest_gap, 1),
        "gap_ratio": round(largest_gap / second_largest_gap, 2) if second_largest_gap else None,
        "clean_bimodal_split": bool(largest_gap > 2 * second_largest_gap) if second_largest_gap == second_largest_gap else None,
        "temp_mean_healthy": round(float(healthy["temp_mean"].mean()), 1) if len(healthy) else None,
        "temp_mean_faulty": round(float(faulty["temp_mean"].mean()), 1) if len(faulty) else None,
        "temp_ttest_p_value": round(float(p_val), 3) if p_val == p_val else None,
        "temp_confound_ruled_out": bool(p_val > 0.05) if p_val == p_val else None,
        "flagged_battery_ids": faulty["battery_id"].tolist(),
    }


def evaluate_classifier(clf, X, y, X_train, X_test, y_train, y_test, outdir: Path) -> dict:
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")

    # 5-fold stratified cross-validation on the full dataset for a more robust estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X, y, cv=cv, scoring="f1")

    report_txt = classification_report(y_test, y_pred, target_names=["Healthy", "Faulty"])
    print("\nClassification report (held-out test set):")
    print(report_txt)

    # --- plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Faulty"], yticklabels=["Healthy", "Faulty"], ax=axes[0])
    axes[0].set_title("Confusion Matrix (test set)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    importances = pd.Series(clf.feature_importances_, index=MODEL_FEATURES).sort_values()
    axes[1].barh(importances.index, importances.values, color=NORMAL_COLOR)
    axes[1].set_title("Feature Importance")
    axes[1].set_xlabel("Importance")

    if auc == auc:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[2].plot(fpr, tpr, color=FLAG_COLOR, lw=2, label=f"AUC = {auc:.3f}")
        axes[2].plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
        axes[2].set_title("ROC Curve (test set)")
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].legend(fontsize=9, frameon=False)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plot_path = outdir / "classifier_evaluation.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {plot_path}")

    return {
        "test_accuracy": round(float(acc), 3),
        "test_precision_faulty": round(float(precision), 3),
        "test_recall_faulty": round(float(recall), 3),
        "test_f1_faulty": round(float(f1), 3),
        "test_roc_auc": round(float(auc), 3) if auc == auc else None,
        "confusion_matrix": cm.tolist(),
        "cv_f1_mean": round(float(cv_f1.mean()), 3),
        "cv_f1_std": round(float(cv_f1.std()), 3),
        "cv_f1_per_fold": [round(float(s), 3) for s in cv_f1],
        "feature_importances": importances.round(3).to_dict(),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "classification_report": report_txt,
    }


def plot_cluster_evidence(summary: pd.DataFrame, df_clean: pd.DataFrame, outdir: Path):
    flagged_ids = summary.loc[summary["flagged"], "battery_id"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = summary.sort_values("drain_rate_pct_per_hr", ascending=False)
    colors = [FLAG_COLOR if f else NORMAL_COLOR for f in plot_df["flagged"]]
    healthy = summary.loc[~summary["flagged"], "drain_rate_pct_per_hr"]
    limit = healthy.mean() + 2 * healthy.std() if len(healthy) > 1 else np.nan

    ax = axes[0]
    ax.bar(plot_df["battery_id"], plot_df["drain_rate_pct_per_hr"], color=colors, width=0.65)
    if limit == limit:
        ax.axhline(limit, color="#555", ls="--", lw=1.2,
                    label=f"Healthy control limit ({limit:.0f} %/hr)")
        ax.legend(fontsize=9, frameon=False)
    ax.set_ylabel("Drain rate (% SOC / hour)")
    ax.set_title("Battery Drain Rate While Riding")
    ax.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    for bid, g in df_clean.groupby("battery_id"):
        ride = g[g["state"] == "in_use"].dropna(subset=["soc_percent"]).sort_values("timestamp")
        if ride.empty:
            continue
        t = (ride["timestamp"] - df_clean["timestamp"].min()).dt.total_seconds() / 60
        is_flag = bid in flagged_ids
        ax2.plot(t, ride["soc_percent"],
                  color=FLAG_COLOR if is_flag else NORMAL_COLOR,
                  lw=2.2 if is_flag else 1.2, alpha=0.95 if is_flag else 0.5)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([0], [0], color=NORMAL_COLOR, lw=2, label="Healthy"),
                         Line2D([0], [0], color=FLAG_COLOR, lw=2, label="Faulty")],
               fontsize=9, frameon=False)
    ax2.set_xlabel("Minutes since data start")
    ax2.set_ylabel("State of Charge (%)")
    ax2.set_title("SOC Trajectory While Riding")

    plt.tight_layout()
    plot_path = outdir / "cluster_evidence.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {plot_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Kore2 battery SOH models.")
    parser.add_argument("--csv", type=str, default="Kore2_battery_performance.csv",
                         help="Path to the Kore2 telemetry CSV.")
    parser.add_argument("--outdir", type=str, default="./eval_outputs",
                         help="Directory to write plots and the JSON report.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and cleaning: {csv_path}")
    df_clean = load_and_clean(csv_path)

    print("Building per-battery health summary...")
    summary = build_battery_summary(df_clean)
    print(summary.to_string(index=False))

    print("\n--- Naive baseline (z-score) ---")
    baseline = run_naive_zscore_baseline(summary)
    print(json.dumps(baseline, indent=2))

    print("\n--- Unsupervised model: K-Means (k=2) ---")
    summary, km, X_cluster, sil = run_kmeans(summary)
    cluster_eval = evaluate_clustering(summary, sil)
    print(json.dumps(cluster_eval, indent=2))
    plot_cluster_evidence(summary, df_clean, outdir)

    print("\n--- Supervised model: Random Forest classifier ---")
    label_map = summary.set_index("battery_id")["flagged"]
    clf, X, y, X_train, X_test, y_train, y_test = train_classifier(df_clean, label_map)
    classifier_eval = evaluate_classifier(clf, X, y, X_train, X_test, y_train, y_test, outdir)

    report = {
        "naive_baseline": baseline,
        "clustering_evaluation": cluster_eval,
        "classifier_evaluation": classifier_eval,
    }
    report_path = outdir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[saved] {report_path}")

    print("\n=== SUMMARY ===")
    print(f"Flagged faulty batteries: {cluster_eval['flagged_battery_ids']}")
    print(f"Cluster separation (silhouette): {cluster_eval['silhouette_score']}")
    print(f"Temp confound ruled out (p>0.05): {cluster_eval['temp_confound_ruled_out']}")
    print(f"Classifier test accuracy: {classifier_eval['test_accuracy']}, "
          f"F1 (faulty): {classifier_eval['test_f1_faulty']}, "
          f"ROC-AUC: {classifier_eval['test_roc_auc']}")
    print(f"Cross-validated F1: {classifier_eval['cv_f1_mean']} ± {classifier_eval['cv_f1_std']}")


if __name__ == "__main__":
    main()
