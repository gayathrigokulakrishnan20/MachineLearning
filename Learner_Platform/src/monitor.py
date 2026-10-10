"""Basic feature-drift screening tool; not a full production monitor."""
import argparse
import json
from pathlib import Path
import pandas as pd

FEATURES = ["study_hours_per_week", "quiz_score_pct", "days_since_last_login",
            "lessons_completed", "courses_started"]

def standardized_mean_difference(reference, current):
    ref_std, cur_std = reference.std(ddof=1), current.std(ddof=1)
    pooled = ((ref_std ** 2 + cur_std ** 2) / 2) ** 0.5
    if pd.isna(pooled) or pooled == 0:
        return 0.0
    return float((current.mean() - reference.mean()) / pooled)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", default="monitoring_report.json")
    args = parser.parse_args()
    reference, current = pd.read_csv(args.reference), pd.read_csv(args.current)
    report = {"threshold": args.threshold, "features": {}, "note":
              "A flag is a screening signal; investigate data quality, business changes and model performance."}
    for feature in FEATURES:
        if feature not in reference or feature not in current:
            continue
        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()
        if ref.empty or cur.empty:
            report["features"][feature] = {"investigate_drift": True,
                                           "note": "Insufficient valid values; investigate data quality."}
            continue
        smd = standardized_mean_difference(ref, cur)
        report["features"][feature] = {
            "reference_rows": int(len(ref)), "current_rows": int(len(cur)),
            "reference_mean": round(float(ref.mean()), 4),
            "current_mean": round(float(cur.mean()), 4),
            "standardized_mean_difference": round(smd, 4),
            "investigate_drift": abs(smd) >= args.threshold
        }
    if not report["features"]:
        raise ValueError("No shared monitored features found in the two CSV files.")
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
