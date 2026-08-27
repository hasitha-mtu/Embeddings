r"""
summarize_labels.py

Produces a small, per-station diagnostic summary of antecedent_state_labels.csv
(too large to upload directly) -- value-range sanity checks, stuck-sensor
detection, peak counts, and NaN counts -- so data-quality issues can be spotted
across all stations without inspecting the full file row by row.

Usage:
    python summarize_labels.py --labels antecedent_state_labels.csv --out labels_summary.csv

Requires: pandas, numpy  (pip install pandas numpy)
"""

import argparse
import sys

import numpy as np
import pandas as pd

STUCK_RUN_THRESHOLD_DAYS = 14  # flag runs of this many identical consecutive values or more


def longest_stuck_run(values):
    """Longest run of consecutive, (near-)identical daily values -- a sign of a
    sensor stuck at one reading rather than a genuinely flat/dry period."""
    values = values.to_numpy()
    if len(values) < 2:
        return 0
    same = np.isclose(values[1:], values[:-1], atol=1e-6, equal_nan=False)
    longest = run = 1
    for s in same:
        run = run + 1 if s else 1
        longest = max(longest, run)
    return longest


def summarize_station(g):
    row = {}
    row["n_days"] = len(g)
    row["date_min"] = g["date"].min()
    row["date_max"] = g["date"].max()

    full_range_days = (pd.Timestamp(g["date"].max()) - pd.Timestamp(g["date"].min())).days + 1
    row["n_missing_days"] = full_range_days - g["date"].nunique()

    row["mean_level_min"] = g["mean_level"].min()
    row["mean_level_max"] = g["mean_level"].max()
    row["longest_stuck_run_days"] = longest_stuck_run(g.sort_values("date")["mean_level"].dropna())

    pct = g["level_percentile"]
    row["level_percentile_out_of_range"] = int(((pct < 0) | (pct > 100)).sum())

    bfi = g["baseflow_index"]
    row["baseflow_index_out_of_range"] = int(((bfi < -1e-6) | (bfi > 1 + 1e-6)).sum())
    row["baseflow_index_nan_pct"] = round(100 * bfi.isna().mean(), 2)

    anomaly = g["standardized_anomaly"]
    row["standardized_anomaly_nan_pct"] = round(100 * anomaly.isna().mean(), 2)
    if anomaly.notna().any():
        max_idx = anomaly.abs().idxmax()
        row["standardized_anomaly_max_abs"] = anomaly.abs().max()
        row["standardized_anomaly_max_date"] = g.loc[max_idx, "date"]
        row["standardized_anomaly_max_signed_value"] = anomaly.loc[max_idx]
    else:
        row["standardized_anomaly_max_abs"] = np.nan
        row["standardized_anomaly_max_date"] = pd.NaT
        row["standardized_anomaly_max_signed_value"] = np.nan
    row["standardized_anomaly_extreme_count"] = int((anomaly.abs() > 8).sum())  # |z|>8 is suspiciously extreme

    row["recession_rate_nan_pct"] = round(100 * g["recession_rate_7d"].isna().mean(), 2)

    n_peaks = int((g["days_since_last_peak"] == 0).sum())
    row["n_peaks_detected"] = n_peaks
    row["days_per_peak"] = round(row["n_days"] / n_peaks, 1) if n_peaks > 0 else np.nan

    return pd.Series(row)


def main():
    parser = argparse.ArgumentParser(description="Summarize antecedent_state_labels.csv per station.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="labels_summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.labels, parse_dates=["date"])
    if "station_id" not in df.columns:
        print("Error: labels file must contain a 'station_id' column.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} rows across {df['station_id'].nunique()} stations.")

    summary = df.groupby("station_id", group_keys=False).apply(summarize_station, include_groups=False)
    summary = summary.reset_index()
    summary.to_csv(args.out, index=False)

    print(f"\nSummary written to: {args.out}  ({len(summary)} rows -- one per station, safe to upload)")

    print("\n=== Flags worth checking ===")
    flags = summary[
        (summary["level_percentile_out_of_range"] > 0)
        | (summary["baseflow_index_out_of_range"] > 0)
        | (summary["longest_stuck_run_days"] >= STUCK_RUN_THRESHOLD_DAYS)
        | (summary["standardized_anomaly_extreme_count"] > 0)
    ]
    if flags.empty:
        print("None -- all stations passed the basic range/stuck-sensor checks.")
    else:
        cols = ["station_id", "n_days", "level_percentile_out_of_range", "baseflow_index_out_of_range",
                "longest_stuck_run_days", "standardized_anomaly_extreme_count"]
        print(flags[cols].to_string(index=False))


if __name__ == "__main__":
    main()
