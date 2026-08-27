r"""
cross_check_labels.py

Visual sanity-check tool: plots a station's raw sub-daily water-level trace
alongside the daily labels computed by build_antecedent_labels.py, so you can
eyeball whether percentile/baseflow/recession/peak-detection are behaving
sensibly around a specific event -- the same kind of check done by hand for
station 11001's early-Feb-2017 double peak.

Also numerically re-derives the daily mean from the raw data and compares it
against the labels file's mean_level for the same window, to catch any
pipeline drift between the two (they should match almost exactly).

Usage:
    python cross_check_labels.py \
        --data-dir "C:\Users\AdikariAdikari\PycharmProjects\Embeddings\dataset\raw\water_level" \
        --labels antecedent_state_labels.csv \
        --station-id 11001 \
        --start 2017-01-25 --end 2017-02-10 \
        --out check_11001.png

Requires: pandas, numpy, matplotlib  (pip install pandas numpy matplotlib)
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%d %H:%M:%S%z",
]


def parse_timestamps(raw_series):
    ts = pd.to_datetime(pd.Series(pd.NaT, index=raw_series.index), utc=True)
    remaining = pd.Series(True, index=raw_series.index)
    for fmt in TIMESTAMP_FORMATS:
        if not remaining.any():
            break
        parsed = pd.to_datetime(raw_series[remaining], format=fmt, utc=True, errors="coerce")
        ts.loc[remaining] = parsed
        remaining = ts.isna()
    if remaining.any():
        parsed = pd.to_datetime(raw_series[remaining], utc=True, errors="coerce")
        ts.loc[remaining] = parsed
    return ts


def read_raw_series(filepath):
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.startswith("#"):
            for _ in range(30):
                pos = f.tell()
                line = f.readline()
                if not line or not line.startswith("#"):
                    f.seek(pos)
                    break
            df = pd.read_csv(
                f, sep=";", names=["timestamp_raw", "value", "quality_code"],
                header=None, dtype={"value": "float64"},
                na_values=["", "NaN", "nan"], low_memory=False,
            )
        else:
            df = pd.read_csv(f, sep=",", header=0, na_values=["", "NaN", "nan"], low_memory=False)
            df = df.rename(columns={"datetime": "timestamp_raw"})
    df["timestamp"] = parse_timestamps(df["timestamp_raw"].astype(str))
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    return df[["timestamp", "value"]]


def find_station_file(data_dir, station_id):
    for pattern in [f"wl_{station_id}.csv", f"{station_id}_wl.csv", f"*{station_id}*.csv"]:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Cross-check raw hydrograph against daily antecedent-state labels.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the raw station CSV files")
    parser.add_argument("--labels", required=True, help="antecedent_state_labels.csv")
    parser.add_argument("--station-id", required=True)
    parser.add_argument("--start", required=True, help="e.g. 2017-01-25")
    parser.add_argument("--end", required=True, help="e.g. 2017-02-10")
    parser.add_argument("--out", default=None, help="Output PNG path (default: check_<station_id>.png)")
    args = parser.parse_args()

    out_path = args.out or f"check_{args.station_id}.png"
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)

    filepath = find_station_file(args.data_dir, args.station_id)
    if filepath is None:
        print(f"No raw file found for station {args.station_id} in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    raw = read_raw_series(filepath)
    raw_window = raw[(raw["timestamp"] >= start) & (raw["timestamp"] < end)]
    if raw_window.empty:
        print(f"No raw readings for station {args.station_id} in [{args.start}, {args.end}]", file=sys.stderr)
        sys.exit(1)

    # Independently re-derive the daily mean from raw data, for comparison against the labels file
    recomputed_daily = (
        raw_window.set_index("timestamp")["value"].resample("1D").mean().rename("recomputed_mean_level")
    )

    labels = pd.read_csv(args.labels, parse_dates=["date"])
    labels["date"] = pd.to_datetime(labels["date"], utc=True)
    station_labels = labels[
        (labels["station_id"].astype(str) == str(args.station_id))
        & (labels["date"] >= start) & (labels["date"] < end)
    ].sort_values("date").set_index("date")

    if station_labels.empty:
        print(f"No label rows found for station {args.station_id} in this window -- "
              f"check the station_id and date range against your labels file.", file=sys.stderr)
        sys.exit(1)

    # Numerical cross-check: recomputed daily mean vs the labels file's mean_level
    compare = pd.DataFrame({
        "recomputed_mean_level": recomputed_daily,
        "labels_mean_level": station_labels["mean_level"],
    }).dropna()
    compare["abs_diff"] = (compare["recomputed_mean_level"] - compare["labels_mean_level"]).abs()
    max_diff = compare["abs_diff"].max() if not compare.empty else float("nan")
    print(f"Numerical cross-check over {len(compare)} overlapping days: "
          f"max |recomputed - labels| = {max_diff:.6f}")
    if not compare.empty and max_diff > 1e-3:
        print("  -> difference is larger than expected floating-point noise; "
              "worth checking whether the labels file was built from a different/older raw data pull.")
    print(compare.to_string())

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    ax = axes[0]
    ax.plot(raw_window["timestamp"], raw_window["value"], color="#7fb3d5", linewidth=0.8, label="raw (sub-daily)")
    ax.plot(station_labels.index, station_labels["mean_level"], color="#1a5276", marker="o",
             markersize=4, linewidth=1.5, label="daily mean_level (labels file)")
    peaks = station_labels[station_labels["days_since_last_peak"] == 0]
    if not peaks.empty:
        ax.scatter(peaks.index, peaks["mean_level"], color="#c0392b", zorder=5, s=60,
                    marker="^", label="detected peak")
    ax.set_ylabel("Water level (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Station {args.station_id}: raw hydrograph vs. daily labels")

    ax = axes[1]
    ax.plot(station_labels.index, station_labels["level_percentile"], color="#8e44ad", marker="o", markersize=3)
    ax.axhline(50, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("level_percentile")
    ax.set_ylim(-5, 105)

    ax = axes[2]
    ax.plot(station_labels.index, station_labels["baseflow_index"], color="#16a085", marker="o", markersize=3)
    ax.set_ylabel("baseflow_index")
    ax.set_ylim(-0.05, 1.05)

    ax = axes[3]
    ax.plot(station_labels.index, station_labels["recession_rate_7d"], color="#d35400", marker="o", markersize=3)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("recession_rate_7d (m/day)")
    ax.set_xlabel("Date")

    for a in axes:
        a.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot written to: {out_path}")


if __name__ == "__main__":
    main()
