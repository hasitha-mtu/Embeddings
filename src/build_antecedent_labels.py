r"""
build_antecedent_labels.py  (Component 3: label engineering)

For each station in your finalized catchment shortlist, reads its raw
water-level file and produces a DAILY, dynamic antecedent-state label table --
the "y" side of the X (satellite embedding) / y (state index) pair.

All labels are derived purely from water level -- no discharge/rating-curve
data is required. Percentile rank and standardized anomaly are invariant
under the (monotonic) stage-to-discharge relationship at a given gauge, so
computing them from stage directly is equivalent to computing them from
discharge for labelling purposes.

Labels produced, per station per day:
  - mean_level                  : daily mean water level (m)
  - level_percentile            : this day's rank (0-100) against the
                                   station's own full Sentinel-era record --
                                   the "pre-event discharge percentile" proxy
  - standardized_anomaly        : (value - seasonal mean) / seasonal std,
                                   using a +/-7 day day-of-year climatology
                                   pooled across all available years (SSI-style)
  - baseflow_index              : daily baseflow / daily total level, from a
                                   one-parameter Lyne-Hollick recursive digital
                                   filter (alpha=0.925) applied to the level
                                   series -- naturally a dynamic, day-varying
                                   quantity, not a single whole-record number
  - recession_rate_7d           : slope (m/day) of a linear fit over the
                                   trailing 7 days -- negative = receding
  - days_since_last_peak         : days elapsed since the last prominent
                                    local peak, where "prominent" means the
                                    peak rises at least PEAK_PROMINENCE_PCT_OF_RANGE
                                    (default 3%) of that station's own full-
                                    record range above its surrounding trough
                                    -- scaled per-station so it self-adjusts
                                    across gauges of very different absolute
                                    scale, rather than a fixed metre threshold

RESUME BEHAVIOUR: same pattern as check_station_availability.py -- results
are written incrementally per station and flushed to disk immediately, so an
interrupted run can just be re-launched with the same command.

Usage:
    python build_antecedent_labels.py \
        --station-list catchment_level_shortlist.csv \
        --data-dir "C:\Users\AdikariAdikari\PycharmProjects\Embeddings\dataset\raw\water_level" \
        --out antecedent_state_labels.csv

Requires: pandas, numpy, scipy  (pip install pandas numpy scipy)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

SENTINEL_ERA_START = pd.Timestamp("2017-01-01", tz="UTC")
LYNE_HOLLICK_ALPHA = 0.925  # standard default from the baseflow-separation literature
DOY_WINDOW_DAYS = 7
RECESSION_WINDOW_DAYS = 7

# A candidate peak must rise at least this fraction of the station's own
# full-record (max - min) daily-level range above its surrounding trough to
# count as a real peak. This is relative, not an absolute metre value,
# because stations in this dataset span wildly different scales (some are
# ~0.1-3 m stage gauges, others -- e.g. lake-outlet gauges -- report level
# in mAOD around 46-62 m). A fixed absolute threshold that works for one
# would be meaningless for the other; scaling by each station's own range
# self-adjusts across the whole shortlist.
PEAK_PROMINENCE_PCT_OF_RANGE = 0.03

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",   # WISKI-style
    "%Y-%m-%d %H:%M:%S%z",     # plain-CSV style (your actual format)
]


# --- File reading (same auto-detecting logic as check_station_availability.py) ---

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


def read_station_series(filepath):
    """Returns a DataFrame with columns ['timestamp','value'], Sentinel-era only,
    sorted, deduplicated, for either the plain-CSV or WISKI-style format."""
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
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= SENTINEL_ERA_START]
    df = df[["timestamp", "value"]].dropna(subset=["value"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return df


def find_station_file(data_dir, station_id):
    for pattern in [f"wl_{station_id}.csv", f"{station_id}_wl.csv", f"*{station_id}*.csv"]:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
    return None


# --- Label computation ---

def lyne_hollick_baseflow(level, alpha=LYNE_HOLLICK_ALPHA):
    """One-parameter recursive digital filter (Lyne & Hollick, 1979).
    Returns the baseflow component, same length as `level`, applied forward
    then backward and averaged (standard practice to remove pass-direction bias)."""
    level = level.to_numpy(dtype=float)
    n = len(level)

    def forward_pass(x):
        qf = np.zeros(n)
        for i in range(1, n):
            qf[i] = alpha * qf[i - 1] + (1 + alpha) / 2 * (x[i] - x[i - 1])
            qf[i] = max(qf[i], 0.0)
        bf = x - qf
        return np.clip(bf, 0, x)

    fwd = forward_pass(level)
    bwd = forward_pass(level[::-1])[::-1]
    baseflow = (fwd + bwd) / 2.0
    return np.clip(baseflow, 0, level)


def build_daily_labels(df, station_id):
    daily = df.set_index("timestamp")["value"].resample("1D").mean().rename("mean_level")
    daily = daily.to_frame()
    daily["station_id"] = station_id

    # Pre-event percentile: this day's rank against the station's own Sentinel-era record
    daily["level_percentile"] = daily["mean_level"].rank(pct=True) * 100

    # Standardized seasonal anomaly (+/- DOY_WINDOW_DAYS pooled across all years)
    doy = daily.index.dayofyear
    vals = daily["mean_level"].to_numpy()
    anomaly = np.full(len(daily), np.nan)
    for i, d in enumerate(doy):
        window_doys = [(d + off - 1) % 366 + 1 for off in range(-DOY_WINDOW_DAYS, DOY_WINDOW_DAYS + 1)]
        mask = np.isin(doy, window_doys)
        window_vals = vals[mask]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) >= 5 and window_vals.std() > 0:
            anomaly[i] = (vals[i] - window_vals.mean()) / window_vals.std()
    daily["standardized_anomaly"] = anomaly

    # Baseflow index (dynamic, from the digital filter)
    filled = daily["mean_level"].interpolate(limit=3)  # small gaps only, avoid over-fabricating
    valid = filled.notna()
    baseflow = pd.Series(np.nan, index=daily.index)
    if valid.sum() > 10:
        bf_vals = lyne_hollick_baseflow(filled[valid])
        baseflow.loc[valid] = bf_vals
    with np.errstate(divide="ignore", invalid="ignore"):
        daily["baseflow_index"] = (baseflow / filled).clip(0, 1)

    # Recession rate: trailing RECESSION_WINDOW_DAYS-day linear slope (m/day)
    def trailing_slope(s):
        y = s.to_numpy()
        if np.isnan(y).sum() > len(y) // 2:
            return np.nan
        x = np.arange(len(y))
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return np.nan
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return slope

    daily["recession_rate_7d"] = (
        daily["mean_level"].rolling(RECESSION_WINDOW_DAYS, min_periods=4).apply(trailing_slope, raw=False)
    )

    # Days since last peak, using scipy's topographic-prominence peak detector
    # rather than a plain "higher than immediate neighbours" check -- the
    # latter fires on any noise-level wiggle, which showed up in real data as
    # a spurious "peak" flagged on a ~0.15 m ripple sitting next to a 12 m
    # flood event at one station. Prominence requires a peak to actually
    # stand out from its surrounding trough by a meaningful amount.
    level_series = daily["mean_level"]
    level_range = level_series.max() - level_series.min()
    prominence_threshold = max(level_range * PEAK_PROMINENCE_PCT_OF_RANGE, 1e-6)

    valid = level_series.notna()
    days_since_peak = np.full(len(level_series), np.nan)
    if valid.sum() > 5:
        valid_vals = level_series[valid].to_numpy()
        peak_positions_within_valid, _ = find_peaks(valid_vals, prominence=prominence_threshold)
        # map positions-within-the-valid-subset back to full-array indices
        valid_full_idx = np.flatnonzero(valid.to_numpy())
        peak_full_idx = valid_full_idx[peak_positions_within_valid]

        is_peak = np.zeros(len(level_series), dtype=bool)
        is_peak[peak_full_idx] = True

        last_peak_idx = None
        for i in range(len(level_series)):
            if is_peak[i]:
                last_peak_idx = i
            if last_peak_idx is not None:
                days_since_peak[i] = i - last_peak_idx
    daily["days_since_last_peak"] = days_since_peak

    daily = daily.reset_index().rename(columns={"timestamp": "date"})
    daily["date"] = daily["date"].dt.date
    return daily[[
        "station_id", "date", "mean_level", "level_percentile", "standardized_anomaly",
        "baseflow_index", "recession_rate_7d", "days_since_last_peak",
    ]]


def load_done_stations(out_path):
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    try:
        prior = pd.read_csv(out_path, usecols=["station_id"])
    except Exception:
        return set()
    return set(prior["station_id"].astype(str).unique())


def main():
    parser = argparse.ArgumentParser(description="Build dynamic antecedent-state labels from water-level records.")
    parser.add_argument("--station-list", required=True,
                         help="CSV containing a 'station_id' column (e.g. catchment_level_shortlist.csv)")
    parser.add_argument("--data-dir", required=True, help="Directory containing the raw station CSV files")
    parser.add_argument("--out", default="antecedent_state_labels.csv")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    stations = pd.read_csv(args.station_list)
    if "station_id" not in stations.columns:
        print("Error: --station-list must contain a 'station_id' column.", file=sys.stderr)
        sys.exit(1)
    station_ids = stations["station_id"].dropna().astype(int).astype(str).unique().tolist()

    if args.restart and os.path.exists(args.out):
        os.remove(args.out)

    done = load_done_stations(args.out)
    todo = [sid for sid in station_ids if sid not in done]
    print(f"{len(station_ids)} stations in shortlist. {len(done)} already done, {len(todo)} remaining.")

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0

    for i, sid in enumerate(todo, 1):
        filepath = find_station_file(args.data_dir, sid)
        if filepath is None:
            print(f"  [{i}/{len(todo)}] station {sid}: no matching file found in {args.data_dir} -- skipped")
            continue
        try:
            df = read_station_series(filepath)
            if len(df) < 30:
                print(f"  [{i}/{len(todo)}] station {sid}: too few Sentinel-era readings ({len(df)}) -- skipped")
                continue
            daily = build_daily_labels(df, sid)
            daily.to_csv(args.out, mode="a", header=write_header, index=False)
            write_header = False
            print(f"  [{i}/{len(todo)}] station {sid}: {len(daily)} daily labels written")
        except Exception as e:
            print(f"  [{i}/{len(todo)}] station {sid}: ERROR -- {e}")

    print(f"\nDone. Labels written to: {args.out}")


if __name__ == "__main__":
    main()
