r"""
Water level station data availability checker (resumable).

Auto-detects TWO possible file formats per station file:

  1. Plain CSV (confirmed real format):
         datetime,value,quality_code,quality_ok,station_ref
         2021-11-16 12:15:00+00:00,72.226,254,True,10051

  2. WISKI-style export, in case some files differ:
         #station_name;Ovens Bridge
         ... more '#key;value' metadata lines ...
         #Timestamp;Value;Quality Code
         2019-07-15T14:15:00.000Z;21.116;254

Produces one summary CSV with per-station coverage statistics -- date range,
completeness within the Sentinel-1/2 era (2017-present), largest data gap,
and quality-code/quality_ok distribution -- to support catchment selection.

RESUME BEHAVIOUR:
Results are written to the output CSV one row at a time, flushed to disk
immediately after each file. On startup, the script reads any existing
output CSV, skips files that previously finished with status "ok", and
retries files that previously errored, were empty, or hadn't been reached
yet. So if the run is interrupted, just re-run the exact same command --
it picks up where it left off rather than starting over.

This script only reads local files and writes one small summary CSV. It does
not need to be uploaded anywhere; run it locally and upload the resulting
summary CSV instead.

Usage:
    python check_station_availability.py --data-dir "C:\Users\AdikariAdikari\PycharmProjects\Embeddings\dataset\raw\water_level" --out station_availability_summary.csv

    # Force a clean re-run from scratch, ignoring any existing output:
    python check_station_availability.py --data-dir "..." --out station_availability_summary.csv --restart

Requires: pandas, numpy  (pip install pandas numpy)
"""

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

SENTINEL_ERA_START = pd.Timestamp("2017-01-01", tz="UTC")
EXPECTED_INTERVAL_MINUTES = 15

# Only relevant for the WISKI-style '#key;value' header block, if present.
METADATA_KEYS = {
    "station_name": "station_name",
    "station_no": "station_no",
    "station_latitude": "latitude",
    "station_longitude": "longitude",
    "stationparameter_name": "parameter",
    "ts_shortname": "ts_shortname",
    "ts_unitsymbol": "unit",
    "WTO_OBJECT": "river",
    "RBD": "rbd",
    "rows": "declared_rows",
}

# Fixed output schema -- every row written has exactly these columns, in this
# order, regardless of status, so incremental appends never produce a
# ragged/misaligned CSV.
ALL_COLUMNS = [
    "station_id", "station_name", "river", "rbd", "latitude", "longitude",
    "status", "usable_for_study", "file_format",
    "record_start", "record_end", "record_span_days", "total_rows",
    "sentinel_era_start", "sentinel_era_end", "sentinel_era_span_days",
    "sentinel_era_rows", "sentinel_era_interval_minutes", "sentinel_era_completeness_pct",
    "sentinel_era_quality_ok_pct",
    "sentinel_era_max_gap_days", "sentinel_era_n_missing_values",
    "quality_code_top5", "n_bad_timestamps", "declared_rows",
    "parameter", "ts_shortname", "unit",
    "error", "file",
]

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",   # WISKI-style: 2019-07-15T14:15:00.000Z
    "%Y-%m-%d %H:%M:%S%z",     # plain-CSV style: 2021-11-16 12:15:00+00:00
]


def parse_metadata(header_lines):
    meta = {}
    for line in header_lines:
        content = line.lstrip("#").strip()
        if content.lower().startswith("timestamp"):
            continue
        if ";" in content:
            key, _, value = content.partition(";")
            key = key.strip()
            if key in METADATA_KEYS:
                meta[METADATA_KEYS[key]] = value.strip()
    return meta


def parse_timestamps(raw_series):
    """Try known fast formats first, fall back to generic parsing for any leftovers."""
    ts = pd.Series(pd.NaT, index=raw_series.index)
    ts = pd.to_datetime(ts, utc=True)
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


def read_normalized(filepath):
    """Returns (df, meta, file_format). df always has columns:
    timestamp_raw, value, quality_code, quality_ok (quality_ok may be all NA)."""
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()
        f.seek(0)

        if first_line.startswith("#"):
            # --- WISKI-style: metadata block + semicolon-delimited data ---
            header_lines = []
            for _ in range(30):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.startswith("#"):
                    f.seek(pos)
                    break
                header_lines.append(line)
            meta = parse_metadata(header_lines)
            df = pd.read_csv(
                f, sep=";", names=["timestamp_raw", "value", "quality_code"],
                header=None, dtype={"value": "float64"},
                na_values=["", "NaN", "nan"], low_memory=False,
            )
            df["quality_ok"] = pd.NA
            return df, meta, "wiski"

        else:
            # --- Plain CSV: datetime,value,quality_code,quality_ok,station_ref ---
            df = pd.read_csv(f, sep=",", header=0, na_values=["", "NaN", "nan"], low_memory=False)
            df = df.rename(columns={"datetime": "timestamp_raw"})
            for col in ["timestamp_raw", "value", "quality_code"]:
                if col not in df.columns:
                    df[col] = pd.NA
            meta = {}
            if "quality_ok" in df.columns:
                df["quality_ok"] = (
                    df["quality_ok"].astype(str).str.strip().str.lower()
                    .map({"true": True, "false": False}).astype("boolean")
                )
            else:
                df["quality_ok"] = pd.NA
            if "station_ref" in df.columns and df["station_ref"].notna().any():
                meta["station_no"] = str(df["station_ref"].dropna().iloc[0])
            return df, meta, "plain_csv"


def analyze_file(filepath):
    filename = os.path.basename(filepath)
    result = {"file": filename, "status": "ok", "error": ""}

    try:
        df, meta, file_format = read_normalized(filepath)
        result["file_format"] = file_format
        for col in METADATA_KEYS.values():
            result[col] = meta.get(col, "")

        # Station ID: prefer the value embedded in the data itself (station_ref),
        # fall back to any digit run in the filename.
        if meta.get("station_no"):
            result["station_id"] = meta["station_no"]
        else:
            m = re.search(r"(\d+)", filename)
            result["station_id"] = m.group(1) if m else filename

        if df.empty:
            result["status"] = "empty"
            return result

        df["timestamp"] = parse_timestamps(df["timestamp_raw"].astype(str))
        n_bad_timestamps = int(df["timestamp"].isna().sum())
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

        if df.empty:
            result["status"] = "no_valid_timestamps"
            result["n_bad_timestamps"] = n_bad_timestamps
            return result

        result["n_bad_timestamps"] = n_bad_timestamps
        result["total_rows"] = len(df)

        full_start, full_end = df["timestamp"].min(), df["timestamp"].max()
        result["record_start"] = full_start.isoformat()
        result["record_end"] = full_end.isoformat()
        result["record_span_days"] = (full_end - full_start).days

        qc_counts = df["quality_code"].value_counts().head(5)
        result["quality_code_top5"] = "; ".join(f"{k}:{v}" for k, v in qc_counts.items())

        sentinel_df = df[df["timestamp"] >= SENTINEL_ERA_START]
        result["sentinel_era_rows"] = len(sentinel_df)

        if sentinel_df.empty:
            result["sentinel_era_span_days"] = 0
            result["sentinel_era_completeness_pct"] = 0.0
            result["sentinel_era_max_gap_days"] = np.nan
            result["usable_for_study"] = False
            return result

        s_start, s_end = sentinel_df["timestamp"].min(), sentinel_df["timestamp"].max()
        result["sentinel_era_start"] = s_start.isoformat()
        result["sentinel_era_end"] = s_end.isoformat()
        span_days = (s_end - s_start).days
        result["sentinel_era_span_days"] = span_days

        # Infer this station's actual native recording cadence rather than assuming
        # a fixed interval -- OPW/EPA stations do NOT all log at the same frequency
        # (observed: mostly 2-3 min or 15 min, with several values in between).
        all_ts_sorted = sentinel_df["timestamp"].sort_values()
        if len(all_ts_sorted) > 1:
            diffs_sec = all_ts_sorted.diff().dropna().dt.total_seconds()
            inferred_interval_sec = max(float(diffs_sec.median()), 1.0)
        else:
            inferred_interval_sec = EXPECTED_INTERVAL_MINUTES * 60
        result["sentinel_era_interval_minutes"] = round(inferred_interval_sec / 60, 2)

        expected_steps = max(
            1, int((s_end - s_start).total_seconds() / inferred_interval_sec) + 1
        )
        non_null_values = int(sentinel_df["value"].notna().sum())
        result["sentinel_era_n_missing_values"] = int(sentinel_df["value"].isna().sum())
        result["sentinel_era_completeness_pct"] = round(100 * non_null_values / expected_steps, 2)

        if sentinel_df["quality_ok"].notna().any():
            result["sentinel_era_quality_ok_pct"] = round(
                100 * sentinel_df["quality_ok"].dropna().mean(), 2
            )
        else:
            result["sentinel_era_quality_ok_pct"] = ""

        valid_ts = sentinel_df.loc[sentinel_df["value"].notna(), "timestamp"]
        if len(valid_ts) > 1:
            gaps = valid_ts.diff().dropna()
            result["sentinel_era_max_gap_days"] = round(gaps.max().total_seconds() / 86400, 2)
        else:
            result["sentinel_era_max_gap_days"] = np.nan

        result["usable_for_study"] = bool(
            span_days >= 5 * 365
            and result["sentinel_era_completeness_pct"] >= 70
            and (pd.isna(result["sentinel_era_max_gap_days"]) or result["sentinel_era_max_gap_days"] <= 60)
        )

        result["status"] = "ok"
        return result

    except Exception as e:
        result["status"] = "read_error"
        result["error"] = str(e)
        return result


def load_already_done(out_path):
    """Return the set of source filenames that previously finished with status 'ok'.
    Anything else (errors, empty, not yet reached) is retried on the next run."""
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return set()
    try:
        prior = pd.read_csv(out_path, usecols=["file", "status"])
    except Exception:
        return set()
    return set(prior.loc[prior["status"] == "ok", "file"])


def main():
    parser = argparse.ArgumentParser(description="Check water level station data availability (resumable).")
    parser.add_argument("--data-dir", required=True, help="Directory containing station CSV files")
    parser.add_argument("--out", default="station_availability_summary.csv", help="Output summary CSV path")
    parser.add_argument("--pattern", default="wl_*.csv", help="Glob pattern for station files")
    parser.add_argument("--restart", action="store_true", help="Ignore any existing output and reprocess everything")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.restart and os.path.exists(args.out):
        os.remove(args.out)

    already_done = load_already_done(args.out)
    todo = [fp for fp in files if os.path.basename(fp) not in already_done]

    print(f"Found {len(files)} files total. {len(already_done)} already completed, {len(todo)} remaining.")
    if not todo:
        print("Nothing to do -- all files already processed. Use --restart to force a full re-run.")
        return

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    n_processed_this_run = 0

    try:
        with open(args.out, "a", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=ALL_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                out_f.flush()
                os.fsync(out_f.fileno())

            for i, fp in enumerate(todo, 1):
                result = analyze_file(fp)
                row = {col: result.get(col, "") for col in ALL_COLUMNS}
                writer.writerow(row)
                out_f.flush()
                os.fsync(out_f.fileno())
                n_processed_this_run += 1
                if i % 10 == 0 or i == len(todo):
                    print(f"  processed {i}/{len(todo)} this run "
                          f"({len(already_done) + i}/{len(files)} overall)")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {n_processed_this_run} files this run. "
              f"Progress saved to {args.out} -- re-run the same command to resume.")
        sys.exit(1)

    summary = pd.read_csv(args.out)
    if summary.duplicated(subset="file").any():
        summary = summary.drop_duplicates(subset="file", keep="last")
        summary.to_csv(args.out, index=False)

    n_ok = (summary["status"] == "ok").sum()
    n_usable = summary["usable_for_study"].sum() if "usable_for_study" in summary.columns else 0
    print(f"\nDone. {n_ok}/{len(summary)} files parsed successfully overall, "
          f"{n_usable} flagged usable_for_study=True.")
    print(f"Summary written to: {args.out}")


if __name__ == "__main__":
    main()
