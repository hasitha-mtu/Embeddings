r"""
run_catchment_pipeline.py  (Component 4+5 orchestrator: process-then-discard)

Ties together acquire_sentinel_imagery_cdse.py, download_cdse_results.py, and
extract_embeddings.py into one per-catchment loop:

    for each catchment:
        1. submit CDSE acquisition jobs (S1 + S2, one target year)
        2. poll until jobs finish, download results
        3. extract_embeddings.py on the downloaded imagery
        4. VERIFY the output looks sane
        5. only if verification passes: delete the raw downloaded imagery
        6. move to the next catchment

So peak local disk usage stays around one (or a few) catchment's worth of
raw imagery, not the full ~1 TB for all 36 catchments at once.

LOOKAHEAD: rather than fully serializing (wait for catchment N's downloads
to finish before even submitting catchment N+1's jobs), this submits
catchment N+1's acquisition jobs immediately after catchment N's finish
submitting -- so CDSE processes them server-side while this script is busy
extracting embeddings and deleting files for catchment N. Recovers most of
the lost parallelism from strict one-at-a-time processing without needing
real concurrency/threading.

SAFETY: raw imagery for a catchment is deleted ONLY after extract_embeddings.py
has run successfully AND the output files pass basic sanity checks (exist,
non-empty, roughly the expected number of dates). If verification fails, the
raw folder is left in place and flagged, not deleted -- never silently lost.

CAVEAT: this orchestrates the GEE/CDSE and embedding-extraction scripts,
none of which have been run end-to-end together from this environment (no
network access to CDSE or model weight hosts here). Test on ONE catchment
first (--station-id-filter) before trusting a full unattended run.

Usage:
    python run_catchment_pipeline.py \
        --catchment-list catchment_level_shortlist.csv \
        --shapefile subcatchments/subcatchments.shp \
        --year 2023 \
        --embeddings-out-dir ./embeddings_permanent \
        --tmp-download-dir ./tmp_downloads \
        --client-id ... --client-secret ... \
        --encoder dummy

    # Test on one catchment first:
    python run_catchment_pipeline.py ... --station-id-filter 11001
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from download_cdse_results import download_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd, description):
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED ({description}):\n{result.stdout}\n{result.stderr}")
        return False
    return True


def submit_acquisition(catchment_list, shapefile, station_id, year, manifest_path,
                        client_id, client_secret):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "acquire_sentinel_imagery_cdse.py"),
        "--catchment-list", catchment_list, "--shapefile", shapefile,
        "--station-id-filter", str(station_id), "--year", str(year),
        "--out-manifest", manifest_path,
    ]
    if client_id and client_secret:
        cmd += ["--client-id", client_id, "--client-secret", client_secret]
    return run_cmd(cmd, "acquisition submit")


def wait_and_download(manifest_path, out_dir, client_id, client_secret,
                       poll_interval_sec=60, timeout_sec=3600 * 4):
    # download_cmd = [
    #     sys.executable, os.path.join(SCRIPT_DIR, "download_cdse_results.py"),
    #     "--manifest", manifest_path, "--out-dir", out_dir,
    # ]
    # if client_id and client_secret:
    #     download_cmd += ["--client-id", client_id, "--client-secret", client_secret]

    elapsed = 0
    while elapsed < timeout_sec:
        # run_cmd(download_cmd, "download poll")
        print("download poll")
        download_data(manifest_path, out_dir, client_id, client_secret)
        if not os.path.exists(manifest_path):
            return False
        df = pd.read_csv(manifest_path, dtype=str)
        pending = df["status"].isin(["SUBMITTED", "running", "queued", "created"]).sum()
        if pending == 0:
            failed = df["status"].str.contains("error|fail", case=False, na=False).sum()
            downloaded = (df["status"] == "DOWNLOADED").sum()
            print(f"    all jobs finished: {downloaded} downloaded, {failed} failed")
            return failed == 0
        print(f"    {pending} jobs still pending, waiting {poll_interval_sec}s...")
        time.sleep(poll_interval_sec)
        elapsed += poll_interval_sec
    print(f"    TIMEOUT after {timeout_sec}s with jobs still pending")
    return False


def extract(input_dir, shapefile, subcatchment_id, sensor, encoder, out_embeddings, out_indices):
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "extract_embeddings.py"),
        "--input-dir", input_dir, "--shapefile", shapefile,
        "--subcatchment-id", str(subcatchment_id), "--sensor", sensor,
        "--encoder", encoder, "--out-embeddings", out_embeddings, "--out-indices", out_indices,
    ]
    return run_cmd(cmd, f"extract {sensor}")


def verify_output(embeddings_path, indices_path, min_expected_dates=5):
    """Never delete raw imagery unless this passes. Checks existence,
    non-emptiness, and a plausible date count -- not a guarantee of
    correctness, but catches the obvious failure modes (crashed partway,
    wrote an empty file, etc.)."""
    if not os.path.exists(embeddings_path) or not os.path.exists(indices_path):
        return False, "output file(s) missing"
    if os.path.getsize(embeddings_path) == 0 or os.path.getsize(indices_path) == 0:
        return False, "output file(s) empty"
    try:
        emb = np.load(embeddings_path)
        n_dates = len(emb.files)
        idx = pd.read_csv(indices_path)
    except Exception as e:
        return False, f"output file(s) unreadable: {e}"
    if n_dates < min_expected_dates:
        return False, f"only {n_dates} dates in embeddings (expected >= {min_expected_dates})"
    if len(idx) != n_dates:
        return False, f"embeddings has {n_dates} dates but indices has {len(idx)} rows -- mismatch"
    all_nan_dates = sum(1 for k in emb.files if np.isnan(emb[k]).all())
    if all_nan_dates == n_dates:
        return False, "every date's embedding is all-NaN -- likely a masking bug, not real data"
    return True, f"{n_dates} dates, {all_nan_dates} all-NaN"


def process_one_catchment(row, args, tmp_dir_for_this_catchment):
    catchment_id = row.get("catchment_id", row.get("subcatchment_id"))
    station_id = row["station_id"]
    subcatchment_id = row["subcatchment_id"]
    print(f"\n=== Catchment {catchment_id} (station {station_id}) ===")

    manifest_path = os.path.join(tmp_dir_for_this_catchment, "manifest.csv")
    print("  [1/4] waiting for acquisition jobs + downloading...")
    ok = wait_and_download(manifest_path, tmp_dir_for_this_catchment,
                            args.client_id, args.client_secret,
                            poll_interval_sec=args.poll_interval_sec, timeout_sec=args.timeout_sec)
    if not ok:
        print(f"  ABORTED for catchment {catchment_id}: acquisition/download did not complete cleanly. "
              f"Raw folder (if any) left in place at {tmp_dir_for_this_catchment} for manual inspection.")
        return False

    os.makedirs(args.embeddings_out_dir, exist_ok=True)
    all_ok = True
    for sensor in ["s1", "s2"]:
        input_dir = os.path.join(tmp_dir_for_this_catchment, f"{sensor}_{station_id}_{args.year}")
        if not os.path.isdir(input_dir):
            print(f"  [2/4] no {sensor} folder found for station {station_id} -- skipping this sensor")
            continue

        out_emb = os.path.join(args.embeddings_out_dir, f"embeddings_{station_id}_{sensor}_{args.year}.npz")
        out_idx = os.path.join(args.embeddings_out_dir, f"indices_{station_id}_{sensor}_{args.year}.csv")

        print(f"  [2/4] extracting {sensor} embeddings...")
        if not extract(input_dir, args.shapefile, subcatchment_id, sensor, args.encoder, out_emb, out_idx):
            all_ok = False
            continue

        print("  [3/4] verifying output...")
        passed, detail = verify_output(out_emb, out_idx)
        print(f"    {'PASS' if passed else 'FAIL'}: {detail}")
        if not passed:
            all_ok = False
            continue

        if args.keep_sample_dates > 0:
            _keep_sample(input_dir, args.embeddings_out_dir, station_id, sensor, args.keep_sample_dates)

    if all_ok:
        print(f"  [4/4] all sensors verified -- deleting raw imagery at {tmp_dir_for_this_catchment}")
        shutil.rmtree(tmp_dir_for_this_catchment, ignore_errors=True)
    else:
        print(f"  [4/4] NOT deleting -- at least one sensor failed verification. "
              f"Raw folder left at {tmp_dir_for_this_catchment} for manual review.")
    return all_ok


def _keep_sample(input_dir, out_dir, station_id, sensor, n):
    """Keep a small number of raw files as permanent visual-QA samples,
    per the earlier discussion that discarding ALL raw imagery loses the
    ability to spot-check it later (the same kind of check done by hand on
    station 11001's hydrograph)."""
    sample_dir = os.path.join(out_dir, "raw_samples", f"{station_id}_{sensor}")
    os.makedirs(sample_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".tif"))
    step = max(1, len(files) // n)
    for f in files[::step][:n]:
        shutil.copy(os.path.join(input_dir, f), os.path.join(sample_dir, f))


def main():
    parser = argparse.ArgumentParser(description="Process-then-discard pipeline: acquire, embed, verify, delete, per catchment.")
    parser.add_argument("--catchment-list", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--year", type=int, required=True, help="One year per run -- loop the whole script per year for multi-year coverage")
    parser.add_argument("--embeddings-out-dir", required=True, help="PERMANENT output location -- never deleted")
    parser.add_argument("--tmp-download-dir", required=True, help="TRANSIENT -- raw imagery lives here briefly, then gets deleted")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    parser.add_argument("--encoder", default="dummy", choices=["dummy", "clay", "presto"])
    parser.add_argument("--station-id-filter", default=None, help="Test on one station first")
    parser.add_argument("--poll-interval-sec", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=3600 * 4)
    parser.add_argument("--keep-sample-dates", type=int, default=3,
                         help="Number of raw files per catchment/sensor to keep permanently for visual QA (0 to disable)")
    args = parser.parse_args()

    catchments = pd.read_csv(args.catchment_list)
    if args.station_id_filter:
        catchments = catchments[catchments["station_id"].astype(str) == args.station_id_filter]
    if catchments.empty:
        print("No matching catchments.", file=sys.stderr)
        sys.exit(1)
    catchments = catchments.reset_index(drop=True)

    os.makedirs(args.tmp_download_dir, exist_ok=True)
    catchment_tmp_dirs = []
    for _, row in catchments.iterrows():
        d = os.path.join(args.tmp_download_dir, f"station_{row['station_id']}")
        os.makedirs(d, exist_ok=True)
        catchment_tmp_dirs.append(d)

    print(f"Processing {len(catchments)} catchment(s) for year {args.year}.")

    # Submit catchment 0's jobs before the loop starts, so the loop body can
    # always submit catchment i+1 while it processes/deletes catchment i --
    # this is the lookahead that recovers most of sequential processing's
    # lost server-side parallelism.
    print("\nSubmitting acquisition jobs for catchment 0...")
    submit_acquisition(args.catchment_list, args.shapefile, catchments.iloc[0]["station_id"],
                        args.year, os.path.join(catchment_tmp_dirs[0], "manifest.csv"),
                        args.client_id, args.client_secret)

    results = []
    for i, row in catchments.iterrows():
        if i + 1 < len(catchments):
            next_row = catchments.iloc[i + 1]
            print(f"\nSubmitting acquisition jobs for catchment {i + 1} "
                  f"(station {next_row['station_id']}) -- overlapping with catchment {i}'s processing...")
            submit_acquisition(args.catchment_list, args.shapefile, next_row["station_id"],
                                args.year, os.path.join(catchment_tmp_dirs[i + 1], "manifest.csv"),
                                args.client_id, args.client_secret)

        ok = process_one_catchment(row, args, catchment_tmp_dirs[i])
        results.append({"station_id": row["station_id"], "success": ok})

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.to_string(index=False))
    n_failed = (~results_df["success"]).sum()
    if n_failed:
        print(f"\n{n_failed} catchment(s) need manual review (raw imagery preserved, not deleted).")


if __name__ == "__main__":
    main()
