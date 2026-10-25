r"""
run_catchment_pipeline.py  (Component 4+5 orchestrator: process-then-discard)

Ties together acquire_sentinel_imagery_cdse.py, download_cdse_results.py, and
extract_embeddings.py into one per-catchment loop:

    for each catchment:
        1. if raw imagery for this catchment is already on disk, skip
           straight to step 3 -- otherwise:
           1a. submit CDSE acquisition jobs (S1 + S2, one target year)
           1b. poll until jobs finish, download results
        2. (folded into 1)
        3. extract_embeddings.py on the downloaded imagery
        4. VERIFY the output looks sane
        5. only if verification passes: delete the raw downloaded imagery
        6. move to the next catchment

So peak local disk usage stays around one (or a few) catchment's worth of
raw imagery, not the full ~1 TB for all 36 catchments at once.

SKIP-IF-DONE: a catchment is skipped entirely (no CDSE polling, no extraction)
when every sensor in --sensors already has permanent embeddings + indices
output that passes verify_output(). Use --reprocess to redo, e.g. after
changing extract_embeddings.py. A requested sensor with no raw folder and no
verified output now counts as a FAILURE (raw imagery is kept, not deleted).

RESUME / ALREADY-DOWNLOADED HANDLING: before submitting or polling anything
for a catchment, the script checks whether that catchment's imagery is
already sitting on disk in --tmp-download-dir -- either because a previous
run got through acquisition/download but was interrupted before extraction,
or because you downloaded it out-of-band. When that's true it skips
acquisition submission AND the download-polling wait entirely and goes
straight to extraction. This is checked in two places: (a) before the
lookahead submits the *next* catchment's jobs (no point re-submitting to
CDSE for imagery you already have), and (b) before processing the *current*
catchment (no point polling a manifest for jobs that already finished).
Detection is manifest-first (every row DOWNLOADED), falling back to "the
expected s1_*/s2_* folders already contain .tif files" if the manifest is
missing but files are still there. Use --force-redownload to disable this
and always re-submit/re-poll from scratch.

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

    # Imagery for this catchment was already downloaded in a prior run --
    # the script auto-detects this and skips straight to extraction. To
    # force a fresh acquisition/download anyway:
    python run_catchment_pipeline.py ... --force-redownload
"""

import argparse
import os
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

from download_cdse_results import download_data
from extract_embeddings import extract_embeddings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



def images_already_downloaded(tmp_dir_for_this_catchment, station_id, year, min_images=5):
    print(f"    Downloading images for catchment {station_id}-{year}")
    print(f"tmp_dir_for_this_catchment: {tmp_dir_for_this_catchment}")
    manifest_path = os.path.join(tmp_dir_for_this_catchment, "manifest.csv")
    if os.path.exists(manifest_path):
        try:
            df = pd.read_csv(manifest_path, dtype=str)
            if len(df) > 0 and (df["status"] == "DOWNLOADED").all():
                return True
            return False   # manifest exists but some jobs are not DOWNLOADED: not ready
        except Exception as e:
            print(f"    FAILED reading {manifest_path}: {e}\n{traceback.format_exc()}")
            # fall through to the filesystem check below

    for sensor in ["s1", "s2"]:
        input_dir = os.path.join(tmp_dir_for_this_catchment, f"{sensor}_{station_id}_{year}")
        print(f"input_dir: {input_dir}")
        if os.path.isdir(input_dir):
            n_tifs = sum(f.endswith(".tif") for f in os.listdir(input_dir))
            print(f"    {sensor}: {n_tifs} .tif files found")
            # A stray file or two is not a finished download.
            if n_tifs >= min_images:
                return True
    return False


def wait_and_download(manifest_path, out_dir, client_id, client_secret,
                       poll_interval_sec=60, timeout_sec=3600 * 4):
    elapsed = 0
    while elapsed < timeout_sec:
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
    result = extract_embeddings(sensor, input_dir, shapefile, subcatchment_id, encoder,
                   out_embeddings, out_indices)
    print(f"Extract finished|results: {result}")
    return result


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


def output_paths(embeddings_out_dir, station_id, sensor, year):
    return (os.path.join(embeddings_out_dir, f"embeddings_{station_id}_{sensor}_{year}.npz"),
            os.path.join(embeddings_out_dir, f"indices_{station_id}_{sensor}_{year}.csv"))


def already_extracted(args, station_id, sensors):
    """True only if EVERY requested sensor already has permanent output that
    passes verify_output(). Used to skip a catchment entirely on rerun."""
    for sensor in sensors:
        out_emb, out_idx = output_paths(args.embeddings_out_dir, station_id, sensor, args.year)
        passed, detail = verify_output(out_emb, out_idx)
        if not passed:
            return False
    return True


def process_one_catchment(row, args, tmp_dir_for_this_catchment):
    catchment_id = row.get("catchment_id", row.get("subcatchment_id"))
    station_id = row["station_id"]
    subcatchment_id = row["subcatchment_id"]
    print(f"\n=== Catchment {catchment_id} (station {station_id}) ===")

    os.makedirs(args.embeddings_out_dir, exist_ok=True)
    all_ok = True
    for sensor in args.sensors:
        out_emb, out_idx = output_paths(args.embeddings_out_dir, station_id, sensor, args.year)
        if not args.reprocess and verify_output(out_emb, out_idx)[0]:
            print(f"  [2/4] {sensor}: verified output already exists -- skipping")
            continue

        input_dir = os.path.join(tmp_dir_for_this_catchment, f"{sensor}_{station_id}_{args.year}")
        if not os.path.isdir(input_dir):
            # A requested sensor with no raw data is a failure, not a silent skip
            # (previously this still reported success, e.g. S2 for 11001).
            print(f"  [2/4] no {sensor} folder found for station {station_id} -- NOT extracted")
            all_ok = False
            continue

        # Write to temporary names first and only replace the permanent output
        # once the new files verify, so a bad run can never overwrite good output.
        tmp_emb = out_emb[:-4] + ".tmp.npz"
        tmp_idx = out_idx[:-4] + ".tmp.csv"
        print(f"  [2/4] extracting {sensor} embeddings...")
        if not extract(input_dir, args.shapefile, subcatchment_id, sensor, args.encoder, tmp_emb, tmp_idx):
            all_ok = False
            continue

        print("  [3/4] verifying output...")
        passed, detail = verify_output(tmp_emb, tmp_idx)
        print(f"    {'PASS' if passed else 'FAIL'}: {detail}")
        if not passed:
            print(f"    new output left at {tmp_emb} / {tmp_idx}; existing permanent output (if any) NOT replaced")
            all_ok = False
            continue
        os.replace(tmp_emb, out_emb)
        os.replace(tmp_idx, out_idx)

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
    parser.add_argument("--sensors", default="s1,s2",
                         help="Comma-separated sensors to process, e.g. 's1' for Sentinel-1 only")
    parser.add_argument("--reprocess", action="store_true",
                         help="Re-extract even if verified output already exists for a sensor "
                              "(use after changing the extractor).")
    parser.add_argument("--force-redownload", action="store_true",
                         help="Always re-submit acquisition jobs and re-poll/download, even if this "
                              "catchment's imagery already appears on disk in --tmp-download-dir. "
                              "Default behavior is to detect already-downloaded imagery and skip "
                              "straight to extraction.")
    args = parser.parse_args()
    args.sensors = [x.strip().lower() for x in args.sensors.split(",") if x.strip()]
    bad = [x for x in args.sensors if x not in ("s1", "s2")]
    if bad or not args.sensors:
        parser.error(f"--sensors must be a comma-separated subset of s1,s2 (got {args.sensors})")

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

    results = []
    for i, row in catchments.iterrows():
        station_id = row["station_id"]
        if not args.reprocess and already_extracted(args, station_id, args.sensors):
            print(f"\nStation {station_id}: verified output already exists for "
                  f"{','.join(args.sensors)} {args.year} -- skipping (use --reprocess to redo).")
            results.append({"station_id": station_id, "success": True})
            continue
        if args.force_redownload or not images_already_downloaded(catchment_tmp_dirs[i], station_id, args.year):
            manifest_path = os.path.join(catchment_tmp_dirs[i], "manifest.csv")
            wait_and_download(manifest_path, catchment_tmp_dirs[i],
                              args.client_id, args.client_secret,
                              poll_interval_sec=args.poll_interval_sec, timeout_sec=args.timeout_sec)
        else:
            print(f"\nCatchment {i + 1}'s (station {station_id}) imagery already on disk "
                  f"-- skipping acquisition submission.")

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
