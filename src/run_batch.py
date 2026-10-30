r"""
run_batch.py  (multi-catchment, multi-year driver around run_catchment_pipeline.py)

Loops over a list of stations x years. For each (station, year) "unit" it does
the same thing run_catchment_pipeline.py does for one catchment:

    submit CDSE jobs -> wait + download -> extract embeddings + pixel cache ->
    verify  (raw imagery is KEPT; see "RAW IMAGERY" below)

but with the bookkeeping a long unattended run needs:

  * RESUMABLE. Re-running the same command skips every unit whose permanent
    output (embeddings AND pixel cache) already passes verification, resumes
    units whose imagery is already on disk, and only submits what is genuinely
    missing. Safe to Ctrl-C.
  * PIXEL CACHE. Each unit also writes results/cache/pixels_*.npz (see
    pixel_cache.py) so encoders other than Clay, e.g. Presto via extract_presto.py,
    never need the raw imagery again. A unit processed before the cache existed
    counts as unfinished: only its cache is built, its embeddings are not touched.
    --no-cache disables this.
  * LOOKAHEAD. While unit i is downloading/extracting, the CDSE jobs for the
    next --lookahead units are already running server-side.
  * ISOLATED FAILURES. A failed unit is logged and left on disk for review;
    the batch carries on. After --max-consecutive-failures failures in a row
    it stops, so a systematic problem (bad credentials, quota) does not burn
    through the whole list.
  * RETRIES FAILED JOBS. Jobs recorded as SUBMIT_ERROR, or that ended in an
    error/failed/cancelled state, are dropped from the manifest and resubmitted
    on the next run (the original submit loop treated every recorded job as
    'done' forever, which would leave a failed unit stuck).
  * LOG. Every unit appends one row to --log (batch_log.csv): outcome, detail,
    timestamp.
  * --dry-run prints the plan (SKIP / RESUME / NEW per unit) and touches
    nothing, including CDSE.

RAW IMAGERY is no longer deleted. It stays in --tmp-download-dir (default
results/scratch/downloads) until you run cleanup_raw.py, or pass --delete-raw,
once you have finished testing on the cache. Mind the free disk space meanwhile.

It reuses run_catchment_pipeline.py's functions (verify, wait_and_download,
process_one_catchment, ...) and acquire_sentinel_imagery_cdse.py's building
blocks unchanged; neither file is modified. Put this file next to them.

Credentials: --client-id/--client-secret, or the environment variables
CDSE_CLIENT_ID / CDSE_CLIENT_SECRET (keeps the secret out of shell history).

Usage:
    python run_batch.py --catchment-list results/selection/selected_catchments.csv \
        --shapefile config/shapefiles/subcatchments/subcatchments.shp \
        --years 2022 2023 --sensors s1 --encoder clay --dry-run

    Cache only (no Clay), e.g. to prepare for Presto:  --encoder none

    Outputs go under the results root (see project_paths.py): embeddings/, cache/,
    raw_samples/, logs/batch_log.csv, scratch/downloads/. Each can be overridden.

    (drop --dry-run to run for real; add --stations 11001 22009 to subset)
"""

import argparse
import csv
import datetime
import os
import sys
import types

import pandas as pd

import run_catchment_pipeline as rcp
from project_paths import Layout
from acquire_sentinel_imagery_cdse import (
    S1_BANDS, S1_COLLECTION, S2_BANDS, S2_COLLECTION,
    build_bbox, build_geometry, connect, load_done_keys, submit_job,
)


def unit_dir(tmp_root, station_id, year):
    return os.path.join(tmp_root, f"station_{station_id}_{year}")


def plan_units(catchments, stations, years, order):
    rows = catchments.to_dict("records")
    if stations:
        rows = [r for r in rows if str(r["station_id"]) in set(stations)]
    elif "selection" in catchments.columns:
        rows = [r for r in rows if r.get("selection") != "reserve"]
    if order == "year":
        return [(r, y) for y in years for r in rows]
    return [(r, y) for r in rows for y in years]


def drop_failed_jobs(manifest_path):
    """Remove rows for jobs that failed to submit or finished in an error/failed/
    cancelled state, so load_done_keys() no longer treats them as done and they
    are resubmitted. Without this a failed job would block its unit forever."""
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return 0
    df = pd.read_csv(manifest_path, dtype=str)
    bad = df["status"].fillna("").str.contains("error|fail|cancel", case=False)
    if bad.any():
        df[~bad].to_csv(manifest_path, index=False)
    return int(bad.sum())


def submit_unit(conn, row, shapefile, year, sensors, manifest_path, max_cloud_pct=60.0):
    """Submit CDSE jobs for one catchment-year, for the requested sensors only.
    Same submission logic as acquire_sentinel_imagery_cdse.py, as direct calls."""
    n_retry = drop_failed_jobs(manifest_path)
    if n_retry:
        print(f"  {n_retry} earlier failed job(s) in the manifest will be resubmitted")
    station_id = row["station_id"]
    catchment_id = row.get("catchment_id", row["subcatchment_id"])
    subcatchment_id = row["subcatchment_id"]
    bbox = build_bbox(shapefile, subcatchment_id)
    geometry = build_geometry(shapefile, subcatchment_id)
    if bbox is None or geometry is None:
        print(f"  subcatchment_id {subcatchment_id} not found in the shapefile -- nothing submitted")
        return 0

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    done = load_done_keys(manifest_path)
    write_header = not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0
    n = 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        if write_header:
            f.write("catchment_id,station_id,subcatchment_id,sensor,year,job_id,status\n")
            f.flush()
        for sensor in sensors:
            key = (str(catchment_id), sensor, str(year))
            if key in done:
                continue
            collection = S2_COLLECTION if sensor == "s2" else S1_COLLECTION
            bands = S2_BANDS if sensor == "s2" else S1_BANDS
            title = f"{sensor}_{station_id}_{year}"
            try:
                job_id = submit_job(conn, collection, bands, bbox, geometry, f"{year}-01-01", f"{year + 1}-01-01",
                                    title, sensor, max_cloud_pct if sensor == "s2" else None)
                status = "SUBMITTED"
            except Exception as e:  # recorded, and retried on the next run
                job_id, status = "", f"SUBMIT_ERROR: {str(e).replace(',', ';').replace(chr(10), ' ')[:200]}"
            f.write(f"{catchment_id},{station_id},{subcatchment_id},{sensor},{year},{job_id},{status}\n")
            f.flush()
            n += 1
            print(f"  station {station_id} {sensor} {year}: {status}")
    return n


def log_row(path, station_id, year, sensors, outcome, detail=""):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "station_id", "year", "sensors", "outcome", "detail"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), station_id, year,
                    ",".join(sensors), outcome, detail])


def classify(row, year, ns, tmp_root, reprocess):
    """SKIP / RESUME / NEW for one unit (no side effects beyond printing from the pipeline helpers)."""
    sid = row["station_id"]
    ns.year = year
    if not reprocess and rcp.already_extracted(ns, sid, ns.sensors):
        return "SKIP"
    if rcp.images_already_downloaded(unit_dir(tmp_root, sid, year), sid, year):
        return "RESUME"
    return "NEW"


def main():
    ap = argparse.ArgumentParser(description="Batch driver: many stations x years, resumable.")
    ap.add_argument("--catchment-list", required=True)
    ap.add_argument("--shapefile", required=True)
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--stations", nargs="*", default=None,
                    help="station ids to run (default: every row in the list except 'reserve' rows)")
    ap.add_argument("--sensors", default="s1", help="comma-separated: s1, s2 (default s1)")
    ap.add_argument("--encoder", default="clay", choices=["dummy", "clay", "presto"],
                    help="none = write only the pixel cache. Presto runs from the cache: extract_presto.py")
    ap.add_argument("--results-root", default=None,
                    help="root for all outputs (default: <project>/results or $FFB_RESULTS_ROOT)")
    ap.add_argument("--embeddings-out-dir", default=None, help="default: <results root>/embeddings_<encoder>")
    ap.add_argument("--cache-dir", default=None, help="default: <results root>/cache")
    ap.add_argument("--no-cache", action="store_true", help="do not write the pixel cache")
    ap.add_argument("--cache-pixels", type=int, default=256, help="sampled locations per catchment (default 256)")
    ap.add_argument("--cache-window", type=int, default=3, help="odd window size averaged at each location (default 3)")
    ap.add_argument("--delete-raw", action="store_true",
                    help="delete each unit's raw imagery once its outputs verify. Default: keep it "
                         "(cleanup_raw.py later). Leave off until the encoder tests are finished.")
    ap.add_argument("--samples-dir", default=None, help="default: <results root>/raw_samples")
    ap.add_argument("--tmp-download-dir", default=None,
                    help="raw imagery, kept until cleanup_raw.py (default: <results root>/scratch/downloads)")
    ap.add_argument("--client-id", default=os.environ.get("CDSE_CLIENT_ID"))
    ap.add_argument("--client-secret", default=os.environ.get("CDSE_CLIENT_SECRET"))
    ap.add_argument("--lookahead", type=int, default=1, help="units whose CDSE jobs are submitted ahead of time")
    ap.add_argument("--order", default="year", choices=["year", "station"],
                    help="year: all stations for a year before the next year (a partial run still supports "
                         "leave-one-station-out). station: all years of a station together.")
    ap.add_argument("--poll-interval-sec", type=int, default=60)
    ap.add_argument("--timeout-sec", type=int, default=3600 * 4)
    ap.add_argument("--keep-sample-dates", type=int, default=3)
    ap.add_argument("--reprocess", action="store_true", help="re-extract even if verified output exists")
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--log", default=None, help="default: <results root>/logs/batch_log.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    layout = Layout(args.results_root)
    args.embeddings_out_dir = args.embeddings_out_dir or layout.embeddings_for(args.encoder if args.encoder != "none" else "clay")
    args.samples_dir = args.samples_dir or layout.raw_samples
    args.tmp_download_dir = args.tmp_download_dir or layout.scratch_downloads
    args.cache_dir = args.cache_dir or layout.cache
    args.log = args.log or os.path.join(layout.logs, "batch_log.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    print(f"Embeddings -> {args.embeddings_out_dir}\nPixel cache -> {'(off)' if args.no_cache else args.cache_dir}\n"
          f"Raw QA samples -> {args.samples_dir}\nRaw downloads -> {args.tmp_download_dir} "
          f"({'deleted after verification' if args.delete_raw else 'KEPT'})\nBatch log -> {args.log}\n")

    sensors = [s.strip().lower() for s in args.sensors.split(",") if s.strip()]
    if not sensors or any(s not in ("s1", "s2") for s in sensors):
        ap.error("--sensors must be a comma-separated subset of s1,s2")
    if args.encoder == "none" and args.no_cache:
        ap.error("--encoder none together with --no-cache would write nothing")

    catchments = pd.read_csv(args.catchment_list)
    units = plan_units(catchments, args.stations, args.years, args.order)
    if not units:
        sys.exit("No matching stations in the catchment list.")

    ns = types.SimpleNamespace(
        embeddings_out_dir=args.embeddings_out_dir, samples_dir=args.samples_dir, year=None, sensors=sensors, reprocess=args.reprocess,
        shapefile=args.shapefile, encoder=args.encoder, keep_sample_dates=args.keep_sample_dates,
        cache_dir=args.cache_dir, write_cache=not args.no_cache, cache_pixels=args.cache_pixels,
        cache_window=args.cache_window, delete_raw=args.delete_raw)
    os.makedirs(args.embeddings_out_dir, exist_ok=True)
    os.makedirs(args.tmp_download_dir, exist_ok=True)

    print(f"{len(units)} unit(s): {len({r['station_id'] for r, _ in units})} station(s) x "
          f"{len(args.years)} year(s), sensors {sensors}, encoder {args.encoder}\n")
    plan = []
    for r, y in units:
        plan.append((r, y, classify(r, y, ns, args.tmp_download_dir, args.reprocess)))
    print("\n=== Plan ===")
    for r, y, st in plan:
        print(f"  {st:<7} station {r['station_id']}  {y}")
    counts = pd.Series([st for _, _, st in plan]).value_counts().to_dict()
    print(f"  -> {counts}\n")
    if args.dry_run:
        print("Dry run: nothing submitted or processed.")
        return

    todo = [(r, y, st) for r, y, st in plan if st != "SKIP"]
    if not todo:
        print("Everything is already done and verified.")
        return
    if any(st == "NEW" for _, _, st in todo) and not (args.client_id and args.client_secret):
        print("No --client-id/--client-secret (or CDSE_* env vars): CDSE will ask for interactive login.")

    conn = None
    submitted = set()

    def ensure_submitted(r, y):
        nonlocal conn
        key = (r["station_id"], y)
        if key in submitted:
            return
        submitted.add(key)
        d = unit_dir(args.tmp_download_dir, r["station_id"], y)
        if rcp.images_already_downloaded(d, r["station_id"], y):
            return
        if conn is None:
            conn = connect(args.client_id, args.client_secret)
        os.makedirs(d, exist_ok=True)
        submit_unit(conn, r, args.shapefile, y, sensors, os.path.join(d, "manifest.csv"))

    results, fails_in_row = [], 0
    for i, (r, y, st) in enumerate(todo):
        sid = r["station_id"]
        print(f"\n######## [{i + 1}/{len(todo)}] station {sid} {y} ({st}) ########")
        for r2, y2, _ in todo[i:i + 1 + max(0, args.lookahead)]:
            try:
                ensure_submitted(r2, y2)
            except Exception as e:
                print(f"  submit step failed for station {r2['station_id']} {y2}: {e}")

        d = unit_dir(args.tmp_download_dir, sid, y)
        os.makedirs(d, exist_ok=True)
        outcome, detail = "ok", ""
        try:
            if not rcp.images_already_downloaded(d, sid, y):
                got = rcp.wait_and_download(os.path.join(d, "manifest.csv"), d, args.client_id, args.client_secret,
                                            poll_interval_sec=args.poll_interval_sec, timeout_sec=args.timeout_sec)
                if not got:
                    outcome, detail = "download_failed", f"see {os.path.join(d, 'manifest.csv')}"
            if outcome == "ok":
                ns.year = y
                ok = rcp.process_one_catchment(r, ns, d)
                if not ok:
                    outcome, detail = "extract_or_verify_failed", f"raw imagery kept at {d}"
                else:
                    detail = "raw deleted" if args.delete_raw else "raw kept"
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run the same command to resume.")
            log_row(args.log, sid, y, sensors, "interrupted")
            sys.exit(130)
        except Exception as e:
            outcome, detail = "error", f"{type(e).__name__}: {e}"
            print(f"  ERROR: {detail}")

        log_row(args.log, sid, y, sensors, outcome, detail)
        results.append({"station_id": sid, "year": y, "outcome": outcome})
        fails_in_row = 0 if outcome == "ok" else fails_in_row + 1
        if fails_in_row >= args.max_consecutive_failures:
            print(f"\nStopping: {fails_in_row} units failed in a row (see {args.log}). "
                  f"Fix the cause and re-run the same command to resume.")
            break

    print("\n=== Batch summary ===")
    print(pd.DataFrame(results).to_string(index=False))
    n_bad = sum(x["outcome"] != "ok" for x in results)
    if n_bad:
        print(f"\n{n_bad} unit(s) need attention; details in {args.log}. Re-running resumes them.")
    print(f"\nNext: python baseline_comparison.py --labels <antecedent_state_labels.csv> "
          f"--sensor {sensors[0]} --cv loso")
    if not args.no_cache:
        print(f"      python extract_presto.py --sensor {sensors[0]}      (Presto, from the cache)")
    if not args.delete_raw:
        print("      python cleanup_raw.py                (dry run: what raw imagery could be removed)")


if __name__ == "__main__":
    main()
