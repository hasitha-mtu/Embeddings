r"""
cleanup_raw.py -- delete downloaded raw imagery, but only where it is safe to.

The pipeline now KEEPS raw imagery (it used to delete it automatically). Run this
once you have finished testing, to free the disk space. It is a DRY RUN unless you
pass --apply, and for every raw folder it first checks that what you would lose is
already safely held elsewhere:

    * the Clay embeddings + indices for that station/sensor/year verify
      (skip this check with --encoder none, for cache-only runs)
    * the pixel cache for that station/sensor/year verifies
    * with --require-presto: the Presto embeddings verify as well. Use this to
      enforce "do not delete until the Presto test has been run".

A folder that fails any check is reported as BLOCKED, with the reason, and is never
touched. Only the s1_<station>_<year> / s2_<station>_<year> image folders are
deleted. Each unit's manifest.csv (the CDSE job ids) is left in place, so the
imagery can be fetched again from CDSE later if its job results still exist there.

Usage:
    python cleanup_raw.py                                  # dry run, everything
    python cleanup_raw.py --require-presto                 # dry run, only if Presto output exists
    python cleanup_raw.py --stations 11001 22009 --years 2023
    python cleanup_raw.py --require-presto --apply         # really delete
"""

import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

from pixel_cache import cache_paths, verify_cache
from project_paths import Layout


def verify_embeddings_output(emb_path, idx_path, min_expected_dates=5):
    """Same checks as run_catchment_pipeline.verify_output (kept here so this script
    needs neither openeo nor geopandas)."""
    if not os.path.exists(emb_path) or not os.path.exists(idx_path):
        return False, "missing"
    if os.path.getsize(emb_path) == 0 or os.path.getsize(idx_path) == 0:
        return False, "empty"
    try:
        emb = np.load(emb_path)
        n = len(emb.files)
        idx = pd.read_csv(idx_path)
        all_nan = sum(1 for k in emb.files if np.isnan(emb[k]).all())
    except Exception as e:
        return False, f"unreadable ({e})"
    if n < min_expected_dates:
        return False, f"only {n} dates"
    if len(idx) != n:
        return False, f"{n} dates but {len(idx)} index rows"
    if all_nan == n:
        return False, "every embedding is all-NaN"
    return True, f"{n} dates"


def folder_size(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


def find_raw_folders(download_root):
    """(path, sensor, station, year) for every s1_/s2_ image folder one or two levels down."""
    pat = re.compile(r"^(s[12])_(.+)_(\d{4})$")
    found = []
    if not os.path.isdir(download_root):
        return found
    for unit in sorted(os.listdir(download_root)):
        udir = os.path.join(download_root, unit)
        if not os.path.isdir(udir):
            continue
        for sub in sorted(os.listdir(udir)):
            m = pat.match(sub)
            if m and os.path.isdir(os.path.join(udir, sub)):
                found.append((os.path.join(udir, sub), m.group(1), m.group(2), int(m.group(3))))
    return found


def main():
    ap = argparse.ArgumentParser(description="Delete raw imagery that is safely held as embeddings + cache (dry run by default).")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--tmp-download-dir", default=None, help="default: <results root>/scratch/downloads")
    ap.add_argument("--embeddings-dir", default=None, help="Clay embeddings (default: <results root>/embeddings_clay)")
    ap.add_argument("--presto-dir", default=None, help="default: <results root>/embeddings_presto")
    ap.add_argument("--cache-dir", default=None, help="default: <results root>/cache")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--sensors", nargs="*", default=None, choices=["s1", "s2"])
    ap.add_argument("--encoder", default="clay", choices=["clay", "none"],
                    help="none = do not require Clay embeddings (cache-only runs)")
    ap.add_argument("--require-presto", action="store_true",
                    help="also require verified Presto embeddings for the same station/sensor/year")
    ap.add_argument("--apply", action="store_true", help="really delete (default: only report)")
    args = ap.parse_args()

    lay = Layout(args.results_root)
    root = args.tmp_download_dir or lay.scratch_downloads
    emb_dir = args.embeddings_dir or lay.embeddings_for("clay")
    presto_dir = args.presto_dir or lay.embeddings_for("presto")
    cache_dir = args.cache_dir or lay.cache

    folders = find_raw_folders(root)
    if args.stations:
        folders = [f for f in folders if f[2] in set(args.stations)]
    if args.years:
        folders = [f for f in folders if f[3] in set(args.years)]
    if args.sensors:
        folders = [f for f in folders if f[1] in set(args.sensors)]
    if not folders:
        print(f"No raw image folders found under {root}")
        return

    print(f"Raw imagery under {root}\n")
    safe, blocked = [], []
    for path, sensor, station, year in folders:
        reasons = []
        if args.encoder != "none":
            ok, d = verify_embeddings_output(os.path.join(emb_dir, f"embeddings_{station}_{sensor}_{year}.npz"),
                                             os.path.join(emb_dir, f"indices_{station}_{sensor}_{year}.csv"))
            if not ok:
                reasons.append(f"Clay embeddings: {d}")
        c_npz, c_stats = cache_paths(cache_dir, station, sensor, year)
        ok, d = verify_cache(c_npz, c_stats)
        if not ok:
            reasons.append(f"pixel cache: {d}")
        if args.require_presto:
            ok, d = verify_embeddings_output(os.path.join(presto_dir, f"embeddings_{station}_{sensor}_{year}.npz"),
                                             os.path.join(presto_dir, f"indices_{station}_{sensor}_{year}.csv"))
            if not ok:
                reasons.append(f"Presto embeddings: {d}")
        size = folder_size(path)
        (blocked if reasons else safe).append((path, size, reasons))
        tag = "BLOCKED" if reasons else "safe   "
        print(f"  {tag} {sensor} station {station} {year}  {size / 1e6:8.0f} MB   {'; '.join(reasons)}")

    s_mb = sum(x[1] for x in safe) / 1e6
    b_mb = sum(x[1] for x in blocked) / 1e6
    print(f"\n{len(safe)} folder(s) safe to delete ({s_mb:.0f} MB); {len(blocked)} blocked ({b_mb:.0f} MB, left alone).")
    if not args.apply:
        print("Dry run: nothing deleted. Add --apply to delete the 'safe' folders. manifest.csv files are kept.")
        return
    for path, _, _ in safe:
        shutil.rmtree(path, ignore_errors=True)
        print(f"  deleted {path}")
    print(f"\nFreed about {s_mb:.0f} MB.")


if __name__ == "__main__":
    main()
