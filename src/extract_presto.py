r"""
extract_presto.py -- Presto embeddings from the pixel cache (no raw imagery needed).

Presto (Tseng et al., NASA Harvest; github.com/nasaharvest/presto) encodes the TIME SERIES
of one location, not a single image. This script builds that series from the pixel cache
that the extraction pass writes (pixel_cache.py, results/cache/pixels_*.npz) and writes
embeddings in exactly the same file format as the Clay ones, so baseline_comparison.py
reads them unchanged:

    results/embeddings_presto/embeddings_<station>_<sensor>_<year>.npz   date -> 128-d vector
    results/embeddings_presto/indices_<station>_<sensor>_<year>.csv      per-date statistics
                                                                         + presto_* columns

HOW ONE DATE'S EMBEDDING IS MADE (for target date t)
  1. Look back --steps x --bin-days days from t (default 12 x 30 = 360 days). The newest
     bin is the 30 days ending at t. The look-back crosses years: the previous year's cache
     is used if it exists.   Only dates <= t are used (causal: nothing from the future).
  2. At each sampled pixel, average the cached VV and VH dB over each bin. A bin with no
     valid observation at that pixel is MASKED, so Presto ignores it.
  3. Presto input (verified against presto/dataops/pipelines/s1_s2_era5_srtm.py):
     Sentinel-1 only, in dB, scaled (dB + 25) / 25; every other band (Sentinel-2, weather,
     elevation, NDVI) masked; Dynamic World class 9 = "missing"; location = the pixel's
     (lat, lon); starting month = calendar month of the oldest bin.
  4. Presto returns a 128-d vector per pixel (mean of the unmasked tokens through its
     2-layer transformer). The date's embedding is the mean over the pixels.

ADAPTATIONS TO KNOW ABOUT (these are my choices, not Presto's design)
  * Presto was trained on MONTHLY composites; here the bins are 30 days ending at t, not
    calendar months, so the newest bin is never a partial month. The month label drifts by
    at most a few days over 12 bins.
  * Sentinel-1 only. Presto was pre-trained mostly with Sentinel-2, weather and terrain
    present, so a radar-only input is a harder case for it than the one it was built for.
  * Pixel series are speckle-reduced (3x3 window, bin averages) before encoding.
  * The embedding is the same for any label: it is not trained on your task. Only the ridge
    on top of it is.
  * A date is skipped (NaN embedding, dropped later by the baseline) if its own observation
    is missing at too many pixels (--min-series), and a pixel series is skipped if fewer than
    --min-valid-steps of its bins have data. The first weeks of the very first year have little
    history, so they have few valid bins; presto_steps_mean in the indices file shows it.

USAGE
    python extract_presto.py --sensor s1                       # every station found in the cache
    python extract_presto.py --stations 11001 22009 --years 2023
    python extract_presto.py --presto-checkpoint checkpoints/presto/presto-main/data/default_model.pt

    then:  python baseline_comparison.py --encoder presto --labels ... --cv loso

Requires: numpy, pandas, torch, einops and the presto package (pip install
git+https://github.com/nasaharvest/presto.git) or its single_file_presto.py on the path.
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from pixel_cache import load_cache, verify_cache
from project_paths import Layout, ensure_parent

REF_DATE = pd.Timestamp("2015-01-01")
# --- Presto constants, verified against presto/dataops/pipelines/s1_s2_era5_srtm.py -------------
S1_SHIFT, S1_DIV = 25.0, 25.0        # S1 values are dB; normalised as (dB + 25) / 25
N_BANDS = 17                         # len(NORMED_BANDS); S1 VV, VH are columns 0 and 1
S1_COLS = (0, 1)
DW_MISSING = 9                       # DynamicWorld2020_2021.class_amount: "no class", ignored
DEFAULT_CKPT = "checkpoints/presto/presto-main/data/default_model.pt"


# --------------------------------------------------------------------------
# Loading the cache for one station (all years, same sampled pixels)
# --------------------------------------------------------------------------

def find_cache_units(cache_dir, sensor):
    """{station: [years]} for every verified pixels_*.npz in the cache."""
    pat = re.compile(rf"^pixels_(.+)_{sensor}_(\d{{4}})\.npz$")
    out = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, f"pixels_*_{sensor}_*.npz"))):
        m = pat.match(os.path.basename(path))
        if m:
            out.setdefault(m.group(1), []).append(int(m.group(2)))
    return out


def load_station(cache_dir, station, sensor, years):
    """Concatenate a station's yearly caches into one time-ordered record.
    Years are only joined if their sampled pixel coordinates are identical; otherwise
    each coordinate set is returned as its own group (history is not shared across them)."""
    groups = []                                   # each: dict(x, y, lon, lat, parts=[(year, cache, stats)])
    for year in sorted(years):
        npz = os.path.join(cache_dir, f"pixels_{station}_{sensor}_{year}.npz")
        stats = os.path.join(cache_dir, f"stats_{station}_{sensor}_{year}.csv")
        ok, detail = verify_cache(npz, stats)
        if not ok:
            print(f"  station {station} {year}: cache not usable ({detail}) -- skipped")
            continue
        c = load_cache(npz)
        if not c["meta"].get("db", sensor == "s1"):
            raise ValueError(f"{npz}: the cache is not in dB; this adapter expects Sentinel-1 dB")
        for g in groups:
            if len(g["x"]) == len(c["x"]) and np.allclose(g["x"], c["x"]) and np.allclose(g["y"], c["y"]):
                g["parts"].append((year, c, pd.read_csv(stats)))
                break
        else:
            groups.append({"x": c["x"], "y": c["y"], "lon": c["lon"], "lat": c["lat"],
                           "parts": [(year, c, pd.read_csv(stats))]})
    if len(groups) > 1:
        print(f"  station {station}: the sampled pixels differ between years; "
              f"{len(groups)} separate groups, history is not shared across them")
    out = []
    for g in groups:
        dates = np.concatenate([c["dates"] for _, c, _ in g["parts"]])
        values = np.concatenate([c["values"] for _, c, _ in g["parts"]], axis=0)
        years_col = np.concatenate([[y] * len(c["dates"]) for y, c, _ in g["parts"]])
        stats = pd.concat([s.assign(year=y) for y, _, s in g["parts"]], ignore_index=True)
        order = np.argsort(dates, kind="stable")
        dates, values, years_col = dates[order], values[order], years_col[order]
        stats = stats.iloc[order].reset_index(drop=True)
        days = np.array([(pd.Timestamp(str(d)) - REF_DATE).days for d in dates])
        out.append({"dates": dates, "days": days, "values": values, "years": years_col,
                    "stats": stats, "lat": g["lat"], "lon": g["lon"]})
    return out


# --------------------------------------------------------------------------
# Series construction (numpy only, testable without torch)
# --------------------------------------------------------------------------

def bin_history(days, values, i, steps, bin_days):
    """Bin the series for target index i. Returns (binned (N, steps, 2) dB, valid (N, steps)).
    Step steps-1 is the newest bin: the bin_days days ending at day days[i]. Only
    observations with day <= days[i] are used."""
    t = days[i]
    age = t - days
    use = (age >= 0) & (age < steps * bin_days)
    N = values.shape[1]
    binned = np.full((N, steps, 2), np.nan, dtype="float32")
    if not use.any():
        return binned, np.zeros((N, steps), bool)
    step_of = steps - 1 - (age // bin_days)
    for s in range(steps):
        sel = use & (step_of == s)
        if not sel.any():
            continue
        v = values[sel]                                         # (k, N, 2)
        ok = np.isfinite(v)
        cnt = ok.sum(axis=0)
        tot = np.where(ok, v, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned[:, s, :] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    valid = np.isfinite(binned).all(axis=2)
    return binned, valid


def build_inputs(binned, valid):
    """Presto inputs for a batch of S1 series: x (N,T,17) float32, mask (N,T,17) float32
    (1 = masked), dynamic_world (N,T) int64."""
    N, T, _ = binned.shape
    x = np.zeros((N, T, N_BANDS), dtype="float32")
    mask = np.ones((N, T, N_BANDS), dtype="float32")
    norm = (np.where(valid[..., None], binned, 0.0) + S1_SHIFT) / S1_DIV
    for k, col in enumerate(S1_COLS):
        x[:, :, col] = np.where(valid, norm[:, :, k], 0.0)
        mask[:, :, col] = (~valid).astype("float32")
    dw = np.full((N, T), DW_MISSING, dtype="int64")
    return x, mask, dw


def start_month(t_day, steps, bin_days):
    """0-based calendar month of the oldest bin's midpoint (Presto's `month` argument)."""
    mid = t_day - (steps - 0.5) * bin_days
    return int((REF_DATE + pd.Timedelta(days=float(mid))).month - 1)


# --------------------------------------------------------------------------
# Presto
# --------------------------------------------------------------------------

class PrestoRunner:
    def __init__(self, checkpoint=DEFAULT_CKPT):
        import torch
        self.torch = torch
        self.library_inputs = None
        try:
            from presto import Presto
            import presto as _presto
            self.library_inputs = getattr(_presto, "construct_batch_presto_input", None)
            origin = "presto package"
        except ImportError:
            try:
                from single_file_presto import Presto
                origin = "single_file_presto.py"
            except ImportError:
                raise RuntimeError(
                    "Presto is not importable. Install it with "
                    "pip install git+https://github.com/nasaharvest/presto.git, or put "
                    "single_file_presto.py (from the repo) next to this script.")
        # Same rule the presto package uses internally (presto/utils.py `device`).
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if checkpoint and os.path.exists(checkpoint):
            model = Presto.construct()
            model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            print(f"  Presto loaded from {origin}, weights {checkpoint}")
        elif hasattr(Presto, "load_pretrained"):
            model = Presto.load_pretrained()
            print(f"  Presto loaded from {origin} (bundled weights); no file at {checkpoint}")
        else:
            raise RuntimeError(f"Presto weights not found at {checkpoint}")
        model.to(self.device)
        self.encoder = model.encoder.eval()
        self.dim = 128

    def crosscheck_inputs(self):
        """If the presto package is available, check that build_inputs() reproduces the
        package's own construct_batch_presto_input() on random data. Raises on a mismatch,
        so a change in Presto's normalisation cannot silently corrupt the embeddings."""
        if self.library_inputs is None:
            print("  (package input constructor not available; the input format was checked against the source)")
            return
        torch = self.torch
        rng = np.random.default_rng(0)
        db = rng.uniform(-30, 0, (5, 12, 2)).astype("float32")
        valid = rng.random((5, 12)) > 0.3
        x, mask, dw = build_inputs(np.where(valid[..., None], db, np.nan), valid)
        lx, lm, ldw = self.library_inputs(s1=torch.from_numpy(db), s1_bands=["VV", "VH"])
        lm = lm.clone()
        lm[:, :, list(S1_COLS)] = torch.from_numpy((~valid).astype("float32"))[:, :, None]
        lx = lx.clone()
        lx[:, :, list(S1_COLS)] = torch.where(torch.from_numpy(valid)[:, :, None], lx[:, :, list(S1_COLS)],
                                              torch.zeros(1))
        # Only the S1 columns carry data; the other columns are masked, so their placeholder values
        # (zeros here, normalised zeros in the library) are never read by the encoder.
        c = list(S1_COLS)
        same = (np.allclose(x[:, :, c], lx.numpy()[:, :, c], atol=1e-6) and np.array_equal(mask, lm.numpy())
                and np.array_equal(dw, ldw.long().numpy()))
        if not same:
            raise RuntimeError("build_inputs() disagrees with presto's construct_batch_presto_input(); "
                               "Presto's input format changed -- do not trust the embeddings until this is fixed")
        print("  input construction matches presto's own constructor")

    def embed(self, x, mask, dw, latlon, month, chunk=1024):
        torch = self.torch
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), chunk):
                sl = slice(s, s + chunk)
                o = self.encoder(torch.from_numpy(x[sl]).to(self.device),
                                 torch.from_numpy(dw[sl]).long().to(self.device),
                                 torch.from_numpy(latlon[sl]).float().to(self.device),
                                 torch.from_numpy(mask[sl]).to(self.device), int(month), eval_task=True)
                outs.append(o.cpu().numpy())
        return np.concatenate(outs, axis=0)


def embed_date(runner, rec, i, steps, bin_days, min_valid_steps, min_series):
    """-> (embedding (128,) or NaN, n_series, mean valid steps)"""
    nan = np.full(128, np.nan, dtype="float32")
    if np.isfinite(rec["values"][i]).all(axis=1).sum() < min_series:
        return nan, 0, 0.0                              # no observation on the target date itself
    binned, valid = bin_history(rec["days"], rec["values"], i, steps, bin_days)
    keep = valid.sum(axis=1) >= min_valid_steps
    if keep.sum() < min_series:
        return nan, int(keep.sum()), float(valid.sum(axis=1)[keep].mean()) if keep.any() else 0.0
    idx = np.flatnonzero(keep)
    month = start_month(rec["days"][i], steps, bin_days)
    # Group by missing-data pattern: some Presto versions require every series in a batch
    # to have the same number of unmasked tokens.
    patterns = {}
    for j in idx:
        patterns.setdefault(valid[j].tobytes(), []).append(j)
    vecs, weights = [], []
    for members in patterns.values():
        m = np.array(members)
        x, mask, dw = build_inputs(binned[m], valid[m])
        latlon = np.stack([rec["lat"][m], rec["lon"][m]], axis=1)
        vecs.append(runner.embed(x, mask, dw, latlon, month))
        weights.append(len(m))
    emb = np.concatenate(vecs, axis=0).mean(axis=0).astype("float32")
    return emb, int(keep.sum()), float(valid.sum(axis=1)[keep].mean())


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def verify_output(emb_path, idx_path, min_dates=5):
    if not (os.path.exists(emb_path) and os.path.exists(idx_path)):
        return False
    try:
        z = np.load(emb_path)
        n = len(z.files)
        idx = pd.read_csv(idx_path)
        finite = sum(1 for k in z.files if np.isfinite(z[k]).all())
    except Exception:
        return False
    return n >= min_dates and len(idx) == n and finite > 0


def run_station(runner, cache_dir, out_dir, station, sensor, years, args):
    groups = load_station(cache_dir, station, sensor, years)
    for rec in groups:
        todo_years = sorted(set(int(y) for y in rec["years"]))
        if args.years:
            todo_years = [y for y in todo_years if y in set(args.years)]
        for year in todo_years:
            out_emb = os.path.join(out_dir, f"embeddings_{station}_{sensor}_{year}.npz")
            out_idx = os.path.join(out_dir, f"indices_{station}_{sensor}_{year}.csv")
            if not args.reprocess and verify_output(out_emb, out_idx):
                print(f"  station {station} {year}: verified Presto output exists -- skipping (--reprocess to redo)")
                continue
            rows = np.flatnonzero(rec["years"] == year)
            embs, info = {}, []
            for n, i in enumerate(rows):
                e, n_series, steps_mean = embed_date(runner, rec, i, args.steps, args.bin_days,
                                                     args.min_valid_steps, args.min_series)
                embs[str(rec["dates"][i])] = e
                info.append((n_series, steps_mean))
                if (n + 1) % 25 == 0 or n + 1 == len(rows):
                    print(f"    station {station} {year}: {n + 1}/{len(rows)} dates")
            stats = rec["stats"].iloc[rows].drop(columns=["year"], errors="ignore").reset_index(drop=True).copy()
            stats["presto_series"] = [a for a, _ in info]
            stats["presto_steps_mean"] = np.round([b for _, b in info], 2)
            tmp_emb, tmp_idx = out_emb[:-4] + ".tmp.npz", out_idx[:-4] + ".tmp.csv"
            np.savez_compressed(tmp_emb, **embs)
            stats.to_csv(tmp_idx, index=False)
            if not verify_output(tmp_emb, tmp_idx):
                print(f"  station {station} {year}: FAILED verification, output left at {tmp_emb}; nothing replaced")
                continue
            os.replace(tmp_emb, out_emb)
            os.replace(tmp_idx, out_idx)
            n_ok = sum(bool(np.isfinite(v).all()) for v in embs.values())
            print(f"  station {station} {year}: {n_ok}/{len(embs)} dates embedded -> {out_emb}")


def main():
    ap = argparse.ArgumentParser(description="Presto embeddings from the pixel cache.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--cache-dir", default=None, help="default: <results root>/cache")
    ap.add_argument("--out-dir", default=None, help="default: <results root>/embeddings_presto")
    ap.add_argument("--sensor", default="s1", choices=["s1", "s2"])
    ap.add_argument("--stations", nargs="*", default=None, help="default: every station in the cache")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="years to WRITE (history from other years is still used)")
    ap.add_argument("--steps", type=int, default=12, help="time steps per series (Presto accepts 1-24; default 12)")
    ap.add_argument("--bin-days", type=int, default=30, help="days per step (default 30, i.e. monthly)")
    ap.add_argument("--min-valid-steps", type=int, default=2, help="skip pixel series with fewer valid bins (default 2)")
    ap.add_argument("--min-series", type=int, default=20, help="skip a date with fewer usable pixel series (default 20)")
    ap.add_argument("--presto-checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--reprocess", action="store_true")
    args = ap.parse_args()

    if args.sensor != "s1":
        sys.exit("Only Sentinel-1 is supported by this adapter so far (the cache holds S2 bands, "
                 "but the Presto input for S2 has not been built or tested).")
    if not 1 <= args.steps <= 24:
        sys.exit("--steps must be between 1 and 24 (Presto's maximum sequence length)")
    warnings.filterwarnings("ignore", category=UserWarning)

    lay = Layout(args.results_root)
    cache_dir = args.cache_dir or lay.cache
    out_dir = ensure_parent(os.path.join(args.out_dir or lay.embeddings_for("presto"), "x"))[:-1]
    units = find_cache_units(cache_dir, args.sensor)
    if args.stations:
        units = {s: y for s, y in units.items() if s in set(args.stations)}
    if not units:
        sys.exit(f"No pixel caches found in {cache_dir} (run run_batch.py first; it writes them).")
    print(f"Cache: {cache_dir}\nPresto embeddings -> {out_dir}\n{len(units)} station(s): {', '.join(sorted(units))}\n")

    runner = PrestoRunner(args.presto_checkpoint)
    runner.crosscheck_inputs()
    for station in sorted(units):
        run_station(runner, cache_dir, out_dir, station, args.sensor, units[station], args)
    print(f"\nNext: python baseline_comparison.py --encoder presto --sensor {args.sensor} "
          f"--labels <antecedent_state_labels.csv> --cv loso")


if __name__ == "__main__":
    main()
