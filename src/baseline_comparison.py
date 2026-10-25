r"""
baseline_comparison.py  (Component 7: do the embeddings beat simple baselines?)

Question answered: when predicting a catchment-state label from satellite data
on dates the model has NOT seen, do the Clay embeddings do better than
  (a) the time of year alone (seasonal cycle), or
  (b) simple hand-crafted radar statistics (VV/VH in dB)?

If the embeddings cannot beat (a) and (b) out of sample, they are not adding
information for this task, however good they look in the verification plots.

WHAT IT DOES
  * Loads every embeddings_<station>_<sensor>_<year>.npz + indices_*.csv in
    --embeddings-dir (any number of stations/years) and joins the labels.
  * Optional TRACK ADJUSTMENT (Sentinel-1 only): dates are grouped by orbit
    track using the 12-day repeat (days since the station's first empty date,
    mod 12), and each station+track mean is subtracted from the features.
    This uses imagery only, never labels.
  * Cross-validation with held-out TIME blocks, never random rows, because
    neighbouring days are nearly identical. Training rows within --gap-days of
    a test block (same station) are dropped so the model cannot interpolate
    across the boundary.
  * Models (all ridge regression, regularisation chosen inside the training
    data only):
      season                  sin/cos of day of year
      stats                   VV/VH dB statistics
      stats [adj]             same, track-adjusted
      emb ridge               all 1024 embedding dims
      emb ridge [adj]
      emb PCA10               first 10 principal components of the embedding
      emb PCA10 [adj]
      stats + emb PCA10 [adj]
  * Reports out-of-fold Spearman rho, R^2 and RMSE per label, the fold-to-fold
    spread, and in how many folds each embedding model beats "stats [adj]".

CV SCHEMES (--cv)
  blocked  each station's year is cut into --k contiguous time blocks; fold i
           holds out block i of every station. Works with one station.
  loyo     leave one year out (needs >= 2 years).
  loso     leave one station out (needs >= 2 stations). The hardest and most
           honest test of "does this transfer to a catchment it never saw".

Usage:
    python baseline_comparison.py --embeddings-dir embeddings_permanent \
        --labels antecedent_state_labels.csv --sensor s1 --cv blocked

    python baseline_comparison.py ... --cv loso --out-csv results.csv

Requires: numpy, pandas, scikit-learn, scipy(optional), matplotlib(optional)
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_COLS = ["level_percentile", "standardized_anomaly", "baseflow_index",
              "recession_rate_7d", "days_since_last_peak"]
STAT_BASE = ["vv_db_mean", "vh_db_mean", "vv_db_std", "vh_db_std", "vv_db_median", "vh_db_median"]
ALPHAS = np.logspace(-2, 6, 33)
REPEAT_DAYS = 12


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_runs(emb_dir, sensor, stations=None, years=None):
    pat = re.compile(rf"embeddings_(.+)_{sensor}_(\d{{4}})\.npz$")
    frames, embs = [], []
    for path in sorted(glob.glob(os.path.join(emb_dir, f"embeddings_*_{sensor}_*.npz"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        station, year = m.group(1), int(m.group(2))
        if stations and station not in stations:
            continue
        if years and year not in years:
            continue
        idx_path = os.path.join(emb_dir, f"indices_{station}_{sensor}_{year}.csv")
        if not os.path.exists(idx_path):
            print(f"  skipping {os.path.basename(path)}: no matching {os.path.basename(idx_path)}")
            continue
        npz = np.load(path)
        idx = pd.read_csv(idx_path, dtype={"date": str})
        dates = sorted(set(npz.files) & set(idx["date"]))
        if not dates:
            continue
        idx = idx.set_index("date").loc[dates]
        E = np.stack([npz[d] for d in dates]).astype("float64")
        df = idx.reset_index()
        df.insert(0, "station_id", station)
        df["year"] = year
        frames.append(df)
        embs.append(E)
    if not frames:
        sys.exit(f"No embeddings_*_{sensor}_*.npz + indices_*.csv pairs found in {emb_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df, np.concatenate(embs, axis=0)


def add_track_phase(df, E, sensor):
    """Orbit-track id per row = (days since the station's first empty date) mod 12.
    Only meaningful for Sentinel-1 with its 12-day repeat; -1 means 'unknown'."""
    df["phase"] = -1
    if sensor != "s1":
        return
    empty = np.isnan(E).all(axis=1)
    for st, g in df.groupby("station_id"):
        e = empty[g.index]
        if e.sum() < 3:
            print(f"  station {st}: fewer than 3 empty dates, cannot anchor tracks -- no track adjustment")
            continue
        empty_dates = g["date"][e].sort_values()
        gaps = np.diff(empty_dates.values).astype("timedelta64[D]").astype(int)
        share = float((gaps == REPEAT_DAYS).mean()) if len(gaps) else 0.0
        if share < 0.6:
            print(f"  station {st}: only {share:.0%} of empty-date gaps are {REPEAT_DAYS} days -- "
                  f"track grouping unreliable (cadence change or irregular coverage); no track adjustment")
            continue
        ref = empty_dates.iloc[0]
        df.loc[g.index, "phase"] = ((g["date"] - ref).dt.days % REPEAT_DAYS).to_numpy()


def adjust_by_track(X, df):
    """Subtract the mean of each (station, track) group. Imagery only, no labels."""
    Xa = X.copy()
    key = df["station_id"].astype(str) + "|" + df["phase"].astype(str)
    for k in key.unique():
        m = (key == k).to_numpy()
        if df.loc[m, "phase"].iloc[0] == -1:
            continue                      # unknown track: leave unadjusted
        Xa[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return Xa


# --------------------------------------------------------------------------
# Cross-validation folds
# --------------------------------------------------------------------------

def make_folds(df, scheme, k, gap_days):
    n = len(df)
    fold_of = np.full(n, -1)
    if scheme == "blocked":
        for st, g in df.groupby("station_id"):
            order = g.sort_values("date").index.to_numpy()
            blocks = (np.arange(len(order)) * k // len(order))
            fold_of[order] = blocks
        n_folds = k
    elif scheme == "loyo":
        years = sorted(df["year"].unique())
        if len(years) < 2:
            sys.exit("--cv loyo needs at least 2 years of data")
        fold_of = df["year"].map({y: i for i, y in enumerate(years)}).to_numpy()
        n_folds = len(years)
    elif scheme == "loso":
        stations = sorted(df["station_id"].unique())
        if len(stations) < 2:
            sys.exit("--cv loso needs at least 2 stations")
        fold_of = df["station_id"].map({s: i for i, s in enumerate(stations)}).to_numpy()
        n_folds = len(stations)
    else:
        raise ValueError(scheme)

    folds = []
    for f in range(n_folds):
        test = np.where(fold_of == f)[0]
        train_mask = fold_of != f
        if scheme != "loso" and gap_days > 0:
            # purge training rows of the same station near the held-out dates
            for st in df.loc[test, "station_id"].unique():
                t_dates = df.loc[test][df.loc[test, "station_id"] == st]["date"]
                lo = t_dates.min() - pd.Timedelta(days=gap_days)
                hi = t_dates.max() + pd.Timedelta(days=gap_days)
                near = ((df["station_id"] == st) & (df["date"] >= lo) & (df["date"] <= hi)).to_numpy()
                train_mask &= ~near
        folds.append((np.where(train_mask)[0], test))
    return folds


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def ridge(steps_front):
    return Pipeline(steps_front + [("ridge", RidgeCV(alphas=ALPHAS))])


def build_models(n_stats, n_emb):
    """name -> (feature-set key, estimator factory)."""
    def pca10():
        return ridge([("sc", StandardScaler()), ("pca", PCA(n_components=10, random_state=0))])

    def combo():
        ct = ColumnTransformer([
            ("stats", StandardScaler(), list(range(n_stats))),
            ("emb", Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=10, random_state=0))]),
             list(range(n_stats, n_stats + n_emb))),
        ])
        return Pipeline([("ct", ct), ("ridge", RidgeCV(alphas=ALPHAS))])

    return {
        "season":                  ("season",    lambda: ridge([("sc", StandardScaler())])),
        "stats":                   ("stats",     lambda: ridge([("sc", StandardScaler())])),
        "stats [adj]":             ("stats_adj", lambda: ridge([("sc", StandardScaler())])),
        "emb ridge":               ("emb",       lambda: ridge([("sc", StandardScaler())])),
        "emb ridge [adj]":         ("emb_adj",   lambda: ridge([("sc", StandardScaler())])),
        "emb PCA10":               ("emb",       pca10),
        "emb PCA10 [adj]":         ("emb_adj",   pca10),
        "stats + emb PCA10 [adj]": ("combo_adj", combo),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def r2(y, p):
    ss = ((y - p) ** 2).sum()
    tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss / tot) if tot > 0 else np.nan


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare embeddings against season-only and VV/VH baselines, out of sample.")
    ap.add_argument("--embeddings-dir", required=True)
    ap.add_argument("--labels", required=True, help="antecedent_state_labels.csv")
    ap.add_argument("--sensor", default="s1", choices=["s1", "s2"])
    ap.add_argument("--targets", nargs="*", default=None, help="label columns (default: all present)")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--cv", default="blocked", choices=["blocked", "loyo", "loso"])
    ap.add_argument("--k", type=int, default=5, help="number of blocks for --cv blocked")
    ap.add_argument("--gap-days", type=int, default=14)
    ap.add_argument("--out-csv", default="baseline_comparison.csv")
    ap.add_argument("--out-png", default="baseline_comparison.png")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading...")
    df, E = load_runs(args.embeddings_dir, args.sensor, args.stations, args.years)
    add_track_phase(df, E, args.sensor)

    missing = [c for c in STAT_BASE if c not in df.columns]
    if missing:
        sys.exit(f"indices csv files lack {missing}. They were written by an older extract_embeddings.py -- "
                 f"re-extract with the current one (run_catchment_pipeline.py --reprocess).")

    usable = np.isfinite(E).all(axis=1) & df[STAT_BASE].notna().all(axis=1).to_numpy()
    print(f"  {len(df)} dates across {df['station_id'].nunique()} station(s), "
          f"{int(usable.sum())} usable (finite embedding and stats)")
    df, E = df[usable].reset_index(drop=True), E[usable]

    # labels
    lab = pd.read_csv(args.labels, parse_dates=["date"])
    lab["station_id"] = lab["station_id"].astype(str)
    targets = args.targets or [c for c in LABEL_COLS if c in lab.columns]
    if not targets:
        sys.exit("none of the expected label columns are in --labels")
    df = df.merge(lab[["station_id", "date"] + targets], on=["station_id", "date"], how="left")
    print(f"  targets: {targets}")

    # feature sets
    stats = df[STAT_BASE].to_numpy(float)
    stats = np.column_stack([stats, df["vv_db_mean"] - df["vh_db_mean"]])
    doy = df["date"].dt.dayofyear.to_numpy()
    season = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    has_track = (df["phase"] >= 0).any()
    stats_adj = adjust_by_track(stats, df) if has_track else stats
    E_adj = adjust_by_track(E, df) if has_track else E
    if not has_track:
        print("  no usable track grouping -> [adj] models equal the unadjusted ones")
    feats = {"season": season, "stats": stats, "stats_adj": stats_adj, "emb": E, "emb_adj": E_adj,
             "combo_adj": np.column_stack([stats_adj, E_adj])}

    models = build_models(stats.shape[1], E.shape[1])
    folds = make_folds(df, args.cv, args.k, args.gap_days)
    print(f"  CV: {args.cv}, {len(folds)} folds, gap {args.gap_days} d "
          f"(train sizes {[len(a) for a, _ in folds]}, test sizes {[len(b) for _, b in folds]})")

    rows = []
    fold_rmse = {}
    for target in targets:
        y_all = df[target].to_numpy(float)
        ok = np.isfinite(y_all)
        if ok.sum() < 20:
            print(f"\n[{target}] only {int(ok.sum())} labelled usable dates -- skipped")
            continue
        for name, (fkey, make) in models.items():
            X = feats[fkey]
            oof = np.full(len(df), np.nan)
            per_fold = []
            for fi, (tr, te) in enumerate(folds):
                tr = tr[ok[tr]]
                te = te[ok[te]]
                if len(tr) < 15 or len(te) < 5:
                    per_fold.append((np.nan, np.nan))
                    continue
                est = make()
                est.fit(X[tr], y_all[tr])
                p = est.predict(X[te])
                oof[te] = p
                per_fold.append((spearman(y_all[te], p), rmse(y_all[te], p)))
            have = np.isfinite(oof) & ok
            if have.sum() < 10:
                continue
            fr = np.array([x[0] for x in per_fold], float)
            fm = np.array([x[1] for x in per_fold], float)
            fold_rmse[(target, name)] = fm
            rows.append({"target": target, "model": name, "n": int(have.sum()),
                         "rho_pooled": spearman(y_all[have], oof[have]),
                         "rho_fold_mean": float(np.nanmean(fr)), "rho_fold_sd": float(np.nanstd(fr)),
                         "r2": r2(y_all[have], oof[have]), "rmse": rmse(y_all[have], oof[have])})

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("no results (too few labelled dates)")

    # relative to season, and fold wins vs the stats baseline
    ref_stats = "stats [adj]" if has_track else "stats"
    for i, r in res.iterrows():
        s = res[(res.target == r.target) & (res.model == "season")]
        res.loc[i, "rmse_vs_season_%"] = 100 * (r.rmse / s.rmse.iloc[0] - 1) if len(s) else np.nan
        a, b = fold_rmse.get((r.target, r.model)), fold_rmse.get((r.target, ref_stats))
        if a is not None and b is not None and r.model != ref_stats:
            m = np.isfinite(a) & np.isfinite(b)
            res.loc[i, "folds_better_than_stats"] = f"{int((a[m] < b[m]).sum())}/{int(m.sum())}"
    res.to_csv(args.out_csv, index=False)

    # ---- report -----------------------------------------------------------
    order = list(models)
    for target in res.target.unique():
        t = res[res.target == target].set_index("model").reindex([m for m in order if m in set(res[res.target == target].model)])
        print(f"\n=== {target}  (out-of-fold, n={int(t['n'].iloc[0])}) ===")
        print(f"{'model':<26}{'rho':>8}{'rho folds':>14}{'R2':>8}{'RMSE':>9}{'vs season':>11}{'beats ' + ref_stats:>20}")
        for m, r in t.iterrows():
            fb = r.get("folds_better_than_stats", "")
            fb = fb if isinstance(fb, str) else ""
            print(f"{m:<26}{r.rho_pooled:>+8.2f}{r.rho_fold_mean:>+8.2f}±{r.rho_fold_sd:<4.2f}"
                  f"{r.r2:>+8.2f}{r.rmse:>9.3f}{r['rmse_vs_season_%']:>+10.1f}%{fb:>20}")

    print("\n=== Summary ===")
    for target in res.target.unique():
        t = res[res.target == target].set_index("model")
        emb_models = [m for m in t.index if m.startswith("emb") or "emb" in m]
        best_emb = t.loc[emb_models, "rmse"].idxmin() if emb_models else None
        base_names = [m for m in ("season", "stats", "stats [adj]") if m in t.index]
        best_base = t.loc[base_names, "rmse"].idxmin()
        if best_emb is None:
            continue
        gain = 100 * (1 - t.loc[best_emb, "rmse"] / t.loc[best_base, "rmse"])
        verdict = ("embeddings beat the best simple baseline" if gain > 5 else
                   "no clear gain over the best simple baseline" if gain > -5 else
                   "embeddings are WORSE than the best simple baseline")
        print(f"  {target:<22} best embedding model '{best_emb}' vs best baseline '{best_base}': "
              f"RMSE {gain:+.1f}% -> {verdict}")
    print("\n  Rules of thumb: a gain under ~5% or one that appears in fewer than ~4 of 5 folds is within noise at this\n"
          "  sample size. 'season' being hard to beat means the label is mostly the annual cycle. Positive R2 on\n"
          "  held-out blocks is the minimum bar. A NEGATIVE pooled rho for a weak model is a known artefact of\n"
          "  leave-block-out CV (a block with high labels leaves a low training mean), read it as 'no signal'.\n"
          "  Treat one station-year as a harness test, not a result.")

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tg = list(res.target.unique())
        fig, axes = plt.subplots(1, len(tg), figsize=(4.6 * len(tg), 4.2), squeeze=False, sharey=True)
        mods = [m for m in order if m in set(res.model)]
        for ax, target in zip(axes[0], tg):
            t = res[res.target == target].set_index("model").reindex(mods)
            colors = ["#9aa0a6" if m.startswith("season") else "#1f77b4" if m.startswith("stats") and "emb" not in m
                      else "#ff7f0e" for m in mods]
            ax.barh(range(len(mods)), t["rho_pooled"], xerr=t["rho_fold_sd"], color=colors, height=0.65,
                    error_kw={"elinewidth": 1, "capsize": 2})
            ax.set_yticks(range(len(mods)))
            ax.set_yticklabels(mods, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(target, fontsize=10)
            ax.set_xlabel("out-of-fold Spearman rho (bar = pooled, whisker = fold sd)", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Grey = season only, blue = VV/VH statistics, orange = embedding models", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=140)
        print(f"\nFigure written to {args.out_png}\nTable written to {args.out_csv}")
    except Exception as e:  # plotting is optional
        print(f"\n(figure skipped: {e})\nTable written to {args.out_csv}")


if __name__ == "__main__":
    main()
