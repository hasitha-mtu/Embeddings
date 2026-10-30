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
  * STATION/TRACK NORMALISATION ("[norm]" models). Labels such as
    level_percentile and standardized_anomaly are relative to each station's
    own record, but raw radar values and embeddings carry station-specific
    offsets (land cover, geometry). A model trained on other catchments then
    mis-sets the baseline on a new one: good within-station correlation, bad
    R^2. So "[norm]" features are expressed relative to the catchment's own
    typical imagery: for Sentinel-1, dates are grouped by orbit track (the
    12-day repeat: (date - fixed reference date) mod 12, the same schedule for
    every catchment), each station+track mean is removed, then each station's
    spread is scaled to 1. This uses imagery only, never labels, and is
    available for any catchment with a satellite history. "[raw]" models use
    features as extracted.
  * Cross-validation with held-out TIME blocks, never random rows, because
    neighbouring days are nearly identical. Training rows within --gap-days of
    a test block (same station) are dropped so the model cannot interpolate
    across the boundary.
  * Models (all ridge regression; the penalty is chosen inside the training
    data only, by contiguous time-block CV -- --alpha-cv loo restores sklearn's
    leave-one-out, which under-regularises on near-duplicate daily images):
      season                           sin/cos of day of year
      stats [raw] / [norm]             VV/VH dB statistics
      emb PCA10 [raw] / [norm]         first 10 principal components of the embedding
      emb ridge [norm]                 all 1024 embedding dims
      season + stats [norm]            calendar AND radar statistics
      season + emb PCA10 [norm]        calendar AND embeddings
      season + stats + emb PCA10 [norm]
      season + emb ridge [norm]        calendar + ALL 1024 embedding dims
      season + stats + emb ridge [norm]
    The "season + ..." models answer the question that matters: does the
    satellite add information BEYOND the time of year? (Many labels are
    largely seasonal.) When calendar, stats and a full-width embedding are
    combined in one ridge, the embedding block is scaled so that, as a block,
    it carries the same total variance as the stats block; otherwise its 1024
    columns would swamp the handful of stats columns.
  * TEMPORAL CONTEXT ("ctx" models). A catchment's antecedent state (baseflow
    index, recession, time since the last peak) depends on the previous weeks,
    which a single image cannot show. "ctx" models add, for every date, the
    mean of the station-normalised features over the PAST --ctx-windows days
    (default 14, 30, 60), strictly before that date. Imagery only, never labels,
    and never future dates, so it is what a real-time estimate could use:
      stats + ctx [norm]               stats now + stats averaged over past windows
      season + stats + ctx [norm]
      emb ridge + ctx [norm]           embedding now + embedding averaged over past windows
      season + emb ridge + ctx [norm]
    Rows with no acquisition in a window get that window's station mean (0 after
    normalisation); the share of such rows is printed.
  * Reports out-of-fold Spearman rho, R^2 and RMSE per label, the fold-to-fold
    spread, and in how many folds each embedding model beats its matching
    statistics-only model.

CV SCHEMES (--cv)
  blocked  each station's year is cut into --k contiguous time blocks; fold i
           holds out block i of every station. Works with one station.
  loyo     leave one year out (needs >= 2 years).
  loso     leave one station out (needs >= 2 stations). The hardest and most
           honest test of "does this transfer to a catchment it never saw".

Usage:
    python baseline_comparison.py --sensor s1 --cv blocked

    python baseline_comparison.py --cv loso --labels some/other/labels.csv
    python baseline_comparison.py --cv loso --ctx-windows 7 21 45 90

Reads <results root>/embeddings_<encoder> and <results root>/labels by default and writes everything for a run into
<results root>/baselines/<run-name>/ (baseline_comparison.csv, .png and
report.txt, a copy of the console output). Run name defaults to "<sensor>_<cv>_alpha-<blocked|loo>",
so different CV schemes and penalty rules never overwrite each other; set --run-name to keep
several runs of the same kind. --embeddings-dir, --out-csv and --out-png
override individual locations. See project_paths.py for the layout.

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
from sklearn.base import BaseEstimator, RegressorMixin
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
        E = np.stack([npz[d] for d in dates]).astype("float64")
        good = np.isfinite(E).all(axis=1)            # time-series encoders (Presto) leave NaN where history is too thin
        if not good.all():
            print(f"  {os.path.basename(path)}: dropping {int((~good).sum())} of {len(good)} dates with no usable embedding")
            dates = [d for d, g in zip(dates, good) if g]
            E = E[good]
            if not dates:
                continue
        idx = idx.set_index("date").loc[dates]
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


PHASE_REF = pd.Timestamp("2015-01-01")   # any fixed date: the 12-day schedule is global


def add_track_phase(df, E, sensor):
    """Orbit-track id per row = (days since a FIXED reference date) mod 12.
    Sentinel-1A/C repeat every 12 days, so the pass schedule is the same for
    every catchment; dates sharing a phase are the same relative orbit at that
    catchment. No empty dates are needed to anchor it. -1 = not applicable (S2)."""
    df["phase"] = -1
    if sensor != "s1":
        return
    df["phase"] = ((df["date"] - PHASE_REF).dt.days % REPEAT_DAYS).astype(int)
    usable = np.isfinite(E).all(axis=1)
    for st, g in df[usable].groupby("station_id"):
        n = g["phase"].nunique()
        note = "" if n <= 6 else "  (more than 6: a satellite constellation change or irregular dates?)"
        print(f"  station {st}: {n} orbit-track group(s) among usable dates{note}")


def normalise(X, df, by_track=True):
    """Make features relative to each catchment's own typical imagery:
    subtract the mean of each (station[, track]) group, then divide by the
    station's standard deviation per feature. Imagery only, no labels."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    st = df["station_id"].astype(str).to_numpy()
    ph = df["phase"].to_numpy()
    for s_id in np.unique(st):
        m = st == s_id
        Xm = X[m].copy()
        if by_track and (ph[m] >= 0).all():
            for k in np.unique(ph[m]):
                g = ph[m] == k
                Xm[g] = Xm[g] - Xm[g].mean(axis=0, keepdims=True)
        else:
            Xm = Xm - Xm.mean(axis=0, keepdims=True)
        sd = Xm.std(axis=0, keepdims=True)
        sd[sd < 1e-9] = 1.0
        out[m] = Xm / sd
    return out


def causal_context(X, df, windows):
    """For each row, the mean of X over the same station's rows dated in
    [t - W, t) for every W in windows (strictly before t, never the future).
    Returns (n, len(windows)*d) and the fraction of rows with an empty window,
    per window. Empty window -> 0 (the station mean, since X is normalised)."""
    X = np.asarray(X, float)
    n, d = X.shape
    out = np.zeros((n, len(windows) * d))
    empty = np.zeros(len(windows))
    days_all = ((df["date"] - PHASE_REF).dt.days).to_numpy()
    st = df["station_id"].astype(str).to_numpy()
    for s_id in np.unique(st):
        idx = np.where(st == s_id)[0]
        idx = idx[np.argsort(days_all[idx], kind="stable")]
        days = days_all[idx]
        C = np.vstack([np.zeros((1, d)), np.cumsum(X[idx], axis=0)])
        hi = np.searchsorted(days, days, side="left")           # rows strictly before t
        for wi, W in enumerate(windows):
            lo = np.searchsorted(days, days - W, side="left")
            cnt = hi - lo
            ok = cnt > 0
            m = np.zeros((len(idx), d))
            m[ok] = (C[hi[ok]] - C[lo[ok]]) / cnt[ok, None]
            out[idx, wi * d:(wi + 1) * d] = m
            empty[wi] += (~ok).sum()
    return out, empty / n


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

class BlockRidgeCV(BaseEstimator, RegressorMixin):
    """Ridge regression whose penalty is chosen by CONTIGUOUS-BLOCK cross-validation
    on the training rows (n_blocks blocks in row order, with gap_rows rows purged on
    each side of the validation block).

    Why not sklearn's RidgeCV? Its default is efficient leave-one-out, which on
    daily imagery is nearly in-sample (the day before and the day after are almost
    copies of the held-out day), so it picks too small a penalty, and the more
    columns the model has (full embeddings, rolling windows) the worse the
    over-fitting. Holding out whole stretches of time, as the outer CV does,
    selects a penalty that suits unseen time/catchments.
    Rows must be in time order within each station (they are: station-year blocks)."""

    def __init__(self, alphas=ALPHAS, n_blocks=5, gap_rows=5):
        self.alphas = alphas
        self.n_blocks = n_blocks
        self.gap_rows = gap_rows

    @staticmethod
    def _path_predict(Xtr, ytr, Xv, alphas):
        """Validation predictions for every alpha at once -> (n_val, n_alpha)."""
        xm, ym = Xtr.mean(axis=0), ytr.mean()
        Xc, yc, Xvc = Xtr - xm, ytr - ym, Xv - xm
        n, d = Xc.shape
        if d <= n:                                       # primal: eigen of X'X (d x d)
            lam, V = np.linalg.eigh(Xc.T @ Xc)
            b = V.T @ (Xc.T @ yc)
            coefs = V @ (b[:, None] / (np.clip(lam, 0, None)[:, None] + alphas[None, :]))
            return Xvc @ coefs + ym
        lam, U = np.linalg.eigh(Xc @ Xc.T)               # dual: eigen of XX' (n x n)
        c = U.T @ yc
        A = U @ (c[:, None] / (np.clip(lam, 0, None)[:, None] + alphas[None, :]))
        return (Xvc @ Xc.T) @ A + ym

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        alphas = np.asarray(self.alphas, float)
        n, d = X.shape
        edges = np.linspace(0, n, self.n_blocks + 1).astype(int)
        err, used = np.zeros(len(alphas)), 0
        for b in range(self.n_blocks):
            v = np.arange(edges[b], edges[b + 1])
            lo, hi = max(0, edges[b] - self.gap_rows), min(n, edges[b + 1] + self.gap_rows)
            tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
            if len(v) < 3 or len(tr) < max(10, d // 50):
                continue
            pred = self._path_predict(X[tr], y[tr], X[v], alphas)
            err += ((pred - y[v][:, None]) ** 2).sum(axis=0)
            used += 1
        self.alpha_ = float(alphas[int(np.argmin(err))]) if used else float(np.median(alphas))
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
        if d <= n:
            w = np.linalg.solve(Xc.T @ Xc + self.alpha_ * np.eye(d), Xc.T @ yc)
        else:
            w = Xc.T @ np.linalg.solve(Xc @ Xc.T + self.alpha_ * np.eye(n), yc)
        self.coef_, self.intercept_ = w, float(ym - xm @ w)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


ALPHA_CV = "blocked"          # set from --alpha-cv in main(); "loo" = sklearn RidgeCV (previous behaviour)


def final_ridge():
    return BlockRidgeCV(alphas=ALPHAS) if ALPHA_CV == "blocked" else RidgeCV(alphas=ALPHAS)


def ridge(steps_front):
    return Pipeline(steps_front + [("ridge", final_ridge())])


def build_models(w):
    """name -> (feature-set key, estimator factory, matched stats-only model or None).
    w = column counts: season, stats, emb, stats_ctx, emb_ctx."""
    def plain():
        return ridge([("sc", StandardScaler())])

    def pca10():
        return ridge([("sc", StandardScaler()), ("pca", PCA(n_components=10, random_state=0))])

    def combo(*blocks):
        """blocks: (kind, width, weight) in column order. kind 'plain' = standardise,
        'pca10' = standardise then 10 PCs. weight multiplies the block after scaling."""
        def make():
            parts, weights, c = [], {}, 0
            for i, (kind, width, wt) in enumerate(blocks):
                cols = list(range(c, c + width))
                c += width
                tr = (Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=10, random_state=0))])
                      if kind == "pca10" else StandardScaler())
                parts.append((f"b{i}", tr, cols))
                weights[f"b{i}"] = wt
            return Pipeline([("ct", ColumnTransformer(parts, transformer_weights=weights)),
                             ("ridge", final_ridge())])
        return make

    def eq(width, ref):                      # give a wide block the total variance of a `ref`-column block
        return float(np.sqrt(ref / width))

    S, T, E, TC, EC = w["season"], w["stats"], w["emb"], w["stats_ctx"], w["emb_ctx"]
    return {
        "season":                              ("season",        plain,   None),
        "stats [raw]":                         ("stats",         plain,   None),
        "stats [norm]":                        ("stats_n",       plain,   None),
        "emb PCA10 [raw]":                     ("emb",           pca10,   "stats [raw]"),
        "emb PCA10 [norm]":                    ("emb_n",         pca10,   "stats [norm]"),
        "emb ridge [norm]":                    ("emb_n",         plain,   "stats [norm]"),
        "season + stats [norm]":               ("s_stats_n",     plain,   None),
        "season + emb PCA10 [norm]":           ("s_emb_n",       combo(("plain", S, 1.0), ("pca10", E, 1.0)),
                                                "season + stats [norm]"),
        "season + stats + emb PCA10 [norm]":   ("s_stats_emb_n", combo(("plain", S, 1.0), ("plain", T, 1.0), ("pca10", E, 1.0)),
                                                "season + stats [norm]"),
        "season + emb ridge [norm]":           ("s_emb_n",       combo(("plain", S, 1.0), ("plain", E, eq(E, T))),
                                                "season + stats [norm]"),
        "season + stats + emb ridge [norm]":   ("s_stats_emb_n", combo(("plain", S, 1.0), ("plain", T, 1.0), ("plain", E, eq(E, T))),
                                                "season + stats [norm]"),
        "stats + ctx [norm]":                  ("stats_ctx_n",   plain,   None),
        "season + stats + ctx [norm]":         ("s_stats_ctx_n", plain,   None),
        "emb ridge + ctx [norm]":              ("emb_ctx_n",     plain,   "stats + ctx [norm]"),
        "season + emb ridge + ctx [norm]":     ("s_emb_ctx_n",   combo(("plain", S, 1.0), ("plain", EC, eq(EC, TC))),
                                                "season + stats + ctx [norm]"),
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
    ap.add_argument("--results-root", default=None, help="default: <project>/results or $FFB_RESULTS_ROOT")
    ap.add_argument("--encoder", default="clay", help="which encoder's embeddings to test: clay (default) or presto "
                    "(reads <results root>/embeddings_<encoder>/ and writes to a run name ending _<encoder>)")
    ap.add_argument("--embeddings-dir", default=None, help="default: <results root>/embeddings_<encoder>")
    ap.add_argument("--run-name", default=None, help='output sub-folder under baselines/ (default "<sensor>_<cv>")')
    ap.add_argument("--labels", default=None, help="antecedent_state_labels.csv (default: <results root>/labels/antecedent_state_labels.csv)")
    ap.add_argument("--sensor", default="s1", choices=["s1", "s2"])
    ap.add_argument("--targets", nargs="*", default=None, help="label columns (default: all present)")
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--years", nargs="*", type=int, default=None)
    ap.add_argument("--cv", default="blocked", choices=["blocked", "loyo", "loso"])
    ap.add_argument("--k", type=int, default=5, help="number of blocks for --cv blocked")
    ap.add_argument("--gap-days", type=int, default=14)
    ap.add_argument("--alpha-cv", default="blocked", choices=["blocked", "loo"],
                    help="how the ridge penalty is chosen inside the training data: contiguous time blocks "
                         "(default, honest for autocorrelated imagery) or sklearn's leave-one-out "
                         "(the earlier behaviour; over-fits high-dimensional models)")
    ap.add_argument("--ctx-windows", nargs="+", type=int, default=[14, 30, 60],
                    help="past windows in days for the temporal-context features (default 14 30 60)")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-png", default=None)
    args = ap.parse_args()
    global ALPHA_CV
    ALPHA_CV = args.alpha_cv

    import os
    from project_paths import Layout, ensure_parent, tee_stdout
    layout = Layout(args.results_root)
    args.embeddings_dir = args.embeddings_dir or layout.embeddings_for(args.encoder)
    args.labels = args.labels or os.path.join(layout.labels, "antecedent_state_labels.csv")
    if not os.path.exists(args.labels):
        sys.exit(f"Labels file not found: {args.labels} (build it with build_antecedent_labels.py or pass --labels)")
    run_dir = os.path.join(layout.baselines, args.run_name or
                           f"{args.sensor}_{args.cv}_alpha-{args.alpha_cv}"
                           + f"_{args.encoder}")
    args.out_csv = ensure_parent(args.out_csv or os.path.join(run_dir, "baseline_comparison.csv"))
    args.out_png = ensure_parent(args.out_png or os.path.join(run_dir, "baseline_comparison.png"))
    tee_stdout(os.path.join(run_dir, "report.txt"))
    print(f"Embeddings from {args.embeddings_dir}\nResults to {run_dir}\nRidge penalty chosen by: "
          f"{'contiguous time-block CV' if args.alpha_cv == 'blocked' else 'leave-one-out (sklearn RidgeCV)'}\n")

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
    stats_n = normalise(stats, df)
    E_n = normalise(E, df)
    stats_c, empty = causal_context(stats_n, df, args.ctx_windows)
    E_c, _ = causal_context(E_n, df, args.ctx_windows)
    print("  temporal context windows (days): " + ", ".join(
        f"{W} ({100 * e:.0f}% of dates have no earlier acquisition in it)" for W, e in zip(args.ctx_windows, empty)))
    stats_cn = np.column_stack([stats_n, stats_c])
    E_cn = np.column_stack([E_n, E_c])
    feats = {"season": season, "stats": stats, "stats_n": stats_n, "emb": E, "emb_n": E_n,
             "s_stats_n": np.column_stack([season, stats_n]),
             "s_emb_n": np.column_stack([season, E_n]),
             "s_stats_emb_n": np.column_stack([season, stats_n, E_n]),
             "stats_ctx_n": stats_cn, "emb_ctx_n": E_cn,
             "s_stats_ctx_n": np.column_stack([season, stats_cn]),
             "s_emb_ctx_n": np.column_stack([season, E_cn])}

    models = build_models({"season": season.shape[1], "stats": stats_n.shape[1], "emb": E.shape[1],
                           "stats_ctx": stats_cn.shape[1], "emb_ctx": E_cn.shape[1]})
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
        for name, (fkey, make, _) in models.items():
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

    # relative to season, and fold wins vs the matched statistics-only model
    matched = {name: m for name, (_, _, m) in models.items()}
    for i, r in res.iterrows():
        s = res[(res.target == r.target) & (res.model == "season")]
        res.loc[i, "rmse_vs_season_%"] = 100 * (r.rmse / s.rmse.iloc[0] - 1) if len(s) else np.nan
        ref = matched.get(r.model)
        a, b = fold_rmse.get((r.target, r.model)), fold_rmse.get((r.target, ref)) if ref else None
        if a is not None and b is not None:
            m = np.isfinite(a) & np.isfinite(b)
            res.loc[i, "folds_better_than_stats"] = f"{int((a[m] < b[m]).sum())}/{int(m.sum())}"
    res.to_csv(args.out_csv, index=False)

    # ---- report -----------------------------------------------------------
    order = list(models)
    for target in res.target.unique():
        t = res[res.target == target].set_index("model").reindex([m for m in order if m in set(res[res.target == target].model)])
        print(f"\n=== {target}  (out-of-fold, n={int(t['n'].iloc[0])}) ===")
        print(f"{'model':<38}{'rho':>7}{'rho folds':>13}{'R2':>8}{'RMSE':>9}{'vs season':>11}{'beats stats':>13}")
        for m, r in t.iterrows():
            fb = r.get("folds_better_than_stats", "")
            fb = fb if isinstance(fb, str) else ""
            print(f"{m:<38}{r.rho_pooled:>+7.2f}{r.rho_fold_mean:>+7.2f}±{r.rho_fold_sd:<4.2f}"
                  f"{r.r2:>+8.2f}{r.rmse:>9.3f}{r['rmse_vs_season_%']:>+10.1f}%{fb:>13}")

    print("\n=== Summary 1: does the satellite add anything? ===")
    for target in res.target.unique():
        t = res[res.target == target].set_index("model")
        emb_models = [m for m in t.index if "emb" in m]
        base_models = [m for m in t.index if "emb" not in m]
        if not emb_models or not base_models:
            continue
        best_emb = t.loc[emb_models, "rmse"].idxmin()
        best_base = t.loc[base_models, "rmse"].idxmin()
        gain = 100 * (1 - t.loc[best_emb, "rmse"] / t.loc[best_base, "rmse"])
        verdict = ("embeddings beat the best non-embedding model" if gain > 5 else
                   "no clear gain over the best non-embedding model" if gain > -5 else
                   "embeddings are WORSE than the best non-embedding model")
        print(f"  {target:<22} best embedding '{best_emb}' vs best non-embedding '{best_base}': "
              f"RMSE {gain:+.1f}% -> {verdict}   (best R2 of any model: {t['r2'].max():+.2f})")

    print("\n=== Summary 2: does temporal context help? (best model with ctx vs best without, same family) ===")
    for target in res.target.unique():
        t = res[res.target == target].set_index("model")
        parts = []
        for fam, keep in (("stats", lambda m: "emb" not in m and m.startswith(("stats", "season + stats"))),
                          ("embedding", lambda m: "emb ridge" in m)):
            c = [m for m in t.index if keep(m) and "ctx" in m]
            n = [m for m in t.index if keep(m) and "ctx" not in m and "[raw]" not in m]
            if c and n:
                g = 100 * (1 - t.loc[c, "rmse"].min() / t.loc[n, "rmse"].min())
                parts.append(f"{fam} {g:+.1f}% RMSE (R2 {t.loc[n, 'r2'].max():+.2f} -> {t.loc[c, 'r2'].max():+.2f})")
        if parts:
            print(f"  {target:<22} " + "; ".join(parts))
    print("\n  How to read this: a gain under ~5% RMSE, or one that wins in only about half the folds, is within noise\n"
          "  at this sample size. If the best R2 of ALL models is near zero, the label is not predictable from these\n"
          "  features at all. 'season' being hard to beat means the label is mostly the annual cycle. A NEGATIVE\n"
          "  pooled rho for a weak model is a known artefact of leave-block-out CV; read it as 'no signal'.")

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tg = list(res.target.unique())
        fig, axes = plt.subplots(1, len(tg), figsize=(4.6 * len(tg), max(4.6, 0.34 * len(order) + 1.6)), squeeze=False, sharey=True)
        mods = [m for m in order if m in set(res.model)]
        for ax, target in zip(axes[0], tg):
            t = res[res.target == target].set_index("model").reindex(mods)
            colors = ["#ff7f0e" if "emb" in m else "#1f77b4" if "stats" in m else "#9aa0a6" for m in mods]
            bars = ax.barh(range(len(mods)), t["rho_pooled"], xerr=t["rho_fold_sd"], color=colors, height=0.65,
                           error_kw={"elinewidth": 1, "capsize": 2})
            for bar, m in zip(bars, mods):
                if "ctx" in m:
                    bar.set_hatch("//")
                    bar.set_edgecolor("white")
            ax.set_yticks(range(len(mods)))
            ax.set_yticklabels(mods, fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(target, fontsize=10)
            ax.set_xlabel("out-of-fold Spearman rho (bar = pooled, whisker = fold sd)", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Grey = season only, blue = VV/VH statistics (with/without season), orange = embedding models, hatched = with temporal context", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=140)
        print(f"\nFigure written to {args.out_png}\nTable written to {args.out_csv}")
    except Exception as e:  # plotting is optional
        print(f"\n(figure skipped: {e})\nTable written to {args.out_csv}")


if __name__ == "__main__":
    main()
