r"""
verify_embeddings.py -- sanity-check the output of extract_embeddings.py /
run_catchment_pipeline.py for ONE catchment+sensor before building on it.

It checks, in order:
  1. Structure      : npz dates match csv dates, all vectors same shape, no inf.
  2. NaN consistency: an embedding is all-NaN exactly when the csv says 0
                      patches / 0 valid pixels; no half-NaN vectors.
  3. Coverage       : how many dates are usable, whether the empty dates
                      follow a repeating pattern (e.g. one orbit track that
                      misses the catchment), longest gap between usable dates.
  4. Hand-crafted   : the csv's simple stats (S1 VV/VH, or S2 NDVI etc.) are
     stats           in physically sensible ranges.
  5. Embedding      : the vectors actually vary between dates (not collapsed
     health          to one constant), no dead dimensions, no wild outliers.
  6. Content check  : do the main axes of variation (PCA) of the embeddings
                      move with the simple stats? If a foundation model is
                      encoding the imagery, they usually do. A weak result is
                      a prompt to look closer, not proof of a bug.
  7. (optional) labels   : same correlation against your water-level labels.
  8. (optional) rerun    : compare against a second extraction of the same
                           dates to test determinism (Clay masking/shuffle).

It also writes a diagnostic PNG. The thresholds are rules of thumb, not
statistics -- the numbers it prints matter more than the PASS/WARN labels.

Usage:
    python verify_embeddings.py \
        --embeddings embeddings_permanent/embeddings_11001_s1_2023.npz \
        --indices    embeddings_permanent/indices_11001_s1_2023.csv

    # plus the labels check (labels file from build_antecedent_labels.py):
    ... --labels antecedent_state_labels.csv --station-id 11001

    # Output: the figure and a text copy of this report are written to
    # <results root>/verification/verify_<station>_<sensor>_<year>.png/.txt
    # (named after the embeddings file). --out-png overrides; --results-root
    # or $FFB_RESULTS_ROOT moves the root. See project_paths.py.

    # plus a determinism check against a second run on the same dates:
    ... --compare other_run.npz

Requires: numpy, pandas, matplotlib
"""

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

LABEL_COLS = ["level_percentile", "standardized_anomaly", "baseflow_index", "recession_rate_7d"]
STAT_COLS = ["vv_mean", "vh_mean", "vv_db_mean", "vh_db_mean", "vv_db_median", "vh_db_median", "ndvi_mean", "ndwi_mean", "mndwi_mean"]

_status = []


def report(status, name, detail=""):
    _status.append(status)
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def day_gaps(date_index):
    """Whole days between consecutive dates in a DatetimeIndex."""
    return np.diff(date_index.values).astype("timedelta64[D]").astype(int)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    ra, rb = pd.Series(a[ok]).rank().to_numpy(), pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def main():
    ap = argparse.ArgumentParser(description="Sanity-check extracted embeddings for one catchment+sensor.")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--labels", default=None, help="antecedent_state_labels.csv (optional)")
    ap.add_argument("--station-id", default=None, help="station to select from --labels")
    ap.add_argument("--compare", default=None, help="second .npz of the same dates, to test determinism")
    ap.add_argument("--out-png", default=None, help="default: <results root>/verification/verify_<run>.png")
    ap.add_argument("--results-root", default=None, help="default: <project>/results or $FFB_RESULTS_ROOT")
    args = ap.parse_args()

    import os
    import re
    from project_paths import Layout, ensure_parent, tee_stdout
    m_name = re.match(r"embeddings_(.+)\.npz$", os.path.basename(args.embeddings))
    tag = f"verify_{m_name.group(1)}" if m_name else "verify_embeddings"
    if args.out_png is None:
        args.out_png = os.path.join(Layout(args.results_root).verification, tag + ".png")
    ensure_parent(args.out_png)
    tee_stdout(os.path.splitext(args.out_png)[0] + ".txt")

    emb = np.load(args.embeddings)
    idx = pd.read_csv(args.indices, dtype={"date": str})
    npz_dates, csv_dates = sorted(emb.files), sorted(idx["date"])

    # ---- 1. structure ---------------------------------------------------
    print("\n=== 1. Structure ===")
    only_npz, only_csv = set(npz_dates) - set(csv_dates), set(csv_dates) - set(npz_dates)
    if only_npz or only_csv or idx["date"].duplicated().any():
        report("FAIL", "dates match between npz and csv",
               f"only in npz: {sorted(only_npz)[:5]}, only in csv: {sorted(only_csv)[:5]}, "
               f"duplicate csv dates: {int(idx['date'].duplicated().sum())}")
        sys.exit(1)
    report("PASS", "dates match between npz and csv", f"{len(npz_dates)} dates")

    shapes = Counter(emb[d].shape for d in npz_dates)
    if len(shapes) != 1:
        report("FAIL", "all embedding vectors have the same shape", f"found {dict(shapes)}")
        sys.exit(1)
    dim = next(iter(shapes))[0]
    E = np.stack([emb[d] for d in npz_dates]).astype("float64")
    report("PASS", "all embedding vectors have the same shape", f"({dim},), dtype {emb[npz_dates[0]].dtype}")
    if np.isinf(E).any():
        report("FAIL", "no infinite values", f"{int(np.isinf(E).sum())} inf entries")
    else:
        report("PASS", "no infinite values")

    idx = idx.set_index("date").reindex(npz_dates)
    dates = pd.to_datetime(npz_dates)

    # ---- 2. NaN consistency --------------------------------------------
    print("\n=== 2. NaN consistency ===")
    nan_all, nan_any = np.isnan(E).all(axis=1), np.isnan(E).any(axis=1)
    partial = nan_any & ~nan_all
    if partial.any():
        report("FAIL", "no partially-NaN vectors", f"{int(partial.sum())} dates, e.g. {[str(x) for x in np.array(npz_dates)[partial][:5]]}")
    else:
        report("PASS", "no partially-NaN vectors")
    csv_empty = (idx["n_patches_used"].to_numpy() == 0) | (idx["valid_pixel_frac"].to_numpy() == 0)
    mismatch = nan_all != csv_empty
    if mismatch.any():
        report("FAIL", "all-NaN embedding <=> 0 patches in csv",
               f"{int(mismatch.sum())} dates disagree, e.g. {[str(x) for x in np.array(npz_dates)[mismatch][:5]]}")
    else:
        report("PASS", "all-NaN embedding <=> 0 patches in csv", f"{int(nan_all.sum())} empty dates agree")

    # ---- 3. coverage ----------------------------------------------------
    print("\n=== 3. Coverage ===")
    ok = ~nan_all
    n_ok = int(ok.sum())
    report("PASS" if n_ok >= 20 else "WARN", "usable dates", f"{n_ok} of {len(ok)} ({n_ok / len(ok):.0%})")
    if nan_all.sum() >= 3:
        gaps = Counter(day_gaps(dates[nan_all]).tolist())
        common, count = gaps.most_common(1)[0]
        share = count / sum(gaps.values())
        report("PASS", "pattern of empty dates",
               f"most common spacing between empty dates is {common} days ({share:.0%} of gaps) -- "
               f"{'a regular spacing like this suggests one orbit track that misses the catchment, not random failures' if share > 0.6 else 'irregular'}")
    if n_ok >= 2:
        g = day_gaps(dates[ok])
        report("PASS" if g.max() <= 30 else "WARN", "longest gap between usable dates", f"{int(g.max())} days (median {int(np.median(g))})")

    # ---- 4. hand-crafted stats -----------------------------------------
    print("\n=== 4. Hand-crafted stats (from the csv) ===")
    feats = {}
    if "vv_mean" in idx.columns:
        if "vv_db_mean" in idx.columns:   # per-pixel dB mean (current extractor)
            vv = idx["vv_db_mean"].to_numpy(float).copy()
            vh = idx["vh_db_mean"].to_numpy(float).copy()
        else:                             # older csv: dB of the linear mean
            vv = 10 * np.log10(np.clip(idx["vv_mean"].to_numpy(float), 1e-6, None))
            vh = 10 * np.log10(np.clip(idx["vh_mean"].to_numpy(float), 1e-6, None))
        vv[~ok], vh[~ok] = np.nan, np.nan
        feats = {"VV mean (dB)": vv, "VH mean (dB)": vh, "VV-VH (dB)": vv - vh}
        m_vv, m_vh = np.nanmean(vv), np.nanmean(vh)
        report("PASS" if -20 <= m_vv <= -3 and -30 <= m_vh <= -8 and m_vh < m_vv else "WARN",
               "S1 backscatter in a plausible range",
               f"VV mean {m_vv:.1f} dB, VH mean {m_vh:.1f} dB (typical land: VV about -5 to -15, VH about -12 to -25, VH below VV)")
    else:
        for c in [c for c in STAT_COLS if c in idx.columns]:
            v = idx[c].to_numpy(float)
            v[~ok] = np.nan
            feats[c] = v
        report("PASS", "index columns present", ", ".join(feats))

    # ---- 5. embedding health -------------------------------------------
    print("\n=== 5. Embedding health ===")
    V = E[ok]
    if len(V) < 5:
        report("FAIL", "enough usable dates to assess", f"only {len(V)}")
        sys.exit(1)
    norms = np.linalg.norm(V, axis=1)
    dim_std = V.std(axis=0)
    dead = float((dim_std < 1e-6).mean())
    Vc = V - V.mean(axis=0)
    total_spread = float(np.sqrt((Vc ** 2).sum(axis=1).mean()))
    rel_spread = total_spread / float(norms.mean())
    U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    scores = U * S
    if rel_spread < 1e-6 or dead > 0.99:
        report("FAIL", "embeddings vary between dates", f"relative spread {rel_spread:.2e} -- vectors are effectively identical")
    else:
        report("PASS", "embeddings vary between dates",
               f"spread between dates is {rel_spread:.1%} of the vector length (norm {norms.mean():.2f})")
    report("PASS" if dead < 0.1 else "WARN", "dead dimensions", f"{dead:.1%} of dims have ~zero variance")
    med = np.median(norms)
    mad = np.median(np.abs(norms - med)) or 1e-12
    out = np.abs(norms - med) / (1.4826 * mad) > 6
    report("PASS" if not out.any() else "WARN", "norm outliers",
           f"{int(out.sum())} dates, e.g. {[str(x) for x in np.array(npz_dates)[ok][out][:5]]}" if out.any() else "none")
    n90 = int(np.searchsorted(np.cumsum(ev), 0.9) + 1)
    print(f"      variance explained: PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}, PC3 {ev[2]:.1%}; {n90} components for 90%")

    # ---- 6. content check ----------------------------------------------
    print("\n=== 6. Do the embeddings move with the image statistics? (Spearman rho) ===")
    pcs = {f"PC{i + 1}": np.full(len(ok), np.nan) for i in range(3)}
    for i in range(3):
        pcs[f"PC{i + 1}"][ok] = scores[:, i]

    def table(rows, cols, title):
        """Prints Spearman rho for every (row, col) pair; returns the largest |rho|."""
        print(f"      {title}")
        print(f"      {'':<16}" + "".join(f"{c[:20]:>22}" for c in cols))
        best = 0.0
        for rn, rv in rows.items():
            cells = []
            for cv in cols.values():
                r, n = spearman(rv, cv)
                cells.append(f"{r:+.2f} (n={n})" if np.isfinite(r) else "n/a")
                if np.isfinite(r):
                    best = max(best, abs(r))
            print(f"      {rn:<16}" + "".join(f"{c:>22}" for c in cells))
        return best

    if feats:
        best = table(pcs, feats, "embedding PCs vs hand-crafted stats")
        if best >= 0.5:
            report("PASS", "embeddings track the image statistics", f"strongest |rho| = {best:.2f}")
        elif best >= 0.3:
            report("WARN", "embeddings only weakly track the image statistics", f"strongest |rho| = {best:.2f}")
        else:
            report("WARN", "no clear link between embeddings and image statistics",
                   f"strongest |rho| = {best:.2f}; not necessarily a bug (the model may encode texture/geometry rather "
                   f"than mean brightness), but worth checking the Clay forward pass")

    # ---- 7. labels (optional) ------------------------------------------
    lab_cols = {}
    if args.labels:
        print("\n=== 7. Against water-level labels (same-day) ===")
        lab = pd.read_csv(args.labels, parse_dates=["date"])
        if args.station_id is not None and "station_id" in lab.columns:
            lab = lab[lab["station_id"].astype(str) == str(args.station_id)]
        lab = lab.set_index("date").reindex(dates)
        cols = {c: lab[c].to_numpy(float) for c in LABEL_COLS if c in lab.columns}
        if not cols or all(np.isnan(v).all() for v in cols.values()):
            report("WARN", "labels overlap with these dates", "no matching label rows (check --station-id and the date range)")
        else:
            lab_cols = cols
            rows = {**pcs, **feats}
            best = table(rows, cols, "embedding PCs / image stats vs labels")
            report("PASS" if best >= 0.3 else "WARN", "any relationship with labels",
                   f"strongest |rho| = {best:.2f}. Same-day only, a few dozen points, and one year: "
                   f"read this as a smoke test, not as evidence")

    # ---- 7b. orbit-track check -----------------------------------------
    # Sentinel-1 repeats every 12 days per relative orbit. Dates that share
    # (days since reference) mod 12 are the same track, so this groups dates by
    # track without needing orbit metadata. Different tracks see the catchment
    # at different incidence angles, which shifts backscatter by a dB or two
    # regardless of the ground state.
    print("\n=== 7b. Orbit-track check (12-day phase groups) ===")
    track_ok = False
    if nan_all.sum() >= 3:
        ref = dates[nan_all][0]
        phase = np.asarray(((dates - ref).days) % 12)
        used = sorted(set(phase[ok]))
        counts = {int(k): int((phase[ok] == k).sum()) for k in used}
        # tolerate +-1 day acquisition-date jitter only if the groups look messy
        report("PASS" if len(used) <= 6 else "WARN", "12-day phase groups among usable dates",
               f"{len(used)} groups, dates per group {counts} (phase 0 = the missing track)")
        track_ok = len(used) >= 2

    if track_ok:
        def eta2(v):
            v = np.asarray(v, float)
            m = np.isfinite(v)
            if m.sum() < 10:
                return np.nan
            tot = ((v[m] - v[m].mean()) ** 2).sum()
            between = sum(((phase[m] == k).sum()) * (v[m][phase[m] == k].mean() - v[m].mean()) ** 2
                          for k in set(phase[m]))
            return float(between / tot) if tot > 0 else np.nan

        def demean(v):
            v = np.asarray(v, float).copy()
            out = np.full_like(v, np.nan)
            for k in set(phase):
                g = (phase == k) & np.isfinite(v)
                if g.sum():
                    out[g] = v[g] - v[g].mean()
            return out

        print("      share of variance explained by track alone (eta^2; 0 = none, 1 = all)")
        shares = {}
        for name, v in {**pcs, **feats}.items():
            shares[name] = eta2(v)
            print(f"      {name:<16}{shares[name]:>8.2f}")
        pc1_share = shares.get("PC1", np.nan)
        if np.isfinite(pc1_share):
            report("WARN" if pc1_share >= 0.3 else "PASS", "embedding PC1 vs orbit track",
                   f"track explains {pc1_share:.0%} of PC1 variance"
                   + (" -- compare features WITHIN a track, see below" if pc1_share >= 0.3 else ""))

        adj_feats = {f"{k} [track-adj]": demean(v) for k, v in feats.items()}
        adj_pcs = {f"{k} [track-adj]": demean(v) for k, v in pcs.items()}
        if lab_cols:
            table({**adj_pcs, **adj_feats}, lab_cols,
                  "track-adjusted (per-track mean removed) PCs / image stats vs labels")
            print("      Compare with section 7. If VV/VH correlations drop a lot, part of that signal was the track "
                  "offset; if the embedding PCs now match or beat VV/VH, they carry information beyond brightness.")
        else:
            table(adj_pcs, adj_feats, "track-adjusted PCs vs track-adjusted image stats")
    else:
        print("      skipped (needs >= 3 empty dates to anchor the 12-day phase)")

    # ---- 8. determinism (optional) -------------------------------------
    if args.compare:
        print("\n=== 8. Rerun comparison ===")
        other = np.load(args.compare)
        shared = [d for d in npz_dates if d in other.files and np.isfinite(emb[d]).all() and np.isfinite(other[d]).all()]
        if not shared:
            report("WARN", "shared dates to compare", "none with finite embeddings in both files")
        else:
            diffs = [float(np.max(np.abs(emb[d].astype('float64') - other[d].astype('float64')))) for d in shared]
            cos = [float(np.dot(emb[d], other[d]) / (np.linalg.norm(emb[d]) * np.linalg.norm(other[d]))) for d in shared]
            scale = float(np.mean([np.abs(emb[d]).max() for d in shared]))
            if max(diffs) <= 1e-4 * scale:
                report("PASS", "extraction is deterministic", f"{len(shared)} dates, max abs diff {max(diffs):.2e}")
            else:
                report("WARN", "extraction is NOT deterministic",
                       f"{len(shared)} dates, max abs diff {max(diffs):.3g} (min cosine {min(cos):.3f}). "
                       f"For Clay this usually means patch masking/shuffling is still active")

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :], sharex=None)
    for name, v in list(feats.items())[:2]:
        ax1.plot(dates, v, "o-", ms=3, lw=0.8, label=name)
    ax1.plot(dates[nan_all], np.full(int(nan_all.sum()), ax1.get_ylim()[0]), "x", color="crimson", ms=5, label="empty date")
    ax1.set_title("Hand-crafted image statistics per date"); ax1.legend(ncol=3, fontsize=8); ax1.grid(alpha=.3)
    for k in ("PC1", "PC2"):
        ax2.plot(dates, pcs[k], "o-", ms=3, lw=0.8, label=f"{k} ({ev[int(k[-1]) - 1]:.0%} of variance)")
    ax2.set_title("Main axes of variation in the embeddings"); ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    ax3, ax4 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
    C = np.corrcoef(Vc)
    im = ax3.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax3.set_title("Date-to-date similarity (centred)\nblocks/stripes can reveal orbit or season groups")
    fig.colorbar(im, ax=ax3, shrink=0.8)
    if feats:
        name, v = next(iter(feats.items()))
        ax4.scatter(v[ok], scores[:, 0], c=dates[ok].month, cmap="twilight", s=14)
        ax4.set_xlabel(name); ax4.set_ylabel("PC1"); ax4.set_title("PC1 vs image statistic (colour = month)"); ax4.grid(alpha=.3)
    fig.savefig(args.out_png, dpi=130)
    print(f"\nFigure written to {args.out_png}")

    n_fail, n_warn = _status.count("FAIL"), _status.count("WARN")
    print(f"\nSummary: {_status.count('PASS')} pass, {n_warn} warn, {n_fail} fail")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
