r"""
select_catchments.py  (pick a small, justified, diverse set of catchments)

Picks N catchments from catchment_level_shortlist.csv for the multi-catchment
embedding-vs-baseline test, so the choice is explainable in a paper and not
just "the first six".

TWO STEPS
  1. ELIGIBILITY (hard filters). A catchment must have
       - gauge record covering the whole target period (default 2022-2023),
       - sentinel-era completeness >= --min-completeness (%),
       - longest data gap <= --max-gap-days,
       - a catchment polygon that matches the gauge's true catchment
         (subcatchment_to_true_area_ratio within --ratio-range),
       - optionally, enough daily labels in every target year
         (--labels antecedent_state_labels.csv, --min-label-days).
  2. DIVERSITY (farthest-point selection). Starting from the catchments you
     already have (default 11001), repeatedly add the eligible catchment that
     is MOST DIFFERENT from everything chosen so far, measured on
       log10(area), latitude, longitude, (optionally) label dynamic range,
     plus a bonus distance for a different river basin district (rbd).
     Ties are broken by data quality. This spreads the sample over size,
     geography and flow regime, which is what a leave-one-catchment-out test
     needs to say anything about unseen catchments.

WHAT THIS CANNOT KNOW
  Whether Sentinel-1 actually covers a catchment on every pass (some tracks
  miss some catchments, as 11001 showed: 31 of 121 dates empty). That only
  appears after extraction. So --reserve extra catchments are listed as
  stand-ins in case one turns out to have poor coverage.

Usage:
    python select_catchments.py --catchment-list dataset/metadata/catchment_level_shortlist.csv \
        --years 2022 2023 --n 6 --reserve 2 \
        --labels output/antecedent_state_labels.csv

Output defaults to <results root>/selection/selected_catchments.csv (see
project_paths.py); --out overrides.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description="Select a diverse, eligible set of catchments.")
    ap.add_argument("--catchment-list", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023])
    ap.add_argument("--n", type=int, default=10, help="total catchments to select (including --include)")
    ap.add_argument("--reserve", type=int, default=2, help="extra stand-in catchments to list")
    ap.add_argument("--include", nargs="*", default=["11001", "19103", "20015"],
                    help="station ids already in hand; always selected first (if eligible)")
    ap.add_argument("--labels", default=None, help="antecedent_state_labels.csv (optional)")
    ap.add_argument("--min-label-days", type=int, default=300, help="labelled days required in EVERY target year")
    ap.add_argument("--min-completeness", type=float, default=98.0)
    ap.add_argument("--max-gap-days", type=float, default=15.0,
                    help="longest gap anywhere in the 2017-2026 gauge record (crude; --labels checks the target years directly)")
    ap.add_argument("--ratio-range", nargs=2, type=float, default=[0.7, 1.3])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--out", default=None, help="default: <results root>/selection/selected_catchments.csv")
    return ap.parse_args()


def label_stats(path, stations, years):
    lab = pd.read_csv(path, parse_dates=["date"])
    lab["station_id"] = lab["station_id"].astype(str)
    lab = lab[lab["station_id"].isin(stations)]
    lab["year"] = lab["date"].dt.year
    out = {}
    for st, g in lab.groupby("station_id"):
        per_year = {y: int(g.loc[g.year == y, "level_percentile"].notna().sum()) for y in years}
        gy = g[g.year.isin(years)]
        lp = gy["level_percentile"].dropna()
        out[st] = {"label_days_min_year": min(per_year.values()),
                   "label_range": float(lp.quantile(0.95) - lp.quantile(0.05)) if len(lp) > 20 else np.nan}
    return pd.DataFrame.from_dict(out, orient="index")


def main():
    args = parse_args()
    from project_paths import Layout, ensure_parent
    args.out = args.out or os.path.join(Layout(args.results_root).selection, "selected_catchments.csv")
    ensure_parent(args.out)
    df = pd.read_csv(args.catchment_list)
    df["station_id"] = df["station_id"].astype(str)
    n0 = len(df)

    y0, y1 = min(args.years), max(args.years)
    start = pd.to_datetime(df["sentinel_era_start"], utc=True, errors="coerce")
    end = pd.to_datetime(df["sentinel_era_end"], utc=True, errors="coerce")
    ratio = df["subcatchment_to_true_area_ratio"].replace([np.inf, -np.inf], np.nan)

    reasons = pd.DataFrame(index=df.index)
    reasons["record does not cover target years"] = ~((start <= pd.Timestamp(f"{y0}-01-01", tz="UTC")) &
                                                      (end >= pd.Timestamp(f"{y1}-12-31", tz="UTC")))
    reasons[f"completeness < {args.min_completeness}%"] = ~(df["sentinel_era_completeness_pct"] >= args.min_completeness)
    reasons[f"max gap > {args.max_gap_days} d"] = ~(df["sentinel_era_max_gap_days"] <= args.max_gap_days)
    reasons[f"polygon/true-area ratio outside {args.ratio_range}"] = ~((ratio >= args.ratio_range[0]) & (ratio <= args.ratio_range[1]))

    if args.labels:
        ls = label_stats(args.labels, set(df["station_id"]), args.years)
        df = df.merge(ls, left_on="station_id", right_index=True, how="left")
        reasons[f"< {args.min_label_days} labelled days in some year"] = ~(df["label_days_min_year"] >= args.min_label_days)

    print(f"{n0} catchments in the shortlist; years {args.years}")
    print("Why catchments were excluded (a catchment can fail several):")
    for c in reasons.columns:
        print(f"  {int(reasons[c].sum()):>3}  {c}")
    forced = df["station_id"].isin(args.include)
    for i in df.index[forced & reasons.any(axis=1)]:
        failed = [c for c in reasons.columns if reasons.loc[i, c]]
        print(f"Note: station {df.loc[i, 'station_id']} (--include) fails: {'; '.join(failed)}. "
              f"Kept anyway because it is already extracted.")
    elig = df[(~reasons.any(axis=1)) | forced].copy()
    print(f"-> {int((~reasons.any(axis=1)).sum())} eligible (+ {int((forced & reasons.any(axis=1)).sum())} forced in)\n")
    if len(elig) == 0:
        sys.exit("Nothing eligible: loosen the thresholds.")

    # ---- features for diversity ------------------------------------------
    feats = pd.DataFrame({
        "log_area": np.log10(elig["subcatchment_area_km2"]),
        "lat": elig["latitude"],
        "lon": elig["longitude"],
    }, index=elig.index)
    if args.labels and "label_range" in elig.columns and elig["label_range"].notna().sum() > 3:
        feats["label_range"] = elig["label_range"].fillna(elig["label_range"].median())
    Z = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    Z = Z.fillna(0.0)
    rbd = elig["rbd"].to_numpy()
    sids = elig["station_id"].to_numpy()

    def dist(i, j):
        return float(np.sqrt(((Z.iloc[i].to_numpy() - Z.iloc[j].to_numpy()) ** 2).sum())) + (1.0 if rbd[i] != rbd[j] else 0.0)

    # data quality for tie-breaks: high completeness, small gap
    q = (elig["sentinel_era_completeness_pct"].rank(pct=True) + (-elig["sentinel_era_max_gap_days"]).rank(pct=True)).to_numpy()

    # ---- farthest-point selection ----------------------------------------
    chosen = [list(sids).index(s) for s in args.include if s in set(sids)]
    skipped = [s for s in args.include if s not in set(sids)]
    if skipped:
        print(f"Note: --include station(s) {skipped} are not in the catchment list and were ignored.\n")
    if not chosen:
        chosen = [int(np.argmax(q))]
    want = args.n + args.reserve
    min_d = {}
    while len(chosen) < min(want, len(elig)):
        best, best_key = None, None
        for i in range(len(elig)):
            if i in chosen:
                continue
            d = min(dist(i, j) for j in chosen)
            key = (round(d, 6), q[i])
            if best_key is None or key > best_key:
                best, best_key = i, key
        min_d[best] = best_key[0]
        chosen.append(best)

    sel_idx, res_idx = chosen[:args.n], chosen[args.n:]

    cols = ["station_id", "station_name", "rbd", "subcatchment_id", "subcatchment_area_km2", "latitude", "longitude",
            "sentinel_era_completeness_pct", "sentinel_era_max_gap_days"]
    if "label_days_min_year" in elig.columns:
        cols += ["label_days_min_year", "label_range"]

    def show(idx, title):
        t = elig.iloc[idx][cols].copy()
        t.insert(0, "role", ["in hand" if sids[i] in args.include else "new" for i in idx] if title == "SELECTED" else "reserve")
        t["dist_to_chosen"] = [min_d.get(i, np.nan) for i in idx]
        print(f"=== {title} ===")
        print(t.rename(columns={"subcatchment_area_km2": "area_km2", "sentinel_era_completeness_pct": "complete_%",
                                "sentinel_era_max_gap_days": "max_gap_d"}).round(2).to_string(index=False))
        print()

    show(sel_idx, "SELECTED")
    if res_idx:
        show(res_idx, "RESERVE (use if a selected catchment has poor Sentinel-1 coverage)")

    out = elig.iloc[sel_idx + res_idx].copy()
    out["selection"] = ["selected"] * len(sel_idx) + ["reserve"] * len(res_idx)
    out.to_csv(args.out, index=False)
    new_ids = [sids[i] for i in sel_idx if sids[i] not in args.include]
    print(f"Wrote {args.out} (same columns as the shortlist, plus 'selection'; usable as --catchment-list).")
    print(f"Spread of the selected set: area {elig.iloc[sel_idx]['subcatchment_area_km2'].min():.0f}-"
          f"{elig.iloc[sel_idx]['subcatchment_area_km2'].max():.0f} km2, "
          f"{elig.iloc[sel_idx]['rbd'].nunique()} river basin district(s).")
    print("\nNext: python run_batch.py --catchment-list", args.out, "--years", " ".join(map(str, args.years)),
          "--shapefile <subcatchments.shp> --dry-run")


if __name__ == "__main__":
    main()
