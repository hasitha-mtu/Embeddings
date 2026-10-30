r"""
compare_encoders.py -- put the baseline results of several encoders side by side.

Each encoder is tested separately with
    python baseline_comparison.py --encoder clay   --cv loso
    python baseline_comparison.py --encoder presto --cv loso
which write  <results root>/baselines/<sensor>_<cv>_alpha-<blocked|loo>_<encoder>/baseline_comparison.csv.
This script reads those files for the SAME sensor / cv / alpha setting and prints, per label,
the matched comparison: the embedding models of every encoder next to the encoder-free
references (season, season + stats + ctx), which are identical across encoders because they
do not use embeddings.

Usage:
    python compare_encoders.py --cv loso                      # every encoder found
    python compare_encoders.py --cv loyo --encoders clay presto
    python compare_encoders.py --cv loso --metric rmse

Adding a new encoder later needs nothing here: run baseline_comparison.py with --encoder <name>
and it shows up as another column.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

from project_paths import Layout, ensure_parent

EMB_MODELS = ["season + emb ridge + ctx [norm]", "emb ridge + ctx [norm]",
              "season + emb ridge [norm]", "emb ridge [norm]", "season + emb PCA10 [norm]"]
REF_MODELS = ["season", "season + stats [norm]", "season + stats + ctx [norm]"]


def find_runs(baselines_dir, sensor, cv, alpha):
    pat = re.compile(rf"^{re.escape(sensor)}_{re.escape(cv)}_alpha-{re.escape(alpha)}_(.+)$")
    runs = {}
    for d in sorted(glob.glob(os.path.join(baselines_dir, "*"))):
        m = pat.match(os.path.basename(d))
        csv = os.path.join(d, "baseline_comparison.csv")
        if m and os.path.exists(csv):
            runs[m.group(1)] = csv
    return runs


def main():
    ap = argparse.ArgumentParser(description="Side-by-side baseline results for several encoders.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--sensor", default="s1")
    ap.add_argument("--cv", default="loso", choices=["blocked", "loyo", "loso"])
    ap.add_argument("--alpha-cv", default="blocked", choices=["blocked", "loo"])
    ap.add_argument("--encoders", nargs="*", default=None, help="default: every encoder with results")
    ap.add_argument("--metric", default="r2", choices=["r2", "rmse", "rho_pooled"])
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="default: <baselines>/encoder_comparison_<sensor>_<cv>.csv")
    args = ap.parse_args()

    lay = Layout(args.results_root)
    runs = find_runs(lay.baselines, args.sensor, args.cv, args.alpha_cv)
    if args.encoders:
        runs = {e: p for e, p in runs.items() if e in args.encoders}
    if len(runs) < 1:
        sys.exit(f"No baseline results for {args.sensor}/{args.cv}/alpha-{args.alpha_cv} in {lay.baselines}. "
                 f"Run baseline_comparison.py --encoder <name> first.")
    print(f"Encoders compared: {', '.join(runs)}   (sensor {args.sensor}, cv {args.cv}, alpha {args.alpha_cv}, metric {args.metric})\n")

    tabs = {e: pd.read_csv(p) for e, p in runs.items()}
    targets = args.targets or list(next(iter(tabs.values()))["target"].unique())
    # reference rows (no embeddings) come from the first encoder's file; n is checked below
    first = next(iter(tabs.values()))
    rows = []
    for tgt in targets:
        print(f"== {tgt} ==")
        header = f"  {'model':<36}" + "".join(f"{e:>12}" for e in tabs)
        print(header)
        for m in REF_MODELS:
            v = first[(first.target == tgt) & (first.model == m)]
            if len(v):
                x = v.iloc[0][args.metric]
                print(f"  {m + '  (no embeddings)':<36}" + f"{x:>12.3f}" + "".join(f"{'':>12}" for _ in list(tabs)[1:]))
                rows.append({"target": tgt, "model": m, "encoder": "(none)", args.metric: x, "n": v.iloc[0]["n"]})
        for m in EMB_MODELS:
            vals, line = {}, f"  {m:<36}"
            for e, t in tabs.items():
                v = t[(t.target == tgt) & (t.model == m)]
                if len(v):
                    vals[e] = v.iloc[0]
                    line += f"{v.iloc[0][args.metric]:>12.3f}"
                    rows.append({"target": tgt, "model": m, "encoder": e, args.metric: v.iloc[0][args.metric], "n": v.iloc[0]["n"]})
                else:
                    line += f"{'-':>12}"
            print(line)
        ns = {e: int(t[t.target == tgt]["n"].max()) for e, t in tabs.items() if (t.target == tgt).any()}
        if len(set(ns.values())) > 1:
            print(f"  note: the encoders were scored on different numbers of dates {ns}; "
                  f"Presto drops dates with too little history, so compare with care")
        print()
    out = ensure_parent(args.out or os.path.join(lay.baselines, f"encoder_comparison_{args.sensor}_{args.cv}.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Table written to {out}")
    print("Reading it: a gain only counts if it beats 'season + stats + ctx' by more than ~5% RMSE and "
          "holds across folds (see each run's report.txt).")


if __name__ == "__main__":
    main()
