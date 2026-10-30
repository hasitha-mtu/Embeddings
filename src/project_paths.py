r"""
project_paths.py -- one place that decides where every output goes.

Everything the pipeline produces lands under a single RESULTS ROOT instead of
the project directory:

    <results root>/
        embeddings_clay/  PERMANENT. Clay embeddings_<station>_<sensor>_<year>.npz and
                          indices_<station>_<sensor>_<year>.csv (what the baseline
                          comparison and the models read)
        embeddings_<encoder>/  PERMANENT. One folder per encoder, same file names (embeddings_presto/ is
                          written by extract_presto.py; any future encoder gets its own)
        cache/            PERMANENT. pixels_<station>_<sensor>_<year>.npz (sampled pixel
                          values over time) and stats_*.csv. Lets any new encoder run
                          without re-downloading imagery (see pixel_cache.py)
        raw_samples/      PERMANENT. A few raw GeoTIFFs per station/sensor for visual QA
        verification/     verify_<station>_<sensor>_<year>.png + .txt (report)
        baselines/<run>/  baseline_comparison.csv/.png + report.txt per run
        selection/        selected_catchments.csv (from select_catchments.py)
        logs/             batch_log.csv (from run_batch.py)
        manifests/        CDSE job manifests from standalone acquire/download runs
        labels/           antecedent_state_labels.csv (from build_antecedent_labels.py)
        checks/           throw-away test extractions (determinism reruns, samples)
        scratch/downloads/  Raw imagery. KEPT after processing by default; removed only
                          by cleanup_raw.py (or --delete-raw). Can be pointed at a
                          bigger/faster drive with --tmp-download-dir.

Where is the results root?  In order of priority:
    1. --results-root on any script
    2. the FFB_RESULTS_ROOT environment variable
    3. <project root>/results   (project root = the folder above src/, if the
       scripts live in src/; otherwise the current directory)

Every explicit path flag (--embeddings-out-dir, --out-png, --out, ...) still
overrides the default for that one thing.

This file is also a small tool:
    python src/project_paths.py                 show the layout
    python src/project_paths.py --init          create the folders
    python src/project_paths.py --migrate-from embeddings_permanent .   dry-run: list
                                                files that would be moved into the layout
    python src/project_paths.py --migrate-from embeddings_permanent . --apply
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def _project_root():
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "src" else Path.cwd()


def resolve_root(arg=None):
    return Path(arg or os.environ.get("FFB_RESULTS_ROOT") or (_project_root() / "results"))


class Layout:
    """Output locations under one results root. Properties return plain strings
    (the scripts use os.path). Nothing is created until ensure()/mkdir is called."""

    def __init__(self, root=None):
        self.root = resolve_root(root)

    def _p(self, *parts):
        return str(self.root.joinpath(*parts))

    @property
    def embeddings(self): return self._p("embeddings")
    @property
    def cache(self): return self._p("cache")
    @property
    def raw_samples(self): return self._p("raw_samples")
    @property
    def verification(self): return self._p("verification")
    @property
    def baselines(self): return self._p("baselines")
    @property
    def selection(self): return self._p("selection")
    @property
    def logs(self): return self._p("logs")
    @property
    def checks(self): return self._p("checks")
    @property
    def manifests(self): return self._p("manifests")
    @property
    def labels(self): return self._p("labels")
    @property
    def scratch_downloads(self): return self._p("scratch", "downloads")

    def embeddings_for(self, encoder):
        """Every encoder has its own folder, embeddings_<encoder>/ (embeddings_clay/,
        embeddings_presto/, ...), so results never mix and encoders can be compared.
        Legacy: if embeddings_clay/ does not exist yet but the old shared embeddings/
        folder holds Clay files, that old folder is used (with a one-line note) until
        you run  python project_paths.py --migrate-embeddings --apply"""
        enc = "clay" if encoder in (None, "") else str(encoder)
        new = self._p(f"embeddings_{enc}")
        if enc == "clay" and not os.path.isdir(new) and self.has_legacy_embeddings():
            print(f"  note: using the old shared folder {self.embeddings} for Clay "
                  f"(move it with: python project_paths.py --migrate-embeddings --apply)")
            return self.embeddings
        return new

    def has_legacy_embeddings(self):
        d = self.embeddings
        return os.path.isdir(d) and any(f.startswith("embeddings_") and f.endswith(".npz") for f in os.listdir(d))

    def all_dirs(self):
        return [self._p("embeddings_clay"), self._p("embeddings_presto"), self.cache, self.raw_samples, self.verification, self.baselines,
                self.selection, self.logs, self.checks, self.manifests, self.labels,
                self.scratch_downloads]

    def ensure(self, *dirs):
        for d in dirs:
            os.makedirs(d, exist_ok=True)


def ensure_parent(path):
    """Create the folder a file will be written into; returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_stdout(path):
    """From now on everything printed also goes to `path` (line-buffered, so it is
    complete even if the script exits early). Returns the open file."""
    ensure_parent(path)
    f = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, f)
    return f


# --------------------------------------------------------------------------
# Migration of files that already landed in the wrong place
# --------------------------------------------------------------------------

def _plan_migration(sources, layout):
    """List of (src, dst) moves for recognisable output files found directly in
    each source directory (and in its raw_samples/ subfolder). Never overwrites."""
    plan, notes = [], []

    def add(src, dst_dir):
        dst = Path(dst_dir) / src.name
        if dst.exists():
            notes.append(f"skip (already exists at destination): {src}")
        elif src.resolve() != dst.resolve():
            plan.append((src, dst))

    for sdir in sources:
        sdir = Path(sdir)
        if not sdir.is_dir():
            notes.append(f"not a folder, ignored: {sdir}")
            continue
        for f in sorted(sdir.iterdir()):
            if not f.is_file():
                continue
            n = f.name
            if ".tmp." in n:
                notes.append(f"leftover temp file, not moved: {f}")
            elif re.match(r"^(check|rerun)\b", n):
                add(f, layout.checks)
            elif re.match(r"^embeddings_.+\.npz$", n) or re.match(r"^indices_.+\.csv$", n):
                add(f, Path(layout._p("embeddings_clay")))
            elif re.match(r"^(pixels|stats)_.+_(s1|s2)_\d{4}\.(npz|csv)$", n):
                add(f, layout.cache)
            elif n.startswith("verify_embeddings") and n.endswith((".png", ".txt")):
                add(f, layout.verification)
            elif n.startswith("baseline_comparison") and n.endswith((".csv", ".png")):
                add(f, os.path.join(layout.baselines, "migrated"))
            elif n == "batch_log.csv":
                add(f, layout.logs)
            elif re.match(r"^cdse_.*manifest.*\.csv$", n):
                add(f, layout.manifests)
            elif re.match(r"^antecedent_state_labels.*\.csv$", n):
                add(f, layout.labels)
            elif n.startswith("selected_catchments") and n.endswith(".csv"):
                add(f, layout.selection)
        rs = sdir / "raw_samples"
        if rs.is_dir():
            for sub in sorted(p for p in rs.iterdir() if p.is_dir()):
                for f in sorted(sub.glob("*.tif")):
                    dst = Path(layout.raw_samples) / sub.name / f.name
                    if dst.exists():
                        notes.append(f"skip (already exists at destination): {f}")
                    else:
                        plan.append((f, dst))
    return plan, notes


def main():
    ap = argparse.ArgumentParser(description="Show / create / migrate into the results layout.")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--init", action="store_true", help="create the folders")
    ap.add_argument("--migrate-from", nargs="+", default=None, metavar="DIR",
                    help="folders to scan for outputs that should live in the layout")
    ap.add_argument("--apply", action="store_true", help="actually move the files (default: only list)")
    ap.add_argument("--migrate-embeddings", action="store_true",
                    help="move the old shared embeddings/ folder (Clay) into embeddings_clay/ (dry run unless --apply)")
    args = ap.parse_args()

    lay = Layout(args.results_root)
    print(f"Results root: {lay.root}")
    for name, d in [("embeddings_clay", lay._p("embeddings_clay")), ("embeddings_presto", lay._p("embeddings_presto")),
                    ("cache", lay.cache), ("raw_samples", lay.raw_samples),
                    ("verification", lay.verification), ("baselines", lay.baselines),
                    ("selection", lay.selection), ("logs", lay.logs), ("checks", lay.checks),
                    ("manifests", lay.manifests), ("labels", lay.labels),
                    ("scratch/downloads", lay.scratch_downloads)]:
        print(f"  {name:<18}{d}{'' if os.path.isdir(d) else '   (not created yet)'}")
    if args.init:
        lay.ensure(*lay.all_dirs())
        print("\nFolders created.")

    if args.migrate_embeddings:
        old, new = Path(lay.embeddings), Path(lay._p("embeddings_clay"))
        files = sorted(p for p in old.iterdir() if p.is_file() and re.match(r"^(embeddings_.+\.npz|indices_.+\.csv)$", p.name)) if old.is_dir() else []
        clash = [p.name for p in files if (new / p.name).exists()]
        print(f"\n{len(files)} Clay file(s) in {old} -> {new}" + (f"  ({len(clash)} already exist there, would be skipped)" if clash else ""))
        if args.apply and files:
            new.mkdir(parents=True, exist_ok=True)
            for p in files:
                if p.name not in clash:
                    shutil.move(str(p), str(new / p.name))
            print("Moved. The old folder is left in place (empty apart from anything else it held).")
        elif files:
            print("Dry run only. Re-run with --apply to move them.")

    if args.migrate_from:
        plan, notes = _plan_migration(args.migrate_from, lay)
        print(f"\n{len(plan)} file(s) to move:")
        for s, d in plan:
            print(f"  {s}\n    -> {d}")
        for n in notes:
            print(f"  note: {n}")
        if args.apply and plan:
            for s, d in plan:
                os.makedirs(d.parent, exist_ok=True)
                shutil.move(str(s), str(d))
            print(f"\nMoved {len(plan)} file(s).")
        elif plan:
            print("\nDry run only. Re-run with --apply to move them.")


if __name__ == "__main__":
    main()
