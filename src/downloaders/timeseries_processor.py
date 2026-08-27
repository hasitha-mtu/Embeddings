"""
timeseries_processor.py
-----------------------
Post-download processing of OPW 15-minute dataset:

1.  Load raw CSVs (output of WaterLevelDownloader / RainfallDownloader)
2.  Build a complete 15-minute DatetimeIndex for the analysis window
3.  Resample / align to that regular grid
4.  Also produce hourly aggregates (mean for WL, sum for rainfall)
5.  Gap detection and quality report
6.  Save combined wide-format CSVs ready for the graph builder

Output files
------------
processed/combined_water_level_15min.csv   -- wide, stations as columns
processed/combined_water_level_hourly.csv
processed/combined_rainfall_15min.csv
processed/combined_rainfall_hourly.csv
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeriesProcessor:

    def __init__(self, config: dict):
        self.config        = config
        self.quality_cfg   = config["quality"]
        self.proc_cfg      = config["processing"]
        self.processed_dir = Path(config["output"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # -- Public entry point ----------------------------------------------------

    def process(
        self,
        waterlevel_stations: list,
        start_date: str,
        end_date: str,
    ) -> dict:
        """
        Align all downloaded stations to a common 15-min grid within
        [start_date, end_date], produce quality reports and combined CSVs.
        """
        freq_15  = self.proc_cfg["resample_freq"]   # "15min"
        freq_1h  = self.proc_cfg["hourly_freq"]      # "h"

        full_15min = pd.date_range(
            start=start_date, end=end_date, freq=freq_15, tz="UTC"
        )
        logger.info(
            "Processing window: %s -> %s  (%d x 15-min timesteps)",
            start_date, end_date, len(full_15min),
        )

        results = {"rainfall": {}, "water_level": {}, "discharge": {}}

        # -- Water level -------------------------------------------------------
        wl_frames_15, wl_frames_1h = {}, {}
        for ref in waterlevel_stations:
            path = Path(self.config["output"]["raw_water_level_dir"]) / f"wl_{ref}.csv"
            if not path.exists():
                logger.warning("Missing raw file: %s", path)
                continue
            rep, s15, s1h = self._process_one(path, ref, full_15min, kind="water_level")
            results["water_level"][ref] = rep
            wl_frames_15[ref] = s15
            wl_frames_1h[ref] = s1h

        # -- Save combined CSVs ------------------------------------------------
        self._save_combined(wl_frames_15,   full_15min, "combined_water_level_15min.csv")
        self._save_combined(wl_frames_1h,   full_15min.floor(freq_1h).unique(), "combined_water_level_hourly.csv")

        self._print_summary(results)
        return results

    # -- Per-station processing ------------------------------------------------

    def _process_one(
        self,
        path: Path,
        ref: str,
        full_15min: pd.DatetimeIndex,
        kind: str,
    ) -> tuple[dict, pd.Series, pd.Series]:
        """
        Load raw CSV, align to 15-min grid, produce hourly aggregate.
        Returns (quality_report, series_15min, series_hourly).
        """
        df = pd.read_csv(path, index_col="datetime", parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        raw_15 = df["value"]

        # -- Resample to regular 15-min grid (dataset is already 15-min, but
        #    may have minor timestamp drift or duplicates from OPW)
        if kind == "rainfall":
            s15 = raw_15.resample("15min").sum()
            s1h = raw_15.resample("h").sum()
        elif kind == "discharge":
            s15 = raw_15.resample("15min").sum()
            s1h = raw_15.resample("h").sum()
        else:
            s15 = raw_15.resample("15min").mean()
            s1h = raw_15.resample("h").mean()

        # Re-index to the complete expected grid
        s15 = s15.reindex(full_15min)
        hourly_index = full_15min.floor("h").unique()
        s1h = s1h.reindex(hourly_index)

        # -- Quality report ----------------------------------------------------
        gaps = self._find_gaps(s15, ref)
        missing_frac = s15.isna().mean()
        completeness = 1.0 - missing_frac
        passes = completeness >= self.quality_cfg["min_record_fraction"]

        # Save processed 15-min file
        out = self.processed_dir / f"proc_{kind}_{ref}.csv"
        s15.to_frame(name="value").to_csv(out)

        report = {
            "station_ref":      ref,
            "kind":             kind,
            "total_15min":      len(full_15min),
            "available_15min":  int(s15.notna().sum()),
            "missing_15min":    int(s15.isna().sum()),
            "completeness_pct": round(completeness * 100, 2),
            "passes_threshold": passes,
            "n_gaps":           len(gaps),
            "longest_gap_h":    max((g["duration_hours"] for g in gaps), default=0),
            "gaps":             gaps,
            "processed_path":   str(out),
        }
        return report, s15, s1h

    # -- Gap detection ---------------------------------------------------------

    def _find_gaps(self, series: pd.Series, ref: str) -> list[dict]:
        max_gap_h = self.quality_cfg["max_gap_hours"]
        gaps, in_gap, gap_start = [], False, None

        for ts, val in series.items():
            if pd.isna(val):
                if not in_gap:
                    in_gap, gap_start = True, ts
            else:
                if in_gap:
                    dur_h = (ts - gap_start).total_seconds() / 3600
                    if dur_h >= max_gap_h:
                        gaps.append({
                            "start": str(gap_start),
                            "end":   str(ts),
                            "duration_hours": round(dur_h, 1),
                        })
                    in_gap = False

        if in_gap and gap_start is not None:
            dur_h = (series.index[-1] - gap_start).total_seconds() / 3600
            if dur_h >= max_gap_h:
                gaps.append({
                    "start": str(gap_start),
                    "end":   str(series.index[-1]),
                    "duration_hours": round(dur_h, 1),
                })

        gaps.sort(key=lambda g: g["duration_hours"], reverse=True)
        return gaps

    # -- Combined output -------------------------------------------------------

    def _save_combined(self, frames: dict, index: pd.DatetimeIndex, filename: str):
        if not frames:
            return
        combined = pd.DataFrame(frames, index=index)
        out = self.processed_dir / filename
        combined.to_csv(out)
        logger.info("Saved -> %s  (%d rows x %d stations)",
                    out, len(combined), len(frames))

    # -- Summary ---------------------------------------------------------------

    def _print_summary(self, results: dict):
        logger.info("")
        logger.info("=" * 65)
        logger.info("  DATA QUALITY SUMMARY")
        logger.info("=" * 65)
        for kind, reps in results.items():
            if not reps:
                continue
            logger.info("  %s", kind.upper().replace("_", " "))
            for ref, r in reps.items():
                flag = "[OK]" if r.get("passes_threshold") else "[!!]"
                logger.info(
                    "    %s  %-12s  complete: %5.1f%%  gaps: %2d  "
                    "(longest: %.0f h)",
                    flag, ref,
                    r.get("completeness_pct", 0),
                    r.get("n_gaps", 0),
                    r.get("longest_gap_h", 0),
                )
        logger.info("=" * 65)
