"""
discharge_downloader.py
------------------------
Downloads OPW discharge dataset via bulk ZIP download.

URL pattern:
 https://waterlevel.ie/hydro-data/data/internet/stations/0/{station_no}/Q/Discharge_complete.zip

"""

import io
import logging
from pathlib import Path

import pandas as pd

from .base_downloader import BaseDownloader, DownloadError

logger = logging.getLogger(__name__)


class DischargeDownloader(BaseDownloader):

    def __init__(self, config: dict):
        super().__init__(config)
        self.url_template = self.api_cfg["discharge_zip_url"]
        self.zip_entry    = self.api_cfg["zip_entry_name"]
        self.out_dir      = Path(self.out_cfg["raw_discharge_dir"])
        self.ensure_dir(self.out_dir)

    # -- Public interface ------------------------------------------------------

    def download(self, stations: list[dict]) -> dict:
        """
        Download discharge ZIP for every station.

        Returns
        -------
        summary dict keyed by station ref
        """
        summary = {}
        for station in stations:
            ref  = station["ref"]
            name = station.get("name", ref)
            logger.info("-- Discharge: %s (%s)", name, ref)

            url = self.url_template.format(station_no=ref)
            try:
                zip_bytes  = self.download_zip(url)
                csv_text   = self.extract_tsvalues(zip_bytes, self.zip_entry)
                meta, df   = self._parse(csv_text, ref)
                df         = self._quality_filter(df)
                out_path   = self._save(df, ref)

                missing_pct = round(df["value"].isna().mean() * 100, 2)
                summary[ref] = {
                    "name":        name,
                    "status":      "ok",
                    "records":     len(df),
                    "start":       str(df.index.min()),
                    "end":         str(df.index.max()),
                    "missing_pct": missing_pct,
                    "mean_discharge_m3": round(df["value"].mean(skipna=True), 3),
                    "path":        str(out_path),
                    "meta":        meta,
                }
                logger.info(
                    "   [OK] %d records @ 15-min  |  missing %.1f%%  "
                    "|  mean %.3f m OD  |  %s -> %s",
                    len(df), missing_pct,
                    summary[ref]["mean_discharge_m3"],
                    summary[ref]["start"][:10],
                    summary[ref]["end"][:10],
                )

            except DownloadError as exc:
                logger.error("   [!!] %s: %s", ref, exc)
                summary[ref] = {"name": name, "status": "failed", "error": str(exc)}
            except Exception as exc:
                logger.error("   [!!] Unexpected error for %s: %s", ref, exc, exc_info=True)
                summary[ref] = {"name": name, "status": "error", "error": str(exc)}

        return summary

    # -- Parsing ---------------------------------------------------------------

    def _parse(self, text: str, station_ref: str) -> tuple[dict, pd.DataFrame]:
        """
        Parse OPW tsvalues.csv for discharge.

        Returns (metadata_dict, DataFrame with DatetimeIndex).

        Columns kept:
          value        -- discharge in metres OD
          quality_code -- OPW quality flag (254 = good)
          quality_ok   -- bool True if quality_code == 254
        """
        meta: dict = {}
        data_lines: list[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Parse metadata: #key;value
                content = line.lstrip("#")
                if ";" in content:
                    k, v = content.split(";", 1)
                    meta[k.strip()] = v.strip()
                # Skip the column-header comment line
            else:
                data_lines.append(line)

        if not data_lines:
            raise DownloadError(f"No dataset rows found for station {station_ref}")

        # tsvalues.csv has NO non-comment header -- columns are defined by #Timestamp;...
        # We parse directly
        raw = "\n".join(data_lines)
        df = pd.read_csv(
            io.StringIO(raw),
            sep=";",
            header=None,
            names=["timestamp", "value", "quality_code"],
            usecols=[0, 1, 2],
        )

        # Parse timestamps (ISO-8601 with Z suffix)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df.index.name = "datetime"

        df["value"]        = pd.to_numeric(df["value"],        errors="coerce")
        df["quality_code"] = pd.to_numeric(df["quality_code"], errors="coerce").fillna(0).astype(int)
        df["quality_ok"]   = df["quality_code"] == self.quality_cfg["good_quality_code"]
        df["station_ref"]  = station_ref

        return meta, df

    # -- Quality filter --------------------------------------------------------

    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag physically impossible OD values as NaN (keep quality_code intact)."""
        lo, hi = self.quality_cfg["discharge_range"]
        mask = (df["value"] < lo) | (df["value"] > hi)
        if mask.any():
            logger.debug("   %d values outside [%.0f, %.0f] m OD -> NaN",
                         int(mask.sum()), lo, hi)
            df.loc[mask, "value"] = float("nan")
        return df

    # -- Save ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame, station_ref: str) -> Path:
        out_path = self.out_dir / f"discharge_{station_ref}.csv"
        df.to_csv(out_path)
        return out_path
