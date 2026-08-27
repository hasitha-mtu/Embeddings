import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from downloaders.waterlevel_downloader import WaterLevelDownloader
from downloaders.discharge_downloader import DischargeDownloader
from downloaders.timeseries_processor  import TimeSeriesProcessor

from utils.config import load_config
from utils.logger import get_logger


def get_subset(config: dict, name: str = None) -> tuple[str, dict]:
    name = name or config["active_subset"]
    subset = config["subsets"].get(name)
    if subset is None:
        raise ValueError(f"Unknown subset '{name}'. Available: {list(config['subsets'])}")
    return name, subset


def skip_existing(stations: list[dict], raw_dir: Path, prefix: str) -> list[dict]:
    keep = []
    for s in stations:
        p = raw_dir / f"{prefix}{s['ref']}.csv"
        if p.exists():
            logging.getLogger(__name__).info(
                "   Skip (exists): %s (%s)", s.get("name"), s["ref"]
            )
        else:
            keep.append(s)
    return keep


def main():
    ap = argparse.ArgumentParser(description="OPW hydrological bulk downloader")
    ap.add_argument("--config",        default="C:\\Users\AdikariAdikari\PycharmProjects\Embeddings\config\config.yaml")
    ap.add_argument("--subset",        default=None)
    ap.add_argument("--start",         default=None, help="Analysis start YYYY-MM-DD")
    ap.add_argument("--end",           default=None, help="Analysis end YYYY-MM-DD")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--no-process",    action="store_true")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    logger = get_logger(config["logging"]["downloader"])
    start_date = args.start or "2020-01-01"  # ZIPs contain full history; this window is for processing only
    end_date   = args.end   or "2026-08-10"

    station_df = pd.read_csv(config["station_overview"])
    wl_stations = station_df["Station No."].astype(str).tolist()

    logger.info("+==========================================================+")
    logger.info("|         OPW Bulk ZIP Downloader                         |")
    logger.info("+==========================================================+")
    logger.info("Analysis     : %s -> %s", start_date, end_date)
    logger.info("Water level  : %d station(s)", len(wl_stations))
    logger.info("Water level URL pattern  : %s", config["api"]["waterlevel_zip_url"].format(station_no="XXXXX"))

    if args.dry_run:
        logger.info("\nDRY RUN -- no HTTP calls will be made")

        logger.info("\nWater level stations:")
        for s in wl_stations:
            url = config["api"]["waterlevel_zip_url"].format(station_no=s["ref"])
            logger.info("  * %-28s  %s  ->  %s", s.get("name"), s["ref"], url)

        return

    # -- Optionally skip already-downloaded stations ------------------------
    wl_dl   = wl_stations
    if args.skip_existing:
        wl_dl   = skip_existing(wl_stations,   Path(config["output"]["raw_water_level_dir"]), "wl_")

    # -- Save station metadata ---------------------------------------------
    meta_dir = Path(config["output"]["metadata_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    if wl_stations:
        pd.DataFrame(wl_stations).to_csv(meta_dir / "waterlevel_stations.csv", index=False)
    summaries = {}
    t0 = datetime.now()

    # -- Download water level ----------------------------------------------
    if wl_dl:
        logger.info("\n---  Downloading WATER LEVEL  (%d stations)  -------------", len(wl_dl))
        with WaterLevelDownloader(config) as dl:
            summaries["water_level"] = dl.download(wl_dl)

    logger.info("\nDownload complete in %.1f s", (datetime.now() - t0).total_seconds())

    # -- Process ------------------------------------------------------------
    if not args.no_process:
        logger.info("\n---  Processing & Quality Report  -------------------------")
        processor = TimeSeriesProcessor(config)
        summaries["quality"] = processor.process(
            waterlevel_stations= wl_stations,
            start_date=start_date,
            end_date=end_date,
        )

    # -- Save summary JSON -------------------------------------------------
    summary_path = meta_dir / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    logger.info("\nSummary -> %s", summary_path)

    # Exit 1 if any station failed
    failed = [
        ref for d in summaries.values() if isinstance(d, dict)
        for ref, info in d.items()
        if isinstance(info, dict) and info.get("status") == "failed"
    ]
    if failed:
        logger.warning("Failed stations: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
