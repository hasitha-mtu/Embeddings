import logging
import sys
from pathlib import Path

def setup_logging(log_cfg: dict):
    log_dir = Path(log_cfg["log_file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt   = "%(asctime)s  %(levelname)-8s  %(message)s"

    # Force UTF-8 on both handlers so Windows cp1252 terminals don't break
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream.reconfigure(encoding="utf-8", errors="replace") if hasattr(stream_handler.stream, "reconfigure") else None

    file_handler = logging.FileHandler(log_cfg["log_file"], encoding="utf-8")

    logging.basicConfig(level=level, format=fmt,
                        handlers=[stream_handler, file_handler])

def get_logger(log_cfg: dict) -> logging.Logger:
    setup_logging(log_cfg)
    logger = logging.getLogger(__name__)
    return logger