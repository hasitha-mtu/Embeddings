"""
base_downloader.py
──────────────────
Abstract base class for OPW ZIP-based bulk downloaders.
Each station = one ZIP file containing tsvalues.csv.
"""

import io
import logging
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    pass


class BaseDownloader(ABC):

    def __init__(self, config: dict):
        self.config      = config
        self.api_cfg     = config["api"]
        self.quality_cfg = config["quality"]
        self.out_cfg     = config["output"]

        self.timeout     = self.api_cfg["timeout"]
        self.max_retries = self.api_cfg["max_retries"]
        self.backoff     = self.api_cfg["retry_backoff"]
        self.rate_limit  = self.api_cfg["rate_limit_delay"]

        self._session: Optional[requests.Session] = None
        self._last_request_time: float = 0.0

    # ── Session ───────────────────────────────────────────────────────────────

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            retry = Retry(
                total=self.max_retries,
                backoff_factor=self.backoff,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            self._session.mount("https://", adapter)
            self._session.mount("http://",  adapter)
            self._session.headers["User-Agent"] = (
                "FloodForecastResearch/1.0"
            )
        return self._session

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # ── HTTP + ZIP helpers ────────────────────────────────────────────────────

    def _rate_wait(self):
        elapsed = time.monotonic() - self._last_request_time
        wait = self.rate_limit - elapsed
        if wait > 0:
            time.sleep(wait)

    def download_zip(self, url: str) -> bytes:
        """GET a ZIP URL, return raw bytes with retry logic."""
        self._rate_wait()
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout, stream=True)
                self._last_request_time = time.monotonic()
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 404:
                    raise DownloadError(f"404 — station not found at {url}")
                logger.warning("Attempt %d/%d — HTTP %d: %s",
                               attempt, self.max_retries, resp.status_code, url)
            except requests.RequestException as exc:
                logger.warning("Attempt %d/%d — %s", attempt, self.max_retries, exc)
            if attempt < self.max_retries:
                time.sleep(self.backoff ** attempt)
        raise DownloadError(f"Failed after {self.max_retries} attempts: {url}")

    def extract_tsvalues(self, zip_bytes: bytes, zip_entry: str) -> str:
        """Extract tsvalues.csv text from ZIP bytes."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            # Find the entry (case-insensitive, handles sub-folders)
            match = next(
                (n for n in names if n.lower().endswith(zip_entry.lower())), None
            )
            if match is None:
                raise DownloadError(
                    f"'{zip_entry}' not found in ZIP. Contents: {names}"
                )
            return zf.read(match).decode("utf-8", errors="replace")

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @abstractmethod
    def download(self, stations: list[dict]) -> dict:
        ...
