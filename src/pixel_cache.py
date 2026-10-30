r"""
pixel_cache.py -- a small, permanent record of the imagery, written during the
normal extraction pass, so the raw GeoTIFFs are not needed again.

WHY: the raw imagery is large, but every encoder that is not Clay (Presto, or
any other time-series model tried later) needs pixel VALUES over time, not a
Clay embedding. Re-downloading a year of imagery for each new encoder is the
expensive part. This cache keeps just enough to run those encoders later:

    results/cache/pixels_<station>_<sensor>_<year>.npz
        dates     (n_dates,)                  'YYYY-MM-DD'
        bands     (n_bands,)                  e.g. ['VV_dB', 'VH_dB']
        values    (n_dates, n_pixels, n_bands) float32, NaN = no valid data
        x, y      (n_pixels,)                 pixel centres in the raster CRS
        lon, lat  (n_pixels,)                 the same locations in EPSG:4326
        meta      JSON: station, sensor, year, n_pixels, window, seed, crs
    results/cache/stats_<station>_<sensor>_<year>.csv
        the per-date catchment statistics (same columns as indices_*.csv)

HOW THE PIXELS ARE CHOSEN: --cache-pixels (default 256) locations drawn at random
(seeded by the station id, so every year and every sensor uses the SAME
locations) from the catchment polygon. At each location the value is the mean
of a small window (default 3x3 = 30 m), which damps radar speckle. Sentinel-1
is stored in dB (averaged in dB, as compute_s1_stats does); Sentinel-2 as the 10
spectral bands in the file's own units. Locations are stored as coordinates, so
the cache stays correct even if the raster grid shifts between dates.

SIZE: a station-year with 120 dates and 256 pixels is about 0.25 MB (S1).

This module has no dependency on any model.
"""

import json
import os
import zlib

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform as warp_transform

S1_CACHE_BANDS = ["VV_dB", "VH_dB"]
S2_CACHE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DB_FLOOR = 1e-6


def cache_paths(cache_dir, station_id, sensor, year):
    return (os.path.join(cache_dir, f"pixels_{station_id}_{sensor}_{year}.npz"),
            os.path.join(cache_dir, f"stats_{station_id}_{sensor}_{year}.csv"))


def station_seed(station_id):
    return zlib.crc32(str(station_id).encode("utf-8")) & 0x7FFFFFFF


class PixelCacheBuilder:
    """Collects sampled-pixel values date by date. Call add() once per date, then save()."""

    def __init__(self, station_id, sensor, year=None, n_pixels=256, window=3):
        if window % 2 == 0 or window < 1:
            raise ValueError("window must be a positive odd number")
        self.station_id, self.sensor, self.year = str(station_id), sensor, year
        self.n_pixels, self.window = int(n_pixels), int(window)
        self.seed = station_seed(station_id)
        self.bands = S1_CACHE_BANDS if sensor == "s1" else S2_CACHE_BANDS
        self.dates, self.values = [], []
        self.x = self.y = self.lon = self.lat = None
        self.crs_str = None

    # -- sampling ----------------------------------------------------------
    def _choose_pixels(self, transform, crs, catchment_mask):
        h = self.window // 2
        eligible = catchment_mask.copy()
        if h:
            eligible[:h, :] = eligible[-h:, :] = False
            eligible[:, :h] = eligible[:, -h:] = False
        flat = np.flatnonzero(eligible)
        if flat.size == 0:
            raise ValueError("no catchment pixels available to sample")
        rng = np.random.default_rng(self.seed)
        n = min(self.n_pixels, flat.size)
        pick = np.sort(rng.choice(flat, size=n, replace=False))
        rows, cols = np.unravel_index(pick, catchment_mask.shape)
        xs, ys = xy(transform, rows, cols)               # pixel centres
        self.x, self.y = np.asarray(xs, float), np.asarray(ys, float)
        lon, lat = warp_transform(crs, "EPSG:4326", self.x.tolist(), self.y.tolist())
        self.lon, self.lat = np.asarray(lon, float), np.asarray(lat, float)
        self.crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)

    # -- per date ------------------------------------------------------------
    def add(self, date, array, transform, crs, valid_mask, catchment_mask=None):
        """array: (bands, H, W) as read from the GeoTIFF (NaN = no data; S1 in linear power).
        valid_mask: True where the pixel is usable on this date (inside the catchment, and
        cloud-free for S2). Locations are fixed on the first call."""
        if self.x is None:
            self._choose_pixels(transform, crs, catchment_mask if catchment_mask is not None else valid_mask)

        if self.sensor == "s1":
            data = 10.0 * np.log10(np.clip(array[:2], DB_FLOOR, None))
        else:
            data = array[:10].astype("float32", copy=True)
        data = data.astype("float32", copy=True)
        data[:, ~valid_mask] = np.nan
        data[~np.isfinite(data)] = np.nan
        if self.sensor == "s1":                         # both polarisations or neither
            both = np.isfinite(data[0]) & np.isfinite(data[1])
            data[:, ~both] = np.nan

        n_b, H, W = data.shape
        rows, cols = rowcol(transform, self.x, self.y)
        rows, cols = np.asarray(rows, int), np.asarray(cols, int)
        h = self.window // 2
        acc = np.zeros((n_b, len(rows)), float)
        cnt = np.zeros((n_b, len(rows)), float)
        for dr in range(-h, h + 1):
            for dc in range(-h, h + 1):
                r, c = rows + dr, cols + dc
                inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
                v = np.full((n_b, len(rows)), np.nan)
                v[:, inb] = data[:, r[inb], c[inb]]
                ok = np.isfinite(v)
                acc += np.where(ok, v, 0.0)
                cnt += ok
        need = int(np.ceil(self.window * self.window / 2))      # at least half the window valid
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt >= need, acc / np.maximum(cnt, 1), np.nan)
        self.dates.append(str(date))
        self.values.append(mean.T.astype("float32"))             # (n_pixels, n_bands)

    # -- output --------------------------------------------------------------
    def save(self, npz_path, stats_rows=None, stats_path=None):
        if not self.dates:
            raise ValueError("pixel cache is empty: no dates were added")
        meta = {"station_id": self.station_id, "sensor": self.sensor, "year": self.year,
                "n_pixels": int(len(self.x)), "window": self.window, "seed": self.seed,
                "crs": self.crs_str, "db": self.sensor == "s1"}
        np.savez_compressed(
            npz_path, dates=np.array(self.dates), bands=np.array(self.bands),
            values=np.stack(self.values).astype("float32"),
            x=self.x, y=self.y, lon=self.lon, lat=self.lat, meta=np.array(json.dumps(meta)))
        if stats_rows is not None and stats_path is not None:
            pd.DataFrame(stats_rows).to_csv(stats_path, index=False)


def verify_cache(npz_path, stats_path, min_dates=5):
    """(ok, detail). Never trust a cache that fails this: raw imagery must not be deleted on it."""
    if not os.path.exists(npz_path) or not os.path.exists(stats_path):
        return False, "cache file(s) missing"
    if os.path.getsize(npz_path) == 0 or os.path.getsize(stats_path) == 0:
        return False, "cache file(s) empty"
    try:
        z = np.load(npz_path, allow_pickle=False)
        dates, values = z["dates"], z["values"]
        stats = pd.read_csv(stats_path)
    except Exception as e:
        return False, f"cache unreadable: {e}"
    if len(dates) < min_dates:
        return False, f"only {len(dates)} dates in the cache (expected >= {min_dates})"
    if values.ndim != 3 or values.shape[0] != len(dates):
        return False, f"values shape {values.shape} does not match {len(dates)} dates"
    if len(stats) != len(dates):
        return False, f"cache has {len(dates)} dates but stats has {len(stats)} rows"
    frac = float(np.isfinite(values).mean())
    if frac == 0.0:
        return False, "cache contains no finite values"
    return True, f"{len(dates)} dates, {values.shape[1]} pixels, {100 * frac:.0f}% finite"


def load_cache(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out
