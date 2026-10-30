r"""
extract_embeddings.py  (Component 5: embedding extraction)

Reads a folder of downloaded, catchment-clipped Sentinel-1 or Sentinel-2
GeoTIFFs (one per date, as produced by download_cdse_results.py), and for
each date:
  1. Masks pixels to the catchment's actual polygon (not just the raster's
     bounding box).
  2. For Sentinel-2, also masks out cloud/shadow/nodata pixels using the SCL
     band, and computes hand-crafted indices (NDVI, NDWI, MNDWI) as a
     permanent, tiny fallback/comparison feature set -- kept even after the
     raw imagery is deleted, since Component 10 (interpretability) needs
     these later and they cost almost nothing to store.
  3. Tiles the valid area into patches, runs each patch through a pluggable
     encoder, and mean-pools patch embeddings into one catchment-level
     vector for that date.

Output: one compact .npz (embedding vectors, keyed by date) and one small
.csv (hand-crafted indices + metadata) per catchment per sensor -- both tiny
enough to keep permanently, unlike the raw imagery.

ENCODER HONESTY NOTE:
  - DummyEncoder (mean+std per band) is the only encoder that has been
    tested end-to-end -- it validates that the whole pipeline (masking,
    tiling, pooling, I/O) works correctly, via synthetic-GeoTIFF tests.
  - ClayEncoder: checkpoint LOADING works on your machine (you reached the
    forward pass). The forward pass is now sensor-aware: it reads the band
    wavelengths / normalization mean+std / GSD for the right platform from
    Clay's own metadata.yaml instead of assuming Sentinel-2. Three details
    are still written from my recollection of Clay's embedding examples, not
    verified against the live repo (no network access to it from here), and
    are marked UNVERIFIED in the class: (a) loading with mask_ratio=0.0 /
    shuffle=False, (b) taking the CLS token as the embedding, (c) the
    time/latlon conditioning vectors (still zero placeholders). Confirm
    against Clay's current example notebook before trusting the numbers.
  - Presto is NOT run from this script. It encodes a per-location time series, not a
    single-date tile, so it runs from the compact pixel cache this script now writes
    (see pixel_cache.py) via extract_presto.py. Nothing about Presto lives here.

PIXEL CACHE: while it reads each date's raster, this script can also write a small
permanent cache of sampled-pixel values (pixel_cache.py), so a later encoder never
needs the raw imagery again. It is written by the same pass, to the same temp-then-
replace rules as the embeddings (see run_catchment_pipeline.py). With no encoder
(cache-only mode) it writes just the cache and the per-date statistics.

Usage (dummy encoder, to test the pipeline):
    python extract_embeddings.py \
        --input-dir ./sentinel_imagery/s2_11001_2023 \
        --shapefile subcatchments/subcatchments.shp --subcatchment-id 11_5 \
        --sensor s2 --encoder dummy \
        --out-embeddings embeddings_11001_s2_2023.npz \
        --out-indices indices_11001_s2_2023.csv
    (--out-* are optional: without them a standalone run writes to
     <results root>/checks/, see project_paths.py)

Usage (Clay):
    python extract_embeddings.py ... --encoder clay \
        --clay-checkpoint /path/to/clay-v1.5.ckpt \
        --clay-metadata /path/to/clay/configs/metadata.yaml

Requires: numpy, pandas, rasterio, shapely, geopandas
Optional (only if using --encoder clay): torch, pyyaml, and the model's own
package (claymodel). These are imported lazily, inside the encoder classes, so
the dummy-encoder path works even if claymodel isn't installed.
"""

import argparse
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from pixel_cache import PixelCacheBuilder

# NOTE: no module-level `from claymodel.module import ClayMAEModule` on
# purpose -- that import lives inside ClayEncoder.__init__ so this script
# still runs (e.g. with --encoder dummy) on a machine that doesn't have the
# claymodel package installed.

# Clay's own band/normalization metadata (ships with the Clay repo under configs/).
MODEL_METADATA_PATH = "/checkpoints/clay/configs/metadata.yaml"

# SCL classes to exclude as invalid: 0=no data, 1=saturated, 3=cloud shadow,
# 8/9=cloud medium/high probability, 10=thin cirrus
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}

S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S1_BAND_ORDER = ["VV", "VH"]

# How this pipeline's band names map onto the band names used in Clay's
# metadata.yaml, per platform. (Band names in the yaml are looked up by name,
# not by position, so the order of the yaml's own band_order doesn't matter.)
CLAY_PLATFORM = {"s1": "sentinel-1-rtc", "s2": "sentinel-2-l2a"}
CLAY_BAND_NAMES = {
    "s1": {"VV": "vv", "VH": "vh"},
    "s2": {"B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1",
           "B06": "rededge2", "B07": "rededge3", "B08": "nir", "B8A": "nir08",
           "B11": "swir16", "B12": "swir22"},
}


# --------------------------------------------------------------------------
# Encoders -- pluggable. DummyEncoder is tested; Clay has verified checkpoint
# loading with a few forward-pass details still unverified (see class docs).
# --------------------------------------------------------------------------

class BaseEncoder:
    patch_size = 64
    embedding_dim = None

    def encode_batch(self, patches):
        """patches: array (N, bands, patch_size, patch_size) -> (N, embedding_dim)"""
        raise NotImplementedError


class DummyEncoder(BaseEncoder):
    """Mean + std per band, as a fixed-length stand-in embedding. This is
    NOT a real foundation-model embedding -- it exists to validate the data
    pipeline (masking/tiling/pooling/I-O) end-to-end without needing real
    model weights. Verified working via synthetic-GeoTIFF tests."""

    def __init__(self, n_bands):
        self.patch_size = 64
        self.embedding_dim = n_bands * 2

    def encode_batch(self, patches):
        means = np.nanmean(patches, axis=(2, 3))
        stds = np.nanstd(patches, axis=(2, 3))
        return np.concatenate([means, stds], axis=1)


def load_clay_band_metadata(metadata_path, sensor):
    """Reads (wavelengths, means, stds, gsd) for this pipeline's bands for
    `sensor` ('s1' or 's2') out of Clay's metadata.yaml, in this pipeline's
    own band order. Raises a descriptive error if the yaml doesn't have the
    structure this expects (platform -> bands -> mean/std/wavelength -> name),
    rather than silently falling back to guessed values.
    """
    import yaml

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    platform = CLAY_PLATFORM[sensor]
    if platform not in meta:
        raise KeyError(f"platform '{platform}' not found in {metadata_path}; "
                       f"available top-level keys: {list(meta)}")
    entry = meta[platform]

    band_map = CLAY_BAND_NAMES[sensor]
    try:
        bands = entry["bands"]
        waves = [float(bands["wavelength"][clay_name]) for clay_name in band_map.values()]
        means = [float(bands["mean"][clay_name]) for clay_name in band_map.values()]
        stds = [float(bands["std"][clay_name]) for clay_name in band_map.values()]
    except KeyError as e:
        raise KeyError(f"metadata for '{platform}' in {metadata_path} is missing {e}; "
                       f"expected bands -> wavelength/mean/std -> {list(band_map.values())}. "
                       f"Found under 'bands': {list(entry.get('bands', {}))}") from e
    gsd = float(entry.get("gsd", 10.0))
    return waves, means, stds, gsd


class ClayEncoder(BaseEncoder):
    """Clay v1.5 patch encoder, sensor-aware.

    WHY `sensor` IS REQUIRED: Clay's patch embedding builds its convolution
    weights on the fly from the wavelengths of the bands you pass in. If the
    wavelengths describe 10 bands but the pixels have 2 (or vice versa) the
    forward pass fails with "expected input[...] to have 10 channels, but got
    2 channels" -- exactly the error you saw when Sentinel-1 patches were fed
    with Sentinel-2's 10 wavelengths. So wavelengths, normalization stats and
    GSD are now looked up per sensor from Clay's metadata.yaml.

    CHECKPOINT LOADING -- works (you reached the forward pass):
        ClayMAEModule.load_from_checkpoint(ckpt, metadata_path=...)

    UNVERIFIED (from my recollection of Clay's embedding examples, not checked
    against the live repo from here -- confirm in Clay's example notebook):
      (a) Loading with mask_ratio=0.0 and shuffle=False. Clay is a masked
          autoencoder; by default it hides most patches, which is wrong when
          you want an embedding of the whole tile. If your claymodel version
          doesn't accept these kwargs, loading will raise a TypeError --
          that is the signal to check the notebook, not to ignore.
      (b) Using the CLS token (index 0 of the encoder output) as the tile's
          embedding. The remaining tokens are per-patch features.
      (c) `time` and `latlon` conditioning vectors are zero placeholders, so
          embeddings don't yet use acquisition date or location.

    Sentinel-1 is converted to dB before normalization, because (to my
    recollection) Clay's Sentinel-1 mean/std in metadata.yaml are in dB
    (means around -12 and -18). CHECK your yaml: if the S1 means are small
    positive numbers instead, set S1_TO_DB = False below.
    """

    S1_TO_DB = True
    DB_FLOOR = 1e-6          # linear floor before log10 (-60 dB), avoids log(0)
    BATCH_SIZE = 64          # patches per forward pass, bounds memory
    embedding_dim = 1024     # Clay v1.5 "large"; matches the [1024, ...] weight in your error

    def __init__(self, checkpoint_path, sensor, metadata_path=MODEL_METADATA_PATH, device="cpu"):
        if not checkpoint_path:
            raise ValueError("ClayEncoder requires --clay-checkpoint")
        if sensor not in CLAY_PLATFORM:
            raise ValueError(f"sensor must be one of {list(CLAY_PLATFORM)}, got {sensor!r}")
        try:
            import torch
        except ImportError:
            raise RuntimeError("ClayEncoder requires torch -- pip install torch")
        try:
            from claymodel.module import ClayMAEModule
        except ImportError:
            raise RuntimeError(
                "ClayEncoder requires the claymodel package -- "
                "pip install git+https://github.com/Clay-foundation/model.git "
                "(or see the repo's current install instructions)"
            )

        self.torch = torch
        self.device = device
        self.sensor = sensor
        self.patch_size = 64
        self.waves_um, self.means, self.stds, self.gsd_m = load_clay_band_metadata(metadata_path, sensor)
        print(f"  ClayEncoder[{sensor}]: wavelengths={self.waves_um} "
              f"means={self.means} stds={self.stds} gsd={self.gsd_m}")

        # UNVERIFIED (a): mask_ratio=0.0 / shuffle=False so every patch is encoded.
        self.model = ClayMAEModule.load_from_checkpoint(
            checkpoint_path, metadata_path=metadata_path, mask_ratio=0.0, shuffle=False)
        self.model.eval()
        self.model.to(device)

    def preprocess(self, patches):
        """(N, C, H, W) raw -> normalized float32 with NaN -> 0 (the dataset
        mean after normalization). numpy-only so it can be tested without torch."""
        x = patches.astype("float32", copy=True)
        if self.sensor == "s1" and self.S1_TO_DB:
            x = 10.0 * np.log10(np.clip(x, self.DB_FLOOR, None))   # NaN stays NaN
        mean = np.asarray(self.means, dtype="float32")[None, :, None, None]
        std = np.asarray(self.stds, dtype="float32")[None, :, None, None]
        x = (x - mean) / std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_datacube(self, pixels_np):
        torch = self.torch
        n = pixels_np.shape[0]
        return {
            "pixels": torch.from_numpy(pixels_np).float().to(self.device),
            # UNVERIFIED (c): zero placeholders, see class docstring.
            "time": torch.zeros((n, 4), dtype=torch.float32, device=self.device),
            "latlon": torch.zeros((n, 4), dtype=torch.float32, device=self.device),
            "waves": torch.tensor(self.waves_um, dtype=torch.float32, device=self.device),
            "gsd": torch.tensor(self.gsd_m, dtype=torch.float32, device=self.device),
        }

    def encode_batch(self, patches):
        if patches.shape[1] != len(self.waves_um):
            raise ValueError(f"{self.sensor} patches have {patches.shape[1]} channels but the "
                             f"metadata defines {len(self.waves_um)} bands")
        x = self.preprocess(patches)
        outputs = []
        with self.torch.no_grad():
            for start in range(0, len(x), self.BATCH_SIZE):
                datacube = self._build_datacube(x[start:start + self.BATCH_SIZE])
                out = self.model.model.encoder(datacube)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                # UNVERIFIED (b): CLS token as the tile embedding.
                emb = out[:, 0, :] if out.dim() == 3 else out
                outputs.append(emb.cpu().numpy())
        return np.concatenate(outputs, axis=0)


# --------------------------------------------------------------------------
# Raster / masking helpers
# --------------------------------------------------------------------------

def read_geotiff(path):
    with rasterio.open(path) as src:
        array = src.read().astype("float32")  # (bands, H, W)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    if nodata is not None:
        array[array == nodata] = np.nan
    return array, transform, crs


def build_catchment_mask(shapefile_path, subcatchment_id, transform, crs, shape):
    """Boolean array, True = inside the catchment polygon, in the raster's own grid."""
    gdf = gpd.read_file(shapefile_path)
    match = gdf[gdf["Subcatchme"] == subcatchment_id]
    if match.empty:
        raise ValueError(f"subcatchment_id {subcatchment_id} not found in {shapefile_path}")
    geom = match.to_crs(crs).geometry.iloc[0]
    # geometry_mask returns True for pixels OUTSIDE the shape by default -- invert
    outside_mask = geometry_mask([geom], transform=transform, invert=False, out_shape=shape)
    return ~outside_mask


def scl_valid_mask(scl_band):
    # read_geotiff turns the file's nodata value into NaN. If the SCL band's
    # nodata tag is 0, SCL class 0 ("no data") becomes NaN and would slip
    # through the != checks below as "valid" (NaN != 0 is True), so NaN is
    # treated as invalid explicitly.
    valid = ~np.isnan(scl_band)
    for cls in SCL_INVALID_CLASSES:
        valid &= (scl_band != cls)
    return valid


def compute_s2_indices(array, catchment_mask, valid_mask):
    band = {name: array[i] for i, name in enumerate(S2_BAND_ORDER)}
    m = catchment_mask & valid_mask
    if m.sum() == 0:
        return {"ndvi_mean": np.nan, "ndwi_mean": np.nan, "mndwi_mean": np.nan, "valid_pixel_frac": 0.0}

    def safe_index(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            idx = (a - b) / (a + b)
        vals = idx[m & np.isfinite(idx)]
        return float(vals.mean()) if vals.size else np.nan

    return {
        "ndvi_mean": safe_index(band["B08"], band["B04"]),
        "ndwi_mean": safe_index(band["B03"], band["B08"]),
        "mndwi_mean": safe_index(band["B03"], band["B11"]),
        "valid_pixel_frac": float(m.sum() / catchment_mask.sum()) if catchment_mask.sum() > 0 else 0.0,
    }


def s1_valid_mask(array, catchment_mask):
    """Catchment pixels where BOTH polarizations have real data. Pixels
    outside the polygon, or in a scene with no coverage, are NaN after
    read_geotiff and are excluded here."""
    return catchment_mask & np.isfinite(array[0]) & np.isfinite(array[1])


def _to_db(x):
    return 10.0 * np.log10(np.clip(x, 1e-6, None))


def compute_s1_stats(array, catchment_mask):
    """Hand-crafted S1 baseline features. Backscatter is averaged in dB
    (per-pixel conversion first), because linear means are dominated by a few
    very bright pixels. The linear vv_mean/vv_std/vh_mean/vh_std columns are
    kept unchanged for backward compatibility."""
    valid = s1_valid_mask(array, catchment_mask)
    n_catchment = int(catchment_mask.sum())
    n_valid = int(valid.sum())
    keys = ["vv_mean", "vv_std", "vh_mean", "vh_std",
            "vv_db_mean", "vv_db_std", "vv_db_median",
            "vh_db_mean", "vh_db_std", "vh_db_median"]
    if n_valid == 0:
        # No usable S1 data over the catchment on this date.
        out = {k: np.nan for k in keys}
        out["valid_pixel_frac"] = 0.0
        return out
    vv = array[0][valid]
    vh = array[1][valid]
    vv_db, vh_db = _to_db(vv), _to_db(vh)
    return {
        "vv_mean": float(vv.mean()),
        "vv_std": float(vv.std()),
        "vh_mean": float(vh.mean()),
        "vh_std": float(vh.std()),
        "vv_db_mean": float(vv_db.mean()),
        "vv_db_std": float(vv_db.std()),
        "vv_db_median": float(np.median(vv_db)),
        "vh_db_mean": float(vh_db.mean()),
        "vh_db_std": float(vh_db.std()),
        "vh_db_median": float(np.median(vh_db)),
        "valid_pixel_frac": float(n_valid / n_catchment) if n_catchment else 0.0,
    }


def tile_patches(array, mask, patch_size, min_valid_frac=0.5):
    """Non-overlapping patches from the masked area. Returns array (N, bands, ps, ps)."""
    bands, h, w = array.shape
    patches = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch_mask = mask[y:y + patch_size, x:x + patch_size]
            if patch_mask.mean() < min_valid_frac:
                continue
            patch = array[:, y:y + patch_size, x:x + patch_size].copy()
            patch[:, ~patch_mask] = np.nan
            patches.append(patch)
    if not patches:
        return np.empty((0, bands, patch_size, patch_size), dtype="float32")
    return np.stack(patches)


def parse_date_from_filename(filename):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return m.group(1) if m else None


# --------------------------------------------------------------------------
# Main per-file / per-directory processing
# --------------------------------------------------------------------------

def process_file(filepath, shapefile_path, subcatchment_id, sensor, encoder, cache=None, date=None):
    """One date's raster -> (embedding, indices). encoder=None means cache-only: no
    embedding is computed (embedding is None). If `cache` (a PixelCacheBuilder) is
    given, the sampled-pixel values for this date are added to it, as the LAST step,
    so a date that fails earlier is in neither the embeddings nor the cache."""
    array, transform, crs = read_geotiff(filepath)
    catchment_mask = build_catchment_mask(shapefile_path, subcatchment_id, transform, crs, array.shape[1:])

    if sensor == "s2":
        valid_mask = scl_valid_mask(array[S2_BAND_ORDER.index("SCL")])
        indices = compute_s2_indices(array, catchment_mask, valid_mask)
        encode_array = array[:10]  # drop SCL before encoding -- it's a label band, not spectral data
        final_mask = catchment_mask & valid_mask
    else:
        indices = compute_s1_stats(array, catchment_mask)
        encode_array = array
        # Only tile where S1 actually has data, so empty scenes yield 0 patches.
        final_mask = s1_valid_mask(array, catchment_mask)

    embedding = None
    if encoder is not None:
        patches = tile_patches(encode_array, final_mask, encoder.patch_size)
        n_patches = len(patches)
        if n_patches == 0:
            dim = encoder.embedding_dim if encoder.embedding_dim else 1
            embedding = np.full(dim, np.nan, dtype="float32")
        else:
            patch_embeddings = encoder.encode_batch(patches)
            embedding = np.nanmean(patch_embeddings, axis=0)
        indices["n_patches_used"] = n_patches

    if cache is not None:
        cache.add(date, array, transform, crs, final_mask, catchment_mask=catchment_mask)
    return embedding, indices


def build_encoder(encoder_name, sensor, clay_checkpoint=None, clay_metadata=MODEL_METADATA_PATH):
    n_bands = len(S2_BAND_ORDER) - 1 if sensor == "s2" else len(S1_BAND_ORDER)
    if encoder_name == "dummy":
        return DummyEncoder(n_bands)
    elif encoder_name == "clay":
        return ClayEncoder(clay_checkpoint, sensor, metadata_path=clay_metadata)
    elif encoder_name in (None, "none"):
        return None
    elif encoder_name == "presto":
        raise ValueError("Presto does not run from the raw imagery. Write the pixel cache here "
                         "(any encoder, or 'none' for cache-only), then run extract_presto.py.")
    raise ValueError(f"Unknown encoder: {encoder_name}")


def run_extraction(sensor, input_dir, shapefile, subcatchment_id, encoder,
                    out_embeddings, out_indices, cache_out=None, cache_station=None,
                    cache_year=None, cache_pixels=256, cache_window=3):
    """Shared implementation used by both the CLI (main()) and the importable
    extract_embeddings() wrapper. `encoder` must already be an encoder INSTANCE
    (build it with build_encoder()), or None for cache-only mode.

    cache_out: (npz_path, stats_csv_path) to also write the pixel cache, else None.
    With encoder=None, out_embeddings / out_indices may be None and only the cache
    is written (cache_out is then required)."""
    if encoder is None and cache_out is None:
        raise ValueError("nothing to do: no encoder and no cache requested")
    files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not files:
        print(f"No .tif files found in {input_dir}", file=sys.stderr)
        return {"n_dates_processed": 0}

    builder = (PixelCacheBuilder(cache_station, sensor, cache_year, cache_pixels, cache_window)
               if cache_out else None)
    embeddings = {}
    rows = []
    n_done = 0
    for fp in files:
        date = parse_date_from_filename(os.path.basename(fp))
        if date is None:
            print(f"  could not parse date from {fp}, skipped")
            continue
        try:
            embedding, indices = process_file(fp, shapefile, subcatchment_id, sensor, encoder,
                                              cache=builder, date=date)
            if embedding is not None:
                embeddings[date] = embedding
            rows.append({"date": date, **indices})
            n_done += 1
            extra = f"{indices['n_patches_used']} patches used, " if "n_patches_used" in indices else ""
            print(f"  {date}: {extra}valid_pixel_frac={indices['valid_pixel_frac']:.2f}")
        except Exception as e:
            print(f"  {date}: ERROR -- {e}")

    if encoder is not None:
        np.savez_compressed(out_embeddings, **embeddings)
        pd.DataFrame(rows).to_csv(out_indices, index=False)
        print(f"\n{len(embeddings)} dates processed.")
        print(f"Embeddings written to: {out_embeddings}")
        print(f"Hand-crafted indices written to: {out_indices}")
    if builder is not None and n_done:
        builder.save(cache_out[0], stats_rows=rows, stats_path=cache_out[1])
        print(f"Pixel cache ({n_done} dates) written to: {cache_out[0]}")
    return {"n_dates_processed": n_done}


def extract_embeddings(sensor, input_dir, shapefile, subcatchment_id, encoder,
                        out_embeddings, out_indices,
                        clay_checkpoint="checkpoints/clay/clay-v1.5.ckpt",
                        clay_metadata=MODEL_METADATA_PATH,
                        cache_out=None, cache_station=None, cache_year=None,
                        cache_pixels=256, cache_window=3):
    """Importable entry point (e.g. for run_catchment_pipeline.py) mirroring the CLI.
    `encoder` here is a STRING ("dummy"/"clay"/"none"), matching --encoder; "none" =
    cache-only (cache_out required). This function builds the encoder instance."""
    encoder_instance = build_encoder(encoder, sensor, clay_checkpoint, clay_metadata)
    return run_extraction(sensor, input_dir, shapefile, subcatchment_id, encoder_instance,
                           out_embeddings, out_indices, cache_out=cache_out,
                           cache_station=cache_station, cache_year=cache_year,
                           cache_pixels=cache_pixels, cache_window=cache_window)


def main():
    parser = argparse.ArgumentParser(description="Extract pooled catchment embeddings (and the pixel cache) from downloaded imagery.")
    parser.add_argument("--input-dir", required=True, help="Folder of downloaded GeoTIFFs for one catchment+sensor")
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--subcatchment-id", required=True)
    parser.add_argument("--sensor", required=True, choices=["s1", "s2"])
    parser.add_argument("--encoder", default="dummy", choices=["dummy", "clay", "none"],
                        help="'none' = write only the pixel cache (needs --cache)")
    parser.add_argument("--clay-checkpoint", default="checkpoints/clay/clay-v1.5.ckpt")
    parser.add_argument("--clay-metadata", default=MODEL_METADATA_PATH)
    parser.add_argument("--out-embeddings", default=None,
                        help="default: <results root>/checks/embeddings_<input folder name>.npz "
                             "(standalone runs are treated as checks; the pipeline writes to results/embeddings)")
    parser.add_argument("--out-indices", default=None, help="default: <results root>/checks/indices_<input folder name>.csv")
    parser.add_argument("--cache", action="store_true",
                        help="also write the pixel cache to <results root>/checks/ (station/year parsed from the folder name)")
    parser.add_argument("--cache-pixels", type=int, default=256)
    parser.add_argument("--cache-window", type=int, default=3)
    parser.add_argument("--results-root", default=None, help="default: <project>/results or $FFB_RESULTS_ROOT")
    args = parser.parse_args()

    from project_paths import Layout, ensure_parent
    from pixel_cache import cache_paths
    tag = os.path.basename(os.path.normpath(args.input_dir))
    checks = Layout(args.results_root).checks
    cache_out = st = yr = None
    if args.cache or args.encoder == "none":
        m = re.match(r"^s[12]_(.+)_(\d{4})$", tag)
        st, yr = (m.group(1), int(m.group(2))) if m else (tag, None)
        cache_out = cache_paths(checks, st, args.sensor, yr)
        ensure_parent(cache_out[0])
    if args.encoder != "none":
        args.out_embeddings = ensure_parent(args.out_embeddings or os.path.join(checks, f"embeddings_{tag}.npz"))
        args.out_indices = ensure_parent(args.out_indices or os.path.join(checks, f"indices_{tag}.csv"))

    encoder = build_encoder(args.encoder, args.sensor, args.clay_checkpoint, args.clay_metadata)
    run_extraction(args.sensor, args.input_dir, args.shapefile, args.subcatchment_id, encoder,
                    args.out_embeddings, args.out_indices, cache_out=cache_out, cache_station=st,
                    cache_year=yr, cache_pixels=args.cache_pixels, cache_window=args.cache_window)


if __name__ == "__main__":
    main()
