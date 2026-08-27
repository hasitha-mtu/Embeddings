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

ENCODER HONESTY NOTE: DummyEncoder (mean+std per band) is the only encoder
tested here -- it validates that the whole pipeline (masking, tiling,
pooling, I/O) works correctly. ClayEncoder/PrestoEncoder below are
best-effort stubs based on how these models are typically loaded, but I have
no network access to huggingface.co or the Clay/Presto repos from this
environment, so their exact current loading API is NOT verified. Confirm
against the model's current documentation/repo before trusting real output
from them, the same way the GEE/CDSE scripts needed real-world verification.

Usage (dummy encoder, to test the pipeline):
    python extract_embeddings.py \
        --input-dir ./sentinel_imagery/s2_11001_2023 \
        --shapefile subcatchments/subcatchments.shp --subcatchment-id 11_5 \
        --sensor s2 --encoder dummy \
        --out-embeddings embeddings_11001_s2_2023.npz \
        --out-indices indices_11001_s2_2023.csv

Usage (Clay, once you've verified the loader against Clay's actual docs):
    python extract_embeddings.py ... --encoder clay --clay-checkpoint /path/to/clay-v1.5.ckpt

Requires: numpy, pandas, rasterio, shapely, geopandas
Optional (only if using --encoder clay/presto): torch, and the model's own package
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

# SCL classes to exclude as invalid: 0=no data, 1=saturated, 3=cloud shadow,
# 8/9=cloud medium/high probability, 10=thin cirrus
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}

S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S1_BAND_ORDER = ["VV", "VH"]


# --------------------------------------------------------------------------
# Encoders -- pluggable. DummyEncoder is tested; Clay/Presto are unverified
# stubs, clearly marked.
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


class ClayEncoder(BaseEncoder):
    """UNVERIFIED STUB. Based on how Clay v1.5 is typically loaded (a
    checkpoint via their `claymodel` package / GitHub repo), but I have no
    network access to confirm the current exact API from this environment.
    Check https://github.com/Clay-foundation/model for the current loading
    code before trusting this."""

    def __init__(self, checkpoint_path, device="cpu"):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError("ClayEncoder requires torch -- pip install torch")
        raise NotImplementedError(
            "ClayEncoder is an unverified stub. Fill in the actual model-loading "
            "code from https://github.com/Clay-foundation/model (current as of "
            "when you read this, not when this script was written), then remove "
            "this raise. Rough shape: load the Clay module + checkpoint, call "
            "its patch encoder on normalized (bands, H, W) tiles, return the "
            "class-token or mean-pooled patch embedding as a fixed-length vector."
        )

    def encode_batch(self, patches):
        raise NotImplementedError


class PrestoEncoder(BaseEncoder):
    """UNVERIFIED STUB -- same caveat as ClayEncoder. See
    https://github.com/nasaharvest/presto for current loading code."""

    def __init__(self, checkpoint_path=None, device="cpu"):
        raise NotImplementedError(
            "PrestoEncoder is an unverified stub -- see nasaharvest/presto "
            "for current model-loading code before using this."
        )

    def encode_batch(self, patches):
        raise NotImplementedError


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
    valid = np.ones(scl_band.shape, dtype=bool)
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
        return np.nanmean(np.where(m, idx, np.nan))

    return {
        "ndvi_mean": safe_index(band["B08"], band["B04"]),
        "ndwi_mean": safe_index(band["B03"], band["B08"]),
        "mndwi_mean": safe_index(band["B03"], band["B11"]),
        "valid_pixel_frac": float(m.sum() / catchment_mask.sum()) if catchment_mask.sum() > 0 else 0.0,
    }


def compute_s1_stats(array, catchment_mask):
    band = {name: array[i] for i, name in enumerate(S1_BAND_ORDER)}
    m = catchment_mask
    if m.sum() == 0:
        return {"vv_mean": np.nan, "vv_std": np.nan, "vh_mean": np.nan, "vh_std": np.nan, "valid_pixel_frac": 0.0}
    return {
        "vv_mean": float(np.nanmean(np.where(m, band["VV"], np.nan))),
        "vv_std": float(np.nanstd(np.where(m, band["VV"], np.nan))),
        "vh_mean": float(np.nanmean(np.where(m, band["VH"], np.nan))),
        "vh_std": float(np.nanstd(np.where(m, band["VH"], np.nan))),
        "valid_pixel_frac": 1.0,  # S1 has no cloud concept; catchment mask only
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

def process_file(filepath, shapefile_path, subcatchment_id, sensor, encoder):
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
        final_mask = catchment_mask

    patches = tile_patches(encode_array, final_mask, encoder.patch_size)
    n_patches = len(patches)
    if n_patches == 0:
        embedding = np.full(encoder.embedding_dim, np.nan, dtype="float32")
    else:
        patch_embeddings = encoder.encode_batch(patches)
        embedding = np.nanmean(patch_embeddings, axis=0)

    indices["n_patches_used"] = n_patches
    return embedding, indices


def main():
    parser = argparse.ArgumentParser(description="Extract pooled catchment embeddings from downloaded imagery.")
    parser.add_argument("--input-dir", required=True, help="Folder of downloaded GeoTIFFs for one catchment+sensor")
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--subcatchment-id", required=True)
    parser.add_argument("--sensor", required=True, choices=["s1", "s2"])
    parser.add_argument("--encoder", default="dummy", choices=["dummy", "clay", "presto"])
    parser.add_argument("--clay-checkpoint", default=None)
    parser.add_argument("--presto-checkpoint", default=None)
    parser.add_argument("--out-embeddings", required=True)
    parser.add_argument("--out-indices", required=True)
    args = parser.parse_args()

    n_bands = len(S2_BAND_ORDER) - 1 if args.sensor == "s2" else len(S1_BAND_ORDER)
    if args.encoder == "dummy":
        encoder = DummyEncoder(n_bands)
    elif args.encoder == "clay":
        encoder = ClayEncoder(args.clay_checkpoint)
    else:
        encoder = PrestoEncoder(args.presto_checkpoint)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.tif")))
    if not files:
        print(f"No .tif files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    embeddings = {}
    rows = []
    for fp in files:
        date = parse_date_from_filename(os.path.basename(fp))
        if date is None:
            print(f"  could not parse date from {fp}, skipped")
            continue
        try:
            embedding, indices = process_file(fp, args.shapefile, args.subcatchment_id, args.sensor, encoder)
            embeddings[date] = embedding
            rows.append({"date": date, **indices})
            print(f"  {date}: {indices['n_patches_used']} patches used")
        except Exception as e:
            print(f"  {date}: ERROR -- {e}")

    np.savez_compressed(args.out_embeddings, **embeddings)
    pd.DataFrame(rows).to_csv(args.out_indices, index=False)
    print(f"\n{len(embeddings)} dates processed.")
    print(f"Embeddings written to: {args.out_embeddings}")
    print(f"Hand-crafted indices written to: {args.out_indices}")


if __name__ == "__main__":
    main()
