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
  - ClayEncoder: checkpoint LOADING is real and verified (Clay v1.5 is
    loaded via `ClayMAEModule.load_from_checkpoint`, per Clay's own repo/
    model card, made-with-clay/Clay on Hugging Face). The forward pass
    (`encode_batch`) -- specifically how the "datacube" dict is assembled
    (pixels + band wavelengths + GSD + time + lat/lon encoding, and which
    output token(s) to pool) -- is BEST EFFORT / UNVERIFIED. I have no
    network access to run Clay's example notebook from this environment.
    Confirm the datacube packaging and output shape against Clay's current
    example notebook (https://github.com/Clay-foundation/model) before
    trusting real numbers out of this path.
  - PrestoEncoder: checkpoint LOADING is real and verified
    (`Presto.load_pretrained()`, bundled weights at
    nasaharvest/presto/data/default_model.pt). Its `encode_batch` is
    DELIBERATELY left unimplemented -- see the class docstring below for
    why this is an architecture mismatch, not just missing code.

Usage (dummy encoder, to test the pipeline):
    python extract_embeddings.py \
        --input-dir ./sentinel_imagery/s2_11001_2023 \
        --shapefile subcatchments/subcatchments.shp --subcatchment-id 11_5 \
        --sensor s2 --encoder dummy \
        --out-embeddings embeddings_11001_s2_2023.npz \
        --out-indices indices_11001_s2_2023.csv

Usage (Clay, once you've verified the forward pass against Clay's docs):
    python extract_embeddings.py ... --encoder clay \
        --clay-checkpoint /path/to/clay-v1.5.ckpt

Requires: numpy, pandas, rasterio, shapely, geopandas
Optional (only if using --encoder clay/presto): torch, and the model's own
package (claymodel / presto). These are imported lazily, inside the encoder
classes, so the dummy-encoder path works even if claymodel/presto aren't
installed.
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
from claymodel.module import ClayMAEModule
from rasterio.features import geometry_mask

MODEL_METADATA_PATH = "C:/Users/AdikariAdikari/PycharmProjects/Embeddings/checkpoints/clay/configs/metadata.yaml"

# NOTE: no module-level `from claymodel.module import ClayMAEModule` here on
# purpose -- that import lives inside ClayEncoder.__init__ so this script
# still runs (e.g. with --encoder dummy) on a machine that doesn't have the
# claymodel package installed. Same reasoning for presto in PrestoEncoder.

# SCL classes to exclude as invalid: 0=no data, 1=saturated, 3=cloud shadow,
# 8/9=cloud medium/high probability, 10=thin cirrus
SCL_INVALID_CLASSES = {0, 1, 3, 8, 9, 10}

S2_BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S1_BAND_ORDER = ["VV", "VH"]

# Sentinel-2 center wavelengths (micrometers) for the bands in S2_BAND_ORDER
# (excluding SCL, which isn't a spectral band). Confirmed values, used by
# ClayEncoder to build the "wavelengths" part of Clay's datacube input.
S2_BAND_WAVELENGTHS_UM = {
    "B02": 0.492, "B03": 0.560, "B04": 0.665, "B05": 0.704, "B06": 0.741,
    "B07": 0.783, "B08": 0.833, "B8A": 0.865, "B11": 1.614, "B12": 2.202,
}
S2_GSD_METERS = 10.0


# --------------------------------------------------------------------------
# Encoders -- pluggable. DummyEncoder is tested; Clay/Presto have verified
# checkpoint loading but differ in forward-pass maturity (see class docs).
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
    """Clay v1.5 patch encoder.

    CHECKPOINT LOADING -- verified against Clay's own repo/model card:
        from claymodel.module import ClayMAEModule
        model = ClayMAEModule.load_from_checkpoint(checkpoint_path)

    FORWARD PASS -- best effort / UNVERIFIED. Clay's encoder expects a
    "datacube" dict roughly like:
        {
          "pixels":     (B, C, H, W) float tensor, normalized per-band,
          "time":       (B, 4) encoded [sin(day), cos(day), sin(year), cos(year)],
          "latlon":     (B, 4) encoded [sin(lat), cos(lat), sin(lon), cos(lon)],
          "waves":      (C,)  band center wavelengths in micrometers,
          "gsd":        scalar, ground sample distance in meters,
        }
    and returns patch/class tokens from `model.model.encoder(datacube)`
    (the exact call and which token(s) to pool -- class token vs mean of
    patch tokens -- needs to be checked against Clay's current example
    notebook: https://github.com/Clay-foundation/model). Below, `time` and
    `latlon` are filled with placeholder zero tensors (since this script
    only has per-tile pixel data, not confirmed date/lat-lon per patch at
    this point in the pipeline) -- replace with real encodings before
    trusting the output. This class will run without crashing but its
    numbers are not yet confirmed correct.
    """

    def __init__(self, checkpoint_path, device="cpu"):
        if not checkpoint_path:
            raise ValueError("ClayEncoder requires --clay-checkpoint")
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
        self.patch_size = 64
        self.model = ClayMAEModule.load_from_checkpoint(checkpoint_path,
                                                        metadata_path=MODEL_METADATA_PATH)
        self.model.eval()
        self.model.to(device)
        # embedding_dim depends on the Clay backbone (ViT) width -- 768 for
        # the standard Clay v1.5 config. Set after first real forward pass
        # if this differs; left as a best-effort default here.
        self.embedding_dim = 768

    def _build_datacube(self, patches, waves_um, gsd_m):
        torch = self.torch
        n = patches.shape[0]
        pixels = torch.from_numpy(np.nan_to_num(patches, nan=0.0)).float().to(self.device)
        waves = torch.tensor(waves_um, dtype=torch.float32, device=self.device)
        # Placeholder time/latlon encodings -- UNVERIFIED, see class docstring.
        time_enc = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        latlon_enc = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        return {
            "pixels": pixels,
            "time": time_enc,
            "latlon": latlon_enc,
            "waves": waves,
            "gsd": torch.tensor(gsd_m, dtype=torch.float32, device=self.device),
        }

    def encode_batch(self, patches, waves_um=None, gsd_m=S2_GSD_METERS):
        if waves_um is None:
            # Assumes S2 band order minus SCL -- pass explicit waves_um for S1.
            waves_um = [S2_BAND_WAVELENGTHS_UM[b] for b in S2_BAND_ORDER if b != "SCL"]
        datacube = self._build_datacube(patches, waves_um, gsd_m)
        with self.torch.no_grad():
            # UNVERIFIED: confirm this call signature and which tokens to
            # pool against Clay's current example notebook before trusting
            # the output.
            out = self.model.model.encoder(datacube)
            if isinstance(out, (tuple, list)):
                out = out[0]
            embedding = out.mean(dim=1) if out.dim() == 3 else out
        return embedding.cpu().numpy()


class PrestoEncoder(BaseEncoder):
    """Presto pixel-timeseries encoder.

    CHECKPOINT LOADING -- verified against nasaharvest/presto:
        from presto import Presto
        encoder_decoder = Presto.load_pretrained()   # loads the bundled
                                                       # data/default_model.pt

    ARCHITECTURAL MISMATCH (not just an unverified detail): Presto is built
    to encode a per-pixel-location TIME SERIES (a sequence of observations
    across dates for one point), not a single-date spatial tile the way
    this script's `process_file()` works (one GeoTIFF = one date = a grid
    of patches). Feeding Presto a one-date spatial patch would require
    either (a) faking a length-1 timeseries per pixel/patch, which throws
    away exactly the temporal signal Presto is designed around, or (b)
    restructuring this script so it assembles, per catchment location, the
    sequence of values across all available dates before calling Presto
    once per location -- a real pipeline change, not a bug fix.

    Two ways to resolve this, left as a design decision rather than
    guessed at here:
      1. Restructure extraction to build per-pixel (or per-small-patch)
         time series across the full date range for a catchment, then call
         Presto once per series. Bigger change, but uses Presto as intended.
      2. Use Clay only for now (Clay is a single-timestep spatial encoder,
         which matches this script's current per-date-tile design), and
         revisit Presto once/if the pipeline is restructured to be
         timeseries-first.

    `encode_batch` is deliberately left raising NotImplementedError below
    until that decision is made -- it should not silently return wrong
    numbers for a shape it was never designed to take.
    """

    def __init__(self, checkpoint_path=None, device="cpu"):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise RuntimeError("PrestoEncoder requires torch -- pip install torch")
        try:
            from presto import Presto
        except ImportError:
            raise RuntimeError(
                "PrestoEncoder requires the presto package -- "
                "pip install git+https://github.com/nasaharvest/presto.git"
            )

        self.device = device
        self.patch_size = 64
        # Verified: loads Presto's bundled pretrained weights.
        self.model = Presto.load_pretrained()
        self.model.eval()
        self.model.to(device)
        self.embedding_dim = None  # depends on how encode_batch ends up structured

    def encode_batch(self, patches):
        raise NotImplementedError(
            "PrestoEncoder.encode_batch is intentionally unimplemented -- "
            "see the PrestoEncoder class docstring: this script processes "
            "one date's spatial tile at a time, but Presto expects a "
            "per-location time series across dates. Restructure the "
            "pipeline to assemble per-location series before calling "
            "Presto, or use --encoder clay for now."
        )


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
        dim = encoder.embedding_dim if encoder.embedding_dim else 1
        embedding = np.full(dim, np.nan, dtype="float32")
    else:
        patch_embeddings = encoder.encode_batch(patches)
        embedding = np.nanmean(patch_embeddings, axis=0)

    indices["n_patches_used"] = n_patches
    return embedding, indices


def build_encoder(encoder_name, sensor, clay_checkpoint=None, presto_checkpoint=None):
    n_bands = len(S2_BAND_ORDER) - 1 if sensor == "s2" else len(S1_BAND_ORDER)
    if encoder_name == "dummy":
        return DummyEncoder(n_bands)
    elif encoder_name == "clay":
        return ClayEncoder(clay_checkpoint)
    elif encoder_name == "presto":
        return PrestoEncoder(presto_checkpoint)
    raise ValueError(f"Unknown encoder: {encoder_name}")


def run_extraction(sensor, input_dir, shapefile, subcatchment_id, encoder,
                    out_embeddings, out_indices):
    """Shared implementation used by both the CLI (main()) and the
    importable extract_embeddings() wrapper below. `encoder` here must
    already be an encoder INSTANCE (DummyEncoder/ClayEncoder/PrestoEncoder),
    not a string -- build it with build_encoder() first.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not files:
        print(f"No .tif files found in {input_dir}", file=sys.stderr)
        return {"n_dates_processed": 0}

    embeddings = {}
    rows = []
    for fp in files:
        date = parse_date_from_filename(os.path.basename(fp))
        if date is None:
            print(f"  could not parse date from {fp}, skipped")
            continue
        try:
            embedding, indices = process_file(fp, shapefile, subcatchment_id, sensor, encoder)
            embeddings[date] = embedding
            rows.append({"date": date, **indices})
            print(f"  {date}: {indices['n_patches_used']} patches used")
        except Exception as e:
            print(f"  {date}: ERROR -- {e}")

    np.savez_compressed(out_embeddings, **embeddings)
    pd.DataFrame(rows).to_csv(out_indices, index=False)
    print(f"\n{len(embeddings)} dates processed.")
    print(f"Embeddings written to: {out_embeddings}")
    print(f"Hand-crafted indices written to: {out_indices}")
    return {"n_dates_processed": len(embeddings)}


def extract_embeddings(sensor, input_dir, shapefile, subcatchment_id, encoder,
                        out_embeddings, out_indices,
                        clay_checkpoint="checkpoints/clay/clay-v1.5.ckpt",
                       presto_checkpoint="checkpoints/presto/presto-main/data/default_model.pt"):
    """Importable entry point (e.g. for run_catchment_pipeline.py) mirroring
    the CLI. `encoder` here is a STRING ("dummy"/"clay"/"presto"), matching
    the --encoder CLI flag -- this function builds the actual encoder
    instance for you, same as main() does.
    """
    encoder_instance = build_encoder(encoder, sensor, clay_checkpoint, presto_checkpoint)
    return run_extraction(sensor, input_dir, shapefile, subcatchment_id,
                           encoder_instance, out_embeddings, out_indices)


def main():
    parser = argparse.ArgumentParser(description="Extract pooled catchment embeddings from downloaded imagery.")
    parser.add_argument("--input-dir", required=True, help="Folder of downloaded GeoTIFFs for one catchment+sensor")
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--subcatchment-id", required=True)
    parser.add_argument("--sensor", required=True, choices=["s1", "s2"])
    parser.add_argument("--encoder", default="dummy", choices=["dummy", "clay", "presto"])
    parser.add_argument("--clay-checkpoint", default="checkpoints/clay/clay-v1.5.ckpt")
    parser.add_argument("--presto-checkpoint", default="checkpoints/presto/presto-main/data/default_model.pt")
    parser.add_argument("--out-embeddings", required=True)
    parser.add_argument("--out-indices", required=True)
    args = parser.parse_args()

    encoder = build_encoder(args.encoder, args.sensor, args.clay_checkpoint, args.presto_checkpoint)
    run_extraction(args.sensor, args.input_dir, args.shapefile, args.subcatchment_id,
                    encoder, args.out_embeddings, args.out_indices)


if __name__ == "__main__":
    main()
