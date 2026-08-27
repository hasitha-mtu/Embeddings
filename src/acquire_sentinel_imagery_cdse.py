r"""
acquire_sentinel_imagery_cdse.py

Alternative to acquire_sentinel_imagery.py (GEE version) -- acquires
Sentinel-1/2 imagery via the Copernicus Data Space Ecosystem's openEO API
instead of Google Earth Engine. No Google account dependency; downloads land
directly on your local disk (no Drive/GCS intermediary).

IMPORTANT -- ONE-TIME SETUP:
  1. pip install openeo geopandas
  2. Create a free CDSE account: https://dataspace.copernicus.eu/
  3a. EITHER authenticate interactively (default -- opens a browser login
      the first time, then caches the token):
          no extra setup needed, just run the script
  3b. OR for unattended/scheduled runs, register an OAuth client:
          Sentinel Hub dashboard (dataspace.copernicus.eu) -> User Settings
          -> OAuth clients -> create one, note the client ID + secret, and
          pass --client-id / --client-secret below.

CAVEAT: like the GEE scripts, this has NOT been run against the live CDSE
API from this environment (no network access to dataspace.copernicus.eu
here). Collection IDs (SENTINEL2_L2A, SENTINEL1_GRD) and exact batch-job
output behaviour should be confirmed with a small dry run before scaling up
-- run with --list-collections first to sanity-check names against the live
backend before submitting real jobs.

WHAT THIS SCRIPT DOES:
  - Submits one openEO BATCH JOB per (catchment, sensor, year) -- chunking
    by year keeps individual jobs a manageable size rather than one huge
    multi-year job. Each job covers that catchment's polygon, that sensor's
    bands, and applies a basic cloud/quality filter for Sentinel-2.
  - Batch jobs run asynchronously on CDSE's cloud infrastructure, same as
    GEE exports -- this script only SUBMITS them and records job IDs in a
    resumable manifest. Use download_cdse_results.py to poll and download
    finished jobs.

Usage (interactive auth, one catchment, one year -- recommended first run):
    python acquire_sentinel_imagery_cdse.py \
        --catchment-list catchment_level_shortlist.csv \
        --shapefile subcatchments/subcatchments.shp \
        --station-id-filter 11001 --year 2023 \
        --out-manifest cdse_manifest_test.csv

Usage (service-account-style auth via OAuth client credentials):
    python acquire_sentinel_imagery_cdse.py \
        --catchment-list catchment_level_shortlist.csv \
        --shapefile subcatchments/subcatchments.shp \
        --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET \
        --start-year 2017 --end-year 2026 \
        --out-manifest cdse_manifest.csv
"""

import argparse
import os
import sys

import geopandas as gpd
import openeo
import pandas as pd

CDSE_BACKEND = "openeo.dataspace.copernicus.eu"

S2_COLLECTION = "SENTINEL2_L2A"
S1_COLLECTION = "SENTINEL1_GRD"
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL"]
S1_BANDS = ["VV", "VH"]


def connect(client_id=None, client_secret=None):
    conn = openeo.connect(CDSE_BACKEND)
    if client_id and client_secret:
        conn.authenticate_oidc_client_credentials(client_id=client_id, client_secret=client_secret)
    else:
        conn.authenticate_oidc()  # interactive: prints a URL, opens browser on first run
    return conn


def build_bbox(shapefile_path, subcatchment_id):
    gdf = gpd.read_file(shapefile_path)
    match = gdf[gdf["Subcatchme"] == subcatchment_id]
    if match.empty:
        return None
    geom_wgs84 = match.to_crs("EPSG:4326").geometry.iloc[0]
    minx, miny, maxx, maxy = geom_wgs84.bounds
    return {"west": minx, "south": miny, "east": maxx, "north": maxy}


def build_geometry(shapefile_path, subcatchment_id):
    """Returns the catchment's actual (irregular) polygon as a GeoJSON
    FeatureCollection, in EPSG:4326 as openEO's spatial filtering processes
    expect. Used with filter_spatial() to mask the cube down to the real
    catchment shape rather than its bounding-box rectangle -- for an
    elongated catchment, the bbox alone can contain substantially more area
    (and therefore downloaded data) than the catchment itself.

    Wrapped as a Feature inside a FeatureCollection rather than passed as a
    bare geometry dict, and not wrapped in a Python list at the call site --
    filter_spatial expects a single GeoJSON object (a Feature/FeatureCollection
    is the most broadly-supported shape across openEO backends), not a list
    containing one.
    """
    gdf = gpd.read_file(shapefile_path)
    match = gdf[gdf["Subcatchme"] == subcatchment_id]
    if match.empty:
        return None
    geom_wgs84 = match.to_crs("EPSG:4326").geometry.iloc[0]
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": geom_wgs84.__geo_interface__}
        ],
    }


def submit_job(conn, collection, bands, bbox, geometry, start_date, end_date, title, sensor, max_cloud_pct=None):
    cube = conn.load_collection(
        collection,
        spatial_extent=bbox,  # cheap first-pass bound; exact shape applied below
        temporal_extent=[start_date, end_date],
        bands=bands,
    )
    # Mask down to the catchment's real (irregular) polygon rather than the
    # bounding-box rectangle -- pixels outside the polygon become nodata,
    # which compresses far better than real, varied land-cover data does.
    cube = cube.filter_spatial(geometry)

    format_options = {
        # CDSE's GTiff writer has no datatype/dtype-cast option (confirmed via
        # --list-file-formats) -- S1 stays float64 regardless, that's a hard
        # limit of this backend, not something worth chasing further. Instead,
        # tune compression: zstd at a high level, plus GDAL's predictor, which
        # meaningfully improves compression of continuous data like this.
        # Server-side batch-job compute, not your local machine, pays the
        # extra compression-time cost, so there's little downside to going high.
        "compression": "zstd",
        "ZLEVEL": 19,
        # predictor=3 (floating-point prediction) suits S1's float64 output;
        # predictor=2 (integer horizontal differencing) suits S2's int16 output.
        "predictor": 3 if sensor == "s1" else 2,
    }

    result = cube.save_result(format="GTiff", options=format_options or None)
    job = result.create_job(title=title)
    job.start_job()
    return job.job_id


def load_done_keys(manifest_path):
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return set()
    try:
        prior = pd.read_csv(manifest_path)
    except Exception:
        return set()
    return set(zip(prior["catchment_id"].astype(str), prior["sensor"], prior["year"].astype(str)))


def main():
    parser = argparse.ArgumentParser(description="Submit CDSE openEO batch jobs for Sentinel-1/2 per catchment.")
    parser.add_argument("--catchment-list", required=True)
    parser.add_argument("--shapefile", required=True)
    parser.add_argument("--client-id", default=None, help="OAuth client ID, for unattended auth")
    parser.add_argument("--client-secret", default=None, help="OAuth client secret, for unattended auth")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=pd.Timestamp.today().year)
    parser.add_argument("--year", type=int, default=None, help="Shortcut: only this one year (overrides start/end)")
    parser.add_argument("--sensors", default="s1,s2")
    parser.add_argument("--max-cloud-pct", type=float, default=60.0)
    parser.add_argument("--station-id-filter", default=None)
    parser.add_argument("--out-manifest", default="cdse_export_manifest.csv")
    parser.add_argument("--list-collections", action="store_true",
                         help="Just connect, print available collection IDs, and exit -- run this first")
    parser.add_argument("--list-file-formats", action="store_true",
                         help="Connect, print the GTiff writer's actual supported output "
                              "options (e.g. the correct datatype/compression keys), and exit")
    args = parser.parse_args()

    print("Connecting to CDSE openEO backend...")
    conn = connect(args.client_id, args.client_secret)
    print("  connected.")

    if args.list_collections:
        collections = [c["id"] for c in conn.list_collections()]
        s2_matches = [c for c in collections if "SENTINEL2" in c.upper()]
        s1_matches = [c for c in collections if "SENTINEL1" in c.upper()]
        print(f"\nSentinel-2-related collections found: {s2_matches}")
        print(f"Sentinel-1-related collections found: {s1_matches}")
        print(f"\n(Script currently assumes '{S2_COLLECTION}' and '{S1_COLLECTION}' -- "
              f"update the constants at the top of the script if these don't match.)")
        return

    if args.list_file_formats:
        formats = conn.list_file_formats()
        gtiff = formats.get("output", {}).get("GTiff") or formats.get("output", {}).get("GTIFF")
        print("\n=== GTiff output format options actually supported by this backend ===")
        if gtiff:
            for opt_name, opt_info in gtiff.get("parameters", {}).items():
                print(f"  {opt_name}: {opt_info}")
        else:
            print("  Could not find a 'GTiff' entry -- full raw response below:")
            import json
            print(json.dumps(formats.get("output", {}), indent=2))
        print("\n(Update the format_options dict in submit_job() to use the correct key name(s) "
              "shown above -- 'datatype' was a guess and evidently isn't the right key.)")
        return

    years = [args.year] if args.year else list(range(args.start_year, args.end_year + 1))
    sensors = [s.strip().lower() for s in args.sensors.split(",")]

    catchments = pd.read_csv(args.catchment_list)
    if args.station_id_filter:
        catchments = catchments[catchments["station_id"].astype(str) == args.station_id_filter]
    if "subcatchment_id" not in catchments.columns:
        print("Error: catchment list must contain a 'subcatchment_id' column.", file=sys.stderr)
        sys.exit(1)

    done_keys = load_done_keys(args.out_manifest)
    write_header = not os.path.exists(args.out_manifest) or os.path.getsize(args.out_manifest) == 0

    n_submitted = 0
    with open(args.out_manifest, "a", newline="", encoding="utf-8") as out_f:
        if write_header:
            out_f.write("catchment_id,station_id,subcatchment_id,sensor,year,job_id,status\n")
            out_f.flush()

        for _, row in catchments.iterrows():
            catchment_id = row.get("catchment_id", row.get("subcatchment_id"))
            station_id = row["station_id"]
            subcatchment_id = row["subcatchment_id"]

            bbox = build_bbox(args.shapefile, subcatchment_id)
            geometry = build_geometry(args.shapefile, subcatchment_id)
            if bbox is None or geometry is None:
                print(f"  catchment {catchment_id}: subcatchment_id {subcatchment_id} not found -- skipped")
                continue

            for sensor in sensors:
                collection = S2_COLLECTION if sensor == "s2" else S1_COLLECTION
                bands = S2_BANDS if sensor == "s2" else S1_BANDS

                for year in years:
                    key = (str(catchment_id), sensor, str(year))
                    if key in done_keys:
                        continue

                    start_date, end_date = f"{year}-01-01", f"{year + 1}-01-01"
                    title = f"{sensor}_{station_id}_{year}"
                    try:
                        job_id = submit_job(conn, collection, bands, bbox, geometry, start_date, end_date,
                                             title, sensor, args.max_cloud_pct if sensor == "s2" else None)
                        status = "SUBMITTED"
                    except Exception as e:
                        job_id, status = "", f"SUBMIT_ERROR: {e}"

                    out_f.write(f"{catchment_id},{station_id},{subcatchment_id},{sensor},{year},{job_id},{status}\n")
                    out_f.flush()
                    done_keys.add(key)
                    n_submitted += 1
                    print(f"  catchment {catchment_id} station {station_id} {sensor} {year}: {status}")

    print(f"\nDone. {n_submitted} jobs submitted this run.")
    print(f"Manifest written to: {args.out_manifest}")
    print("Use download_cdse_results.py to poll and download finished jobs.")


if __name__ == "__main__":
    main()
