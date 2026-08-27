r"""
acquire_sentinel_imagery.py  (Component 4: satellite data acquisition)

Submits Google Earth Engine batch export tasks for Sentinel-1 (GRD) and
Sentinel-2 (L2A) imagery, clipped to each catchment in your finalized
shortlist, over the Sentinel era (2017-present). Each qualifying scene is
exported as a separate clipped GeoTIFF to Google Drive.

IMPORTANT -- ONE-TIME SETUP (required before this will run):
  1. pip install earthengine-api geopandas
  2. Sign up for Google Earth Engine access (free, non-commercial):
     https://earthengine.google.com/signup/
  3. Create/select a Google Cloud project and enable the Earth Engine API
     for it: https://console.cloud.google.com/
  4a. EITHER authenticate interactively once from the command line:
         earthengine authenticate
      (opens a browser login and stores a local credential -- default if
      --service-account-email / --key-file are not passed)
  4b. OR use a service account (better for unattended/scheduled runs):
      create a service account + JSON key in the Cloud project, grant it
      Earth Engine access, then pass --service-account-email and --key-file
      below. IMPORTANT: if exporting to Drive under a service account, the
      exports land in THAT SERVICE ACCOUNT'S Drive, not yours -- share your
      destination Drive folder with the service account's email address
      (Editor access) first, or you won't be able to see the exported files.

WHAT THIS SCRIPT DOES:
  - Loads your catchment shortlist + subcatchment shapefile, builds an
    Earth Engine geometry (AOI) per catchment from the matched polygon.
  - Queries Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) and Sentinel-1 GRD
    (COPERNICUS/S1_GRD, IW mode, VV+VH) image collections per catchment,
    filtered by date range and (for S2) a cloud-cover pre-filter.
  - Submits one export task per (catchment, sensor, scene date), clipped to
    that catchment's polygon, to a Google Drive folder.
  - Writes a resumable manifest CSV (catchment, sensor, date, task_id,
    status) -- re-running the same command skips (catchment, sensor, date)
    combinations already submitted.

WHAT THIS SCRIPT DOES NOT DO:
  - It does not wait for exports to finish -- GEE exports run asynchronously
    in the cloud and can take anywhere from minutes to many hours depending
    on queue load. Use check_export_status.py (companion script) to poll
    and update the manifest with completion status.
  - It has not been run against live GEE from this environment (no network
    access to Earth Engine here) -- please do a small dry run first (see
    below) before submitting the full 36-catchment job.

RECOMMENDED FIRST RUN (dry run on one catchment, short date range):
    python acquire_sentinel_imagery.py \
        --catchment-list catchment_level_shortlist.csv \
        --shapefile subcatchments/subcatchments.shp \
        --ee-project YOUR_GCP_PROJECT_ID \
        --drive-folder sentinel_export_test \
        --start-date 2023-01-01 --end-date 2023-02-01 \
        --station-id-filter 11001 \
        --out-manifest manifest_test.csv

FULL RUN:
    python acquire_sentinel_imagery.py \
        --catchment-list catchment_level_shortlist.csv \
        --shapefile subcatchments/subcatchments.shp \
        --ee-project YOUR_GCP_PROJECT_ID \
        --drive-folder sentinel_export \
        --start-date 2017-01-01 \
        --out-manifest sentinel_export_manifest.csv
"""

import argparse
import os
import sys
import time

import ee
import geopandas as gpd
import pandas as pd

# Bands kept per sensor. S2: 10 m/20 m surface-reflectance bands commonly
# used by geospatial foundation models (Clay/Presto), skipping coarse
# atmospheric-only bands (B1, B9, B10). SCL is included for later per-pixel
# cloud/shadow masking during Component 5/6 pooling, not for the model input
# itself.
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
S1_BANDS = ["VV", "VH"]

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"


def build_aoi(shapefile_path, subcatchment_id):
    gdf = gpd.read_file(shapefile_path)
    match = gdf[gdf["Subcatchme"] == subcatchment_id]
    if match.empty:
        return None
    geom_wgs84 = match.to_crs("EPSG:4326").geometry.iloc[0]
    # Earth Engine wants plain GeoJSON-style coordinates
    return ee.Geometry(geom_wgs84.__geo_interface__)


def list_s2_scenes(aoi, start_date, end_date, max_cloud_pct):
    coll = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )
    ids = coll.aggregate_array("system:index").getInfo()
    dates = coll.aggregate_array("system:time_start").getInfo()
    return list(zip(ids, dates))


def list_s1_scenes(aoi, start_date, end_date):
    coll = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )
    ids = coll.aggregate_array("system:index").getInfo()
    dates = coll.aggregate_array("system:time_start").getInfo()
    return list(zip(ids, dates))


def submit_export(image_id, collection, bands, aoi, drive_folder, description, scale):
    img = ee.Image(f"{collection}/{image_id}").select(bands).clip(aoi)
    task = ee.batch.Export.image.toDrive(
        image=img,
        description=description[:100],  # GEE description length limit
        folder=drive_folder,
        fileNamePrefix=description[:100],
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e13,
    )
    task.start()
    return task.id


def load_done_keys(manifest_path):
    if not os.path.exists(manifest_path) or os.path.getsize(manifest_path) == 0:
        return set()
    try:
        prior = pd.read_csv(manifest_path, usecols=["catchment_id", "sensor", "scene_date"])
    except Exception:
        return set()
    return set(zip(prior["catchment_id"].astype(str), prior["sensor"], prior["scene_date"].astype(str)))


def main():
    parser = argparse.ArgumentParser(description="Submit GEE Sentinel-1/2 export tasks per catchment.")
    parser.add_argument("--catchment-list", required=True, help="e.g. catchment_level_shortlist.csv")
    parser.add_argument("--shapefile", required=True, help="Path to subcatchments .shp")
    parser.add_argument("--ee-project", required=True, help="Your Google Cloud project ID with EE API enabled")
    parser.add_argument("--service-account-email", default=None,
                         help="Optional: service account email for unattended auth, "
                              "e.g. gee-automation@your-project-id.iam.gserviceaccount.com")
    parser.add_argument("--key-file", default=None, help="Path to the service account's JSON key file")
    parser.add_argument("--drive-folder", required=True, help="Google Drive folder name exports will land in")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default=None, help="Default: today")
    parser.add_argument("--max-cloud-pct", type=float, default=60.0,
                         help="Sentinel-2 scene-level cloud filter (metadata pre-filter, not per-pixel)")
    parser.add_argument("--sensors", default="s1,s2", help="Comma-separated: s1,s2")
    parser.add_argument("--station-id-filter", default=None,
                         help="Optional: restrict to one station_id, for a dry run")
    parser.add_argument("--out-manifest", default="sentinel_export_manifest.csv")
    parser.add_argument("--max-tasks-per-run", type=int, default=2000,
                         help="Safety cap on tasks submitted in a single run")
    args = parser.parse_args()

    end_date = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    sensors = [s.strip().lower() for s in args.sensors.split(",")]

    print("Initializing Earth Engine...")
    if args.service_account_email and args.key_file:
        credentials = ee.ServiceAccountCredentials(args.service_account_email, args.key_file)
        ee.Initialize(credentials, project=args.ee_project)
        print(f"  authenticated as service account: {args.service_account_email}")
    else:
        ee.Initialize(project=args.ee_project)
        print("  authenticated via interactive user credentials")

    catchments = pd.read_csv(args.catchment_list)
    if args.station_id_filter:
        catchments = catchments[catchments["station_id"].astype(str) == args.station_id_filter]
    if "subcatchment_id" not in catchments.columns:
        print("Error: catchment list must contain a 'subcatchment_id' column "
              "(e.g. catchment_level_shortlist.csv from match_stations_to_catchments.py).", file=sys.stderr)
        sys.exit(1)

    done_keys = load_done_keys(args.out_manifest)
    write_header = not os.path.exists(args.out_manifest) or os.path.getsize(args.out_manifest) == 0

    n_submitted = 0
    with open(args.out_manifest, "a", newline="", encoding="utf-8") as out_f:
        if write_header:
            out_f.write("catchment_id,station_id,subcatchment_id,sensor,scene_date,image_id,task_id,status\n")
            out_f.flush()

        for _, row in catchments.iterrows():
            catchment_id = row.get("catchment_id", row.get("subcatchment_id"))
            station_id = row["station_id"]
            subcatchment_id = row["subcatchment_id"]

            aoi = build_aoi(args.shapefile, subcatchment_id)
            if aoi is None:
                print(f"  catchment {catchment_id} (station {station_id}): "
                      f"subcatchment_id {subcatchment_id} not found in shapefile -- skipped")
                continue

            for sensor in sensors:
                if n_submitted >= args.max_tasks_per_run:
                    print(f"\nHit --max-tasks-per-run ({args.max_tasks_per_run}). "
                          f"Re-run the same command to continue submitting the rest.")
                    return

                if sensor == "s2":
                    scenes = list_s2_scenes(aoi, args.start_date, end_date, args.max_cloud_pct)
                    bands, scale, collection = S2_BANDS, 10, S2_COLLECTION
                elif sensor == "s1":
                    scenes = list_s1_scenes(aoi, args.start_date, end_date)
                    bands, scale, collection = S1_BANDS, 10, S1_COLLECTION
                else:
                    print(f"Unknown sensor '{sensor}', skipping.")
                    continue

                print(f"  catchment {catchment_id} (station {station_id}), {sensor}: {len(scenes)} scenes found")

                for image_id, time_start_ms in scenes:
                    scene_date = pd.to_datetime(time_start_ms, unit="ms", utc=True).strftime("%Y-%m-%d")
                    key = (str(catchment_id), sensor, scene_date)
                    if key in done_keys:
                        continue

                    description = f"{sensor}_{station_id}_{scene_date}".replace("-", "")
                    try:
                        task_id = submit_export(image_id, collection, bands, aoi, args.drive_folder, description, scale)
                        status = "SUBMITTED"
                    except Exception as e:
                        task_id = ""
                        status = f"SUBMIT_ERROR: {e}"

                    out_f.write(f"{catchment_id},{station_id},{subcatchment_id},{sensor},{scene_date},"
                                f"{image_id},{task_id},{status}\n")
                    out_f.flush()
                    done_keys.add(key)
                    n_submitted += 1

                    # Light throttling -- avoid hammering the task-submission API
                    if n_submitted % 20 == 0:
                        time.sleep(1)

    print(f"\nDone. {n_submitted} export tasks submitted this run.")
    print(f"Manifest written to: {args.out_manifest}")
    print("Use check_export_status.py to poll these tasks and update completion status.")


if __name__ == "__main__":
    main()
