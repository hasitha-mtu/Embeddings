r"""
test_ee_access.py

Standalone diagnostic for Earth Engine setup -- run this BEFORE
acquire_sentinel_imagery.py, so authentication/permission problems surface
in seconds rather than partway through a real batch job.

Runs a sequence of independent checks and reports pass/fail on each, rather
than stopping at the first failure, so you can see everything that needs
fixing in one go.

Checks performed:
  1. Required packages importable (earthengine-api, geopandas)
  2. Earth Engine initializes with the given credentials
  3. Basic API access (fetch metadata for a public test image)
  4. Sentinel-2 collection access + query over a small test AOI
  5. Sentinel-1 collection access + query over the same AOI
  6. (Optional, --catchment-list/--shapefile) build a real AOI from your
     actual shortlist + shapefile, same code path acquire_sentinel_imagery.py uses
  7. (Optional, --run-export-test) submit one tiny real export task, so you
     can confirm it actually lands in your Drive folder -- this is the only
     way to catch the service-account-Drive-sharing gotcha before it costs
     you a failed multi-hour batch

Usage (interactive auth):
    python test_ee_access.py --ee-project YOUR_GCP_PROJECT_ID

Usage (service account):
    python test_ee_access.py --ee-project YOUR_GCP_PROJECT_ID \
        --service-account-email gee-automation@your-project-id.iam.gserviceaccount.com \
        --key-file /path/to/secret-key.json

Usage (also test a real export lands in Drive):
    python test_ee_access.py --ee-project YOUR_GCP_PROJECT_ID \
        --service-account-email ... --key-file ... \
        --run-export-test --drive-folder sentinel_export_test

Usage (also test AOI construction from your real shortlist + shapefile):
    python test_ee_access.py --ee-project YOUR_GCP_PROJECT_ID \
        --catchment-list catchment_level_shortlist.csv --shapefile subcatchments/subcatchments.shp

Requires: earthengine-api, geopandas
"""

import argparse
import sys

results = []  # (check_name, passed: bool, detail: str)


def check(name):
    """Decorator-ish helper: run fn(), record pass/fail, never let one check kill the others."""
    def wrapper(fn):
        try:
            detail = fn()
            results.append((name, True, detail or "OK"))
            print(f"  [PASS] {name}: {detail or 'OK'}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  [FAIL] {name}: {e}")
    return wrapper


def main():
    parser = argparse.ArgumentParser(description="Diagnose Earth Engine access before running the real pipeline.")
    parser.add_argument("--ee-project", required=True)
    parser.add_argument("--service-account-email", default=None)
    parser.add_argument("--key-file", default=None)
    parser.add_argument("--test-lat", type=float, default=53.4, help="Small test AOI center latitude (default: near Ireland's center)")
    parser.add_argument("--test-lon", type=float, default=-8.0, help="Small test AOI center longitude")
    parser.add_argument("--catchment-list", default=None, help="Optional: test AOI construction from your real shortlist")
    parser.add_argument("--shapefile", default=None, help="Optional: required if --catchment-list is given")
    parser.add_argument("--station-id-filter", default=None, help="Which station's catchment to test (with --catchment-list)")
    parser.add_argument("--run-export-test", action="store_true",
                         help="Submit one tiny real export task, to verify it actually lands where expected")
    parser.add_argument("--drive-folder", default="ee_access_test", help="Drive folder for the export test")
    args = parser.parse_args()

    print("=== 1. Package imports ===")

    @check("earthengine-api import")
    def _():
        global ee
        import ee
        return f"version {ee.__version__ if hasattr(ee, '__version__') else 'unknown'}"

    @check("geopandas import")
    def _():
        global gpd
        import geopandas as gpd
        return f"version {gpd.__version__}"

    if not results[0][1]:
        print("\nCan't proceed without earthengine-api. Run: pip install earthengine-api")
        sys.exit(1)

    print("\n=== 2. Earth Engine initialization ===")

    @check("ee.Initialize")
    def _():
        if args.service_account_email and args.key_file:
            credentials = ee.ServiceAccountCredentials(args.service_account_email, args.key_file)
            ee.Initialize(credentials, project=args.ee_project)
            return f"authenticated as service account {args.service_account_email}"
        else:
            ee.Initialize(project=args.ee_project)
            return "authenticated via interactive user credentials"

    if not results[-1][1]:
        print("\nCan't proceed without a working Earth Engine session. Common causes:")
        print("  - Project not registered for Earth Engine access (console.cloud.google.com)")
        print("  - Service account not granted Earth Engine access")
        print("  - Wrong/expired key file path, or interactive auth never run (earthengine authenticate)")
        sys.exit(1)

    print("\n=== 3. Basic API access ===")

    @check("fetch public image metadata")
    def _():
        info = ee.Image("USGS/SRTMGL1_003").getInfo()
        return f"got metadata, {len(info.get('bands', []))} band(s)"

    print("\n=== 4. Sentinel-2 collection access ===")
    test_aoi = ee.Geometry.Point([args.test_lon, args.test_lat]).buffer(2000).bounds()

    @check("query Sentinel-2 SR over test AOI")
    def _():
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(test_aoi)
            .filterDate("2023-01-01", "2023-02-01")
        )
        n = coll.size().getInfo()
        if n == 0:
            raise RuntimeError("query succeeded but returned 0 scenes -- check test AOI coordinates")
        return f"{n} scene(s) found in test window"

    print("\n=== 5. Sentinel-1 collection access ===")

    @check("query Sentinel-1 GRD over test AOI")
    def _():
        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(test_aoi)
            .filterDate("2023-01-01", "2023-02-01")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )
        n = coll.size().getInfo()
        if n == 0:
            raise RuntimeError("query succeeded but returned 0 scenes -- check test AOI coordinates")
        return f"{n} scene(s) found in test window"

    if args.catchment_list:
        print("\n=== 6. Real AOI construction from your shortlist + shapefile ===")

        @check("build AOI from catchment_level_shortlist.csv + shapefile")
        def _():
            if not args.shapefile:
                raise RuntimeError("--shapefile is required alongside --catchment-list")
            import pandas as pd
            catchments = pd.read_csv(args.catchment_list)
            if args.station_id_filter:
                catchments = catchments[catchments["station_id"].astype(str) == args.station_id_filter]
            if catchments.empty:
                raise RuntimeError("no matching row in catchment list (check --station-id-filter)")
            row = catchments.iloc[0]
            gdf = gpd.read_file(args.shapefile)
            match = gdf[gdf["Subcatchme"] == row["subcatchment_id"]]
            if match.empty:
                raise RuntimeError(f"subcatchment_id {row['subcatchment_id']} not found in shapefile")
            geom_wgs84 = match.to_crs("EPSG:4326").geometry.iloc[0]
            aoi = ee.Geometry(geom_wgs84.__geo_interface__)
            area_km2 = aoi.area().divide(1e6).getInfo()
            return f"station {row['station_id']}, subcatchment {row['subcatchment_id']}, area ~{area_km2:.1f} km2"

    if args.run_export_test:
        print("\n=== 7. Real export test (submits an actual task -- the only way to catch "
              "the service-account/Drive-sharing gotcha before a real batch) ===")

        @check("submit tiny test export task")
        def _():
            small_aoi = ee.Geometry.Point([args.test_lon, args.test_lat]).buffer(200).bounds()
            img = ee.Image("USGS/SRTMGL1_003").clip(small_aoi)
            task = ee.batch.Export.image.toDrive(
                image=img,
                description="ee_access_test",
                folder=args.drive_folder,
                fileNamePrefix="ee_access_test",
                region=small_aoi,
                scale=30,
                maxPixels=1e9,
            )
            task.start()
            return (f"task submitted, id={task.id} -- check your Drive folder "
                    f"'{args.drive_folder}' in a few minutes for 'ee_access_test.tif'. "
                    f"If using a service account: this is where you'll find out whether "
                    f"the folder-sharing step actually worked.")

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed")
    if n_fail > 0:
        print("\nFailed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    else:
        print("\nAll checks passed. Safe to proceed to acquire_sentinel_imagery.py.")
        if not args.run_export_test:
            print("(Note: export/Drive permissions were NOT tested -- re-run with "
                  "--run-export-test --drive-folder <name> to check that too before a real batch.)")


if __name__ == "__main__":
    main()
