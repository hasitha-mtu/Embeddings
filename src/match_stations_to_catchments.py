"""
match_stations_to_catchments.py

Joins three sources into one catchment-selection shortlist:

  1. station_availability_summary.csv -- per-station Sentinel-era data
     coverage stats (from check_station_availability.py)
  2. station_overview_table.csv -- OPW station metadata: name, lat/lon,
     OPW's own catchment/RBD labels
  3. a subcatchment polygon shapefile -- used to spatially locate each
     station (via its lat/lon) inside a subcatchment/catchment polygon

Output: one row per station that has both availability data and a spatial
match, with its matched subcatchment/catchment, catchment area, and all
coverage metrics -- plus a second, catchment-level rollup picking the best
candidate station per catchment.

Usage:
    python match_stations_to_catchments.py \
        --availability station_availability_summary.csv \
        --overview station_overview_table.csv \
        --shapefile subcatchments/subcatchments.shp \
        --out-stations catchment_station_shortlist.csv \
        --out-catchments catchment_level_shortlist.csv

Requires: pandas, geopandas, shapely  (pip install pandas geopandas shapely)
"""

import argparse
import re
import sys

import geopandas as gpd
import pandas as pd

OUTPUT_PATH = "C:\\Users\AdikariAdikari\PycharmProjects\Embeddings\dataset\metadata"


def load_overview(path):
    df = pd.read_csv(path)
    # the header row is written as "#Status,Station No.,..." -- strip the leading '#'
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    df = df.rename(columns={
        "Station No.": "station_id",
        "Station": "station_name",
        "Water Body": "water_body",
        "Catchment": "opw_catchment_name",
        "River Basin District": "rbd",
        "Catchment Area": "opw_catchment_area",
        "Latitude": "latitude",
        "Longitude": "longitude",
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Match gauge stations to catchment polygons.")
    parser.add_argument("--availability", required=True, help="station_availability_summary.csv")
    parser.add_argument("--overview", required=True, help="station_overview_table.csv")
    parser.add_argument("--shapefile", required=True, help="Path to the subcatchment .shp file")
    parser.add_argument("--out-stations", default="catchment_station_shortlist.csv")
    parser.add_argument("--out-catchments", default="catchment_level_shortlist.csv")
    parser.add_argument("--min-span-years", type=float, default=5.0)
    parser.add_argument("--min-completeness-pct", type=float, default=70.0)
    parser.add_argument("--max-gap-days", type=float, default=60.0)
    args = parser.parse_args()

    avail = pd.read_csv(args.availability)
    overview = load_overview(args.overview)

    # station_availability_summary.csv carries its own (mostly-blank, WISKI-format-only)
    # station_name/rbd/latitude/longitude columns -- drop them so the overview table's
    # populated versions of these fields survive the merge instead of colliding.
    avail = avail.drop(columns=["station_name", "rbd", "latitude", "longitude"], errors="ignore")

    # --- Merge overview metadata with availability stats ---
    merged = overview.merge(avail, on="station_id", how="left", indicator=True)
    missing_avail = merged[merged["_merge"] == "left_only"]["station_id"].tolist()
    unmatched_from_avail = set(avail["station_id"]) - set(overview["station_id"])
    merged = merged.drop(columns="_merge")

    # --- Spatial join: locate each station inside a subcatchment polygon ---
    gdf_sub = gpd.read_file(args.shapefile)

    has_coords = merged["latitude"].notna() & merged["longitude"].notna()
    stations_gdf = gpd.GeoDataFrame(
        merged[has_coords].copy(),
        geometry=gpd.points_from_xy(merged.loc[has_coords, "longitude"], merged.loc[has_coords, "latitude"]),
        crs="EPSG:4326",
    ).to_crs(gdf_sub.crs)

    joined = gpd.sjoin(
        stations_gdf,
        gdf_sub[["Subcatchme", "Name", "CatchmentI", "Shape_STAr", "geometry"]],
        how="left",
        predicate="intersects",
    )

    # Flag/resolve any station that matched more than one polygon (boundary edge case)
    dupe_counts = joined.groupby(joined.index).size()
    ambiguous_idx = dupe_counts[dupe_counts > 1].index
    joined["ambiguous_match"] = joined.index.isin(ambiguous_idx)
    joined = joined[~joined.index.duplicated(keep="first")]

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = joined.rename(columns={
        "Subcatchme": "subcatchment_id",
        "Name": "subcatchment_name",
        "CatchmentI": "catchment_id",
        "Shape_STAr": "subcatchment_area_m2",
    })
    joined["subcatchment_area_km2"] = (joined["subcatchment_area_m2"] / 1e6).round(2)
    joined["derived_catchment_name"] = joined["subcatchment_name"].str.replace(r"_SC_\d+$", "", regex=True)

    # OPW's own declared catchment area (the true upstream drainage area for that
    # gauge) vs. the area of the single subcatchment polygon it physically sits in.
    # The shapefile has no upstream flow-network topology, so for a gauge far
    # downstream, its real catchment may be many subcatchments combined -- this
    # ratio flags which gauges that applies to, vs. which ones are already a
    # near-1:1 match (small headwater gauges, where the containing subcatchment
    # *is* essentially the whole drainage area -- exactly what this study wants).
    joined["opw_catchment_area_km2"] = (
        joined["opw_catchment_area"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        .replace("", pd.NA).astype(float)
    )
    joined["subcatchment_to_true_area_ratio"] = (
        joined["subcatchment_area_km2"] / joined["opw_catchment_area_km2"]
    ).round(2)
    joined["single_subcatchment_gauge"] = joined["subcatchment_to_true_area_ratio"].between(0.7, 1.3)

    unmatched_spatial = joined[joined["subcatchment_id"].isna()]

    joined["meets_thresholds"] = (
        (joined["sentinel_era_span_days"] >= args.min_span_years * 365)
        & (joined["sentinel_era_completeness_pct"] >= args.min_completeness_pct)
        & (joined["sentinel_era_max_gap_days"] <= args.max_gap_days)
    )

    lead_cols = [
        "station_id", "station_name", "water_body", "opw_catchment_name", "rbd",
        "derived_catchment_name", "catchment_id", "subcatchment_id", "subcatchment_name",
        "subcatchment_area_km2", "opw_catchment_area_km2", "subcatchment_to_true_area_ratio",
        "single_subcatchment_gauge", "ambiguous_match",
        "meets_thresholds", "status", "sentinel_era_span_days",
        "sentinel_era_completeness_pct", "sentinel_era_max_gap_days",
        "sentinel_era_quality_ok_pct", "sentinel_era_interval_minutes",
        "latitude", "longitude",
    ]
    cols = [c for c in lead_cols if c in joined.columns] + [c for c in joined.columns if c not in lead_cols]
    joined = joined[cols].sort_values(["catchment_id", "meets_thresholds"], ascending=[True, False])
    joined.to_csv(f'{OUTPUT_PATH}/{args.out_stations}', index=False)

    # --- Catchment-level rollup: best candidate station per catchment ---
    candidates = joined[joined["meets_thresholds"] & joined["subcatchment_id"].notna()].copy()
    # Prefer single-subcatchment gauges first (cleanest, smallest, most defensible
    # boundary given this shapefile has no upstream flow-network topology), then
    # rank by data quality within that.
    candidates = candidates.sort_values(
        ["catchment_id", "single_subcatchment_gauge", "sentinel_era_completeness_pct", "sentinel_era_max_gap_days"],
        ascending=[True, False, False, True],
    )
    n_candidates_per_catchment = candidates.groupby("catchment_id")["station_id"].transform("count")
    candidates["n_candidate_stations_in_catchment"] = n_candidates_per_catchment
    best_per_catchment = candidates.drop_duplicates(subset="catchment_id", keep="first")
    best_per_catchment.to_csv(f'{OUTPUT_PATH}/{args.out_catchments}', index=False)

    # --- Report ---
    print(f"Overview stations: {len(overview)}")
    print(f"Availability records: {len(avail)}")
    print(f"Stations missing from availability summary (in overview only): {len(missing_avail)}")
    if unmatched_from_avail:
        print(f"Stations in availability summary but not in overview table: {len(unmatched_from_avail)}")
    print(f"Stations with no spatial match to any subcatchment polygon: {len(unmatched_spatial)}")
    print(f"Stations with an ambiguous (multi-polygon) spatial match: {int(joined['ambiguous_match'].sum())}")
    print()
    print(f"Stations meeting availability thresholds AND spatially matched: {len(candidates)}")
    print(f"Distinct catchments covered by at least one qualifying station: {best_per_catchment['catchment_id'].nunique()}")
    print()
    print(f"Station-level shortlist written to: {args.out_stations}")
    print(f"Catchment-level shortlist (best station per catchment) written to: {args.out_catchments}")


if __name__ == "__main__":
    main()
