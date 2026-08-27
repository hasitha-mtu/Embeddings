r"""
download_cdse_results.py  (companion to acquire_sentinel_imagery_cdse.py)

Polls the CDSE openEO backend for the status of every job in the manifest,
and downloads the results of any job that has finished -- directly to a
local output directory, no Drive/GCS intermediary needed.

Usage:
    python download_cdse_results.py \
        --manifest cdse_export_manifest.csv \
        --out-dir ./sentinel_imagery \
        --client-id YOUR_CLIENT_ID --client-secret YOUR_CLIENT_SECRET

Run periodically until the summary shows 0 remaining in submitted/running state.

Requires: openeo, pandas
"""

import argparse
import os
import sys

import openeo
import pandas as pd

CDSE_BACKEND = "openeo.dataspace.copernicus.eu"


def connect(client_id=None, client_secret=None):
    conn = openeo.connect(CDSE_BACKEND)
    if client_id and client_secret:
        conn.authenticate_oidc_client_credentials(client_id=client_id, client_secret=client_secret)
    else:
        conn.authenticate_oidc()
    return conn

def download_data(manifest, out_dir, client_id = None, client_secret = None):
    os.makedirs(out_dir, exist_ok=True)
    conn = connect(client_id, client_secret)

    df = pd.read_csv(manifest, dtype=str)
    if "job_id" not in df.columns:
        print("Error: manifest must contain a 'job_id' column.", file=sys.stderr)
        sys.exit(1)

    pending_mask = (
            df["status"].isin(["SUBMITTED", "running", "queued", "created"])
            & df["job_id"].notna() & (df["job_id"] != "")
    )
    pending = df[pending_mask]
    print(f"{len(pending)} jobs to check (out of {len(df)} total manifest rows).")

    for idx, row in pending.iterrows():
        job_id = row["job_id"]
        try:
            job = conn.job(job_id)
            job_status = job.status()
        except Exception as e:
            df.loc[idx, "status"] = f"CHECK_ERROR: {e}"
            continue

        if job_status == "finished":
            dest = os.path.join(
                out_dir, f"{row['sensor']}_{row['station_id']}_{row['year']}"
            )
            os.makedirs(dest, exist_ok=True)
            try:
                results = job.get_results()
                results.download_files(dest)
                df.loc[idx, "status"] = "DOWNLOADED"
                print(f"  station {row['station_id']} {row['sensor']} {row['year']}: downloaded to {dest}")
            except Exception as e:
                df.loc[idx, "status"] = f"DOWNLOAD_ERROR: {e}"
        else:
            df.loc[idx, "status"] = job_status

    df.to_csv(manifest, index=False)

    print("\n=== Current status breakdown ===")
    print(df["status"].value_counts().to_string())
    downloaded = (df["status"] == "DOWNLOADED").sum()
    failed = df["status"].str.contains("error|fail", case=False, na=False).sum()
    still_pending = df["status"].isin(["SUBMITTED", "running", "queued", "created"]).sum()
    print(f"\nDownloaded: {downloaded}  |  Failed/errored: {failed}  |  Still pending: {still_pending}")


def main():
    parser = argparse.ArgumentParser(description="Poll and download finished CDSE openEO jobs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--client-secret", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conn = connect(args.client_id, args.client_secret)

    df = pd.read_csv(args.manifest, dtype=str)
    if "job_id" not in df.columns:
        print("Error: manifest must contain a 'job_id' column.", file=sys.stderr)
        sys.exit(1)

    pending_mask = (
        df["status"].isin(["SUBMITTED", "running", "queued", "created"])
        & df["job_id"].notna() & (df["job_id"] != "")
    )
    pending = df[pending_mask]
    print(f"{len(pending)} jobs to check (out of {len(df)} total manifest rows).")

    for idx, row in pending.iterrows():
        job_id = row["job_id"]
        try:
            job = conn.job(job_id)
            job_status = job.status()
        except Exception as e:
            df.loc[idx, "status"] = f"CHECK_ERROR: {e}"
            continue

        if job_status == "finished":
            dest = os.path.join(
                args.out_dir, f"{row['sensor']}_{row['station_id']}_{row['year']}"
            )
            os.makedirs(dest, exist_ok=True)
            try:
                results = job.get_results()
                results.download_files(dest)
                df.loc[idx, "status"] = "DOWNLOADED"
                print(f"  station {row['station_id']} {row['sensor']} {row['year']}: downloaded to {dest}")
            except Exception as e:
                df.loc[idx, "status"] = f"DOWNLOAD_ERROR: {e}"
        else:
            df.loc[idx, "status"] = job_status

    df.to_csv(args.manifest, index=False)

    print("\n=== Current status breakdown ===")
    print(df["status"].value_counts().to_string())
    downloaded = (df["status"] == "DOWNLOADED").sum()
    failed = df["status"].str.contains("error|fail", case=False, na=False).sum()
    still_pending = df["status"].isin(["SUBMITTED", "running", "queued", "created"]).sum()
    print(f"\nDownloaded: {downloaded}  |  Failed/errored: {failed}  |  Still pending: {still_pending}")


import os
import rasterio

def main1():
    for folder in [r".\output\sentinel_imagery_test4\s1_11001_2023", r".\output\sentinel_imagery_test4\s2_11001_2023"]:
        files = sorted(os.listdir(folder))
        print(f"\n=== {folder} ===")
        print(f"{len(files)} files: {files[:5]}{' ...' if len(files) > 5 else ''}")
        if files:
            first_tif = next((f for f in files if f.lower().endswith((".tif", ".tiff"))), None)
            if first_tif:
                with rasterio.open(os.path.join(folder, first_tif)) as src:
                    print(f"  bands: {src.count}, shape: {src.shape}, dtype: {src.dtypes[0]}")
                    print(f"  CRS: {src.crs}")
                    print(f"  bounds: {src.bounds}")


    folder = r".\output\sentinel_imagery_test4\s2_11001_2023"
    files = [f for f in os.listdir(folder) if f.endswith(".tif")]
    sizes = [os.path.getsize(os.path.join(folder, f)) for f in files]
    print(f"one file: {sizes[0] / 1e6:.1f} MB")
    print(f"total folder ({len(sizes)} files): {sum(sizes) / 1e9:.2f} GB")

    folder = r".\output\sentinel_imagery_test4\s1_11001_2023"
    files = [f for f in os.listdir(folder) if f.endswith(".tif")]
    sizes = [os.path.getsize(os.path.join(folder, f)) for f in files]
    print(f"one file: {sizes[0] / 1e6:.1f} MB, total ({len(sizes)} files): {sum(sizes) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
