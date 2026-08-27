r"""
check_export_status.py  (companion to acquire_sentinel_imagery.py)

Polls Google Earth Engine for the current status of every task in the export
manifest and updates it in place. GEE exports run asynchronously in the
cloud -- this is how you find out when they've actually finished (or failed)
without watching the Earth Engine Tasks web UI by hand.

Usage:
    python check_export_status.py --manifest sentinel_export_manifest.csv --ee-project YOUR_GCP_PROJECT_ID

Run this periodically (e.g. every 30-60 min while a large batch is
processing) until the summary shows 0 remaining in SUBMITTED/RUNNING state.

Requires: earthengine-api, pandas
"""

import argparse
import sys

import ee
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Poll GEE export task status and update the manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ee-project", required=True)
    parser.add_argument("--service-account-email", default=None)
    parser.add_argument("--key-file", default=None)
    args = parser.parse_args()
    ee.Initialize(project='105430693423933239962')
    if args.service_account_email and args.key_file:
        credentials = ee.ServiceAccountCredentials(args.service_account_email, args.key_file)
        ee.Initialize(credentials, project=args.ee_project)
    else:
        ee.Initialize(project=args.ee_project)

    df = pd.read_csv(args.manifest, dtype=str)
    if "task_id" not in df.columns:
        print("Error: manifest must contain a 'task_id' column.", file=sys.stderr)
        sys.exit(1)

    pending_mask = df["status"].isin(["SUBMITTED", "RUNNING", "READY"]) & df["task_id"].notna() & (df["task_id"] != "")
    pending_ids = df.loc[pending_mask, "task_id"].unique().tolist()
    print(f"{len(pending_ids)} tasks to check (out of {len(df)} total manifest rows).")

    updated = 0
    for task_id in pending_ids:
        try:
            info = ee.data.getTaskStatus(task_id)[0]
            new_status = info.get("state", "UNKNOWN")
        except Exception as e:
            new_status = f"CHECK_ERROR: {e}"
        rows = df["task_id"] == task_id
        if (df.loc[rows, "status"] != new_status).any():
            updated += 1
        df.loc[rows, "status"] = new_status

    df.to_csv(args.manifest, index=False)

    print(f"\n{updated} rows changed status this check.")
    print("\n=== Current status breakdown ===")
    print(df["status"].value_counts().to_string())

    completed = (df["status"] == "COMPLETED").sum()
    failed = df["status"].str.contains("FAIL|ERROR", case=False, na=False).sum()
    still_pending = df["status"].isin(["SUBMITTED", "RUNNING", "READY", "UNSUBMITTED"]).sum()
    print(f"\nCompleted: {completed}  |  Failed/errored: {failed}  |  Still pending: {still_pending}")
    if failed > 0:
        print("\nFailed rows (worth inspecting -- e.g. GEE quota limits, invalid geometry, etc.):")
        print(df[df["status"].str.contains("FAIL|ERROR", case=False, na=False)]
              [["catchment_id", "sensor", "scene_date", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
