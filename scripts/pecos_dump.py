#!/usr/bin/env python3
"""
CMS PECOS-style org/location dump built from the public PPEF enrollment assets.

This script:
1. Downloads the same public CMS enrollment, reassignment, and practice-location assets.
2. Converts discovered CSV assets to parquet.
3. Builds a dated `pecos_orgs_YYYYMMDD.parquet` from the organization/location rows
   in the enrollment file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from cms_enrollment_common import (
    PARQUET_DIR,
    build_logger,
    column_as_string,
    download_assets,
    ensure_directories,
    find_latest_parquet,
    new_session,
    warn_if_schema_drift,
)

CMS_ENROLLMENT_FILE_PATTERNS = [r"ENROLL", r"REASSIGN", r"PRACTICE", r"LOCATION"]

logger = build_logger("pecos_org_dump", "pecos_org_dump.log")


def build_orgs_parquet(enrollment_parquet: Path, output_date: str | None = None) -> Path:
    output_date = output_date or datetime.now().strftime("%Y%m%d")
    logger.info("Building organization/location parquet from %s", enrollment_parquet)
    df = pd.read_parquet(enrollment_parquet)

    enrlmt_id = column_as_string(df, "ENRLMT_ID")
    first_name = column_as_string(df, "FIRST_NAME")
    last_name = column_as_string(df, "LAST_NAME")
    org_name = column_as_string(df, "ORG_NAME")
    provider_type_desc = column_as_string(df, "PROVIDER_TYPE_DESC")
    npi = column_as_string(df, "NPI")

    is_org = (
        enrlmt_id.str.startswith("O")
        | (org_name != "")
        | (~provider_type_desc.str.contains("PRACTITIONER", case=False, na=False))
    )
    is_org &= ~((first_name != "") & (last_name != "") & org_name.eq(""))

    orgs = df.loc[is_org].copy()
    orgs["SOURCE_ENROLLMENT_ID"] = enrlmt_id.loc[orgs.index]
    orgs["SOURCE_NPI"] = npi.loc[orgs.index]
    orgs["SOURCE_ORG_NAME"] = org_name.loc[orgs.index]
    orgs["SOURCE_PROVIDER_TYPE_DESC"] = provider_type_desc.loc[orgs.index]

    output_path = PARQUET_DIR / f"pecos_orgs_{output_date}.parquet"
    orgs.to_parquet(output_path, index=False, compression="snappy")
    warn_if_schema_drift(orgs, "pecos_orgs", logger)
    logger.info(
        "Saved org/location parquet: %s (%s rows, %s columns)",
        output_path,
        f"{len(orgs):,}",
        len(orgs.columns),
    )
    return output_path


def find_latest_enrollment_parquet() -> Path | None:
    return find_latest_parquet(
        include_patterns=[r"ENROLL"],
        exclude_patterns=[r"REASSIGN", r"PRACTICE", r"LOCATION", r"pecos_orgs"],
    )


def main():
    ensure_directories()
    session = new_session()

    print("\n" + "=" * 80)
    print("CMS PECOS-STYLE ORGANIZATION / LOCATION DUMP")
    print("=" * 80)
    print("\nThis tool will:")
    print(" - Download CMS enrollment, reassignment, and practice-location assets")
    print(" - Convert discovered CSV files to parquet")
    print(" - Build a dated pecos_orgs_YYYYMMDD.parquet from organization/location rows")
    print("=" * 80 + "\n")

    choice = input(
        "Choose:\n"
        "1. Download + convert + build org parquet (default)\n"
        "2. Build org parquet from latest existing enrollment parquet\n\n"
        "Enter 1 or 2: "
    ).strip() or "1"

    force = input("Force re-download if files exist? (y/N): ").strip().lower() == "y"

    if choice == "1":
        download_assets(
            session=session,
            file_patterns=CMS_ENROLLMENT_FILE_PATTERNS,
            source_name="ppef",
            logger=logger,
            force_redownload=force,
        )

    enrollment_parquet = find_latest_enrollment_parquet()
    if enrollment_parquet is None or not enrollment_parquet.exists():
        logger.error("Could not find an enrollment parquet to build PECOS org rows from.")
        return

    build_orgs_parquet(enrollment_parquet)


if __name__ == "__main__":
    main()
