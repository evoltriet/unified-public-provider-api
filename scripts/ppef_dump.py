#!/usr/bin/env python3
"""
CMS PPEF downloader focused on individual providers.

This script:
1. Downloads the public CMS enrollment, reassignment, and practice-location assets.
2. Converts discovered CSV assets to parquet.
3. Builds a dated `ppef_individuals_YYYYMMDD.parquet` from the enrollment file.

The downstream `process_individuals.py` script uses the dated individual parquet as
its base dataset and links back to the relational files for affiliation mapping.
"""

from __future__ import annotations

import re
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

PPEF_FILE_PATTERNS = [r"ENROLL", r"REASSIGN", r"PRACTICE", r"LOCATION"]

logger = build_logger("ppef_individual_provider_dump", "ppef_individual_provider_dump.log")


def build_individuals_parquet(enrollment_parquet: Path, output_date: str | None = None) -> Path:
    output_date = output_date or datetime.now().strftime("%Y%m%d")
    logger.info("Building individual provider parquet from %s", enrollment_parquet)
    df = pd.read_parquet(enrollment_parquet)

    enrlmt_id = column_as_string(df, "ENRLMT_ID")
    first_name = column_as_string(df, "FIRST_NAME")
    middle_name = column_as_string(df, "MDL_NAME")
    last_name = column_as_string(df, "LAST_NAME")
    org_name = column_as_string(df, "ORG_NAME")
    provider_type_desc = column_as_string(df, "PROVIDER_TYPE_DESC")
    npi = column_as_string(df, "NPI")

    is_individual = (
        enrlmt_id.str.startswith("I")
        | (last_name != "")
        | (first_name != "")
        | provider_type_desc.str.contains("PRACTITIONER", case=False, na=False)
    )
    is_individual &= org_name.eq("") | enrlmt_id.str.startswith("I")

    individuals = df.loc[is_individual].copy()
    full_name = (
        first_name.loc[individuals.index]
        + " "
        + middle_name.loc[individuals.index].where(middle_name.loc[individuals.index] != "", "")
        + " "
        + last_name.loc[individuals.index]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    individuals["PROVIDER_FULL_NAME"] = full_name
    individuals["PROVIDER_ID"] = npi.loc[individuals.index].where(npi.loc[individuals.index] != "", enrlmt_id.loc[individuals.index])

    output_path = PARQUET_DIR / f"ppef_individuals_{output_date}.parquet"
    individuals.to_parquet(output_path, index=False, compression="snappy")
    warn_if_schema_drift(individuals, "ppef_individuals", logger)
    logger.info(
        "Saved individual parquet: %s (%s rows, %s columns)",
        output_path,
        f"{len(individuals):,}",
        len(individuals.columns),
    )
    return output_path


def find_latest_enrollment_parquet() -> Path | None:
    return find_latest_parquet(
        include_patterns=[r"ENROLL"],
        exclude_patterns=[r"REASSIGN", r"PRACTICE", r"LOCATION", r"ppef_individuals"],
    )


def main():
    ensure_directories()
    session = new_session()

    print("\n" + "=" * 80)
    print("CMS PPEF INDIVIDUAL PROVIDER DUMP")
    print("=" * 80)
    print("\nThis tool will:")
    print(" - Download CMS enrollment, reassignment, and practice-location assets")
    print(" - Convert discovered CSV files to parquet")
    print(" - Build a dated ppef_individuals_YYYYMMDD.parquet from enrollment data")
    print("=" * 80 + "\n")

    choice = input(
        "Choose:\n"
        "1. Download + convert + build individuals parquet (default)\n"
        "2. Build individuals parquet from latest existing enrollment parquet\n\n"
        "Enter 1 or 2: "
    ).strip() or "1"

    force = input("Force re-download if files exist? (y/N): ").strip().lower() == "y"

    if choice == "1":
        download_assets(
            session=session,
            file_patterns=PPEF_FILE_PATTERNS,
            source_name="ppef",
            logger=logger,
            force_redownload=force,
        )

    enrollment_parquet = find_latest_enrollment_parquet()
    if enrollment_parquet is None or not enrollment_parquet.exists():
        logger.error("Could not find a PPEF enrollment parquet to build individuals from.")
        return

    build_individuals_parquet(enrollment_parquet)


if __name__ == "__main__":
    main()
