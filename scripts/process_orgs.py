#!/usr/bin/env python3
"""
Process PECOS-style org/location rows into clinic/system entities.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from process_hgi import NominatimClient, load_overpass
from provider_pipeline_common import (
    add_mode_completion,
    address_similarity,
    build_logger,
    coalesce_columns,
    combine_address,
    ensure_dirs,
    extract_date_from_filename,
    find_latest_parquet,
    is_hospital_from_row,
    is_mode_completed,
    load_metadata,
    load_taxonomy_lookup,
    location_url,
    make_hash_id,
    name_similarity,
    normalize_text,
    parse_date_series,
    parquet_columns,
    read_projected_parquet,
    resolve_primary_taxonomy_code,
    resolve_primary_taxonomy_description,
    root_url,
    safe_phone,
    safe_str,
    save_metadata,
    zip5,
)

LOGGER = build_logger("process_orgs")


def _mode_prompt() -> str:
    print("\n" + "=" * 80)
    print("PROCESS ORGS (PECOS/PPEF)")
    print("=" * 80)
    print("Modes:")
    print(" 1. npi_enrichment")
    print(" 2. hgi_enrichment")
    print(" 3. provider_count")
    print(" 4. website_mapping")
    print(" 5. all")
    value = input("\nMode [1/2/3/4/5 or mode name]: ").strip().lower()
    mapping = {
        "1": "npi_enrichment",
        "2": "hgi_enrichment",
        "3": "provider_count",
        "4": "website_mapping",
        "5": "all",
    }
    return mapping.get(value, value or "all")


class OrgProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.parquet_dir = data_dir / "parquet"
        self.processed_dir = data_dir / "processed_data"
        self.hash_dir = data_dir / "hashes"
        ensure_dirs(data_dir)
        self.taxonomy_lookup = load_taxonomy_lookup(data_dir)

    def find_latest_base(self) -> Optional[Path]:
        return find_latest_parquet(self.parquet_dir, "pecos_orgs_*.parquet")

    def find_latest_npi_orgs(self) -> Optional[Path]:
        return find_latest_parquet(self.processed_dir, "npi_organizations_processed_*.parquet") or find_latest_parquet(
            self.parquet_dir, "npi_organizations_*.parquet"
        )

    def find_latest_hgi(self) -> Optional[Path]:
        return find_latest_parquet(self.processed_dir, "hgi_processed_*.parquet") or find_latest_parquet(
            self.parquet_dir, "hgi_*.parquet"
        )

    def find_latest_processed_individuals(self) -> Optional[Path]:
        return find_latest_parquet(self.processed_dir, "ppef_individuals_processed_*.parquet")

    def output_path(self, data_date: str) -> Path:
        return self.processed_dir / f"pecos_orgs_processed_{data_date}.parquet"

    def metadata_path(self, data_date: str) -> Path:
        return self.processed_dir / f"pecos_orgs_processed_{data_date}_metadata.json"

    def load_working_frame(self, base_path: Path, data_date: str) -> pd.DataFrame:
        processed_path = self.output_path(data_date)
        if processed_path.exists():
            LOGGER.info("Loading existing processed parquet for resume: %s", processed_path)
            return pd.read_parquet(processed_path)
        LOGGER.info("Loading base org parquet: %s", base_path)
        return pd.read_parquet(base_path)

    def _coalesce_org_core_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["source_enrollment_id"] = coalesce_columns(
            out,
            ["source_enrollment_id", "SOURCE_ENROLLMENT_ID", "ENRLMT_ID", "ORG_ENRLMT_ID"],
        )
        out["org_npi"] = coalesce_columns(out, ["org_npi", "SOURCE_NPI", "NPI"])
        out["org_name"] = coalesce_columns(
            out,
            [
                "org_name",
                "SOURCE_ORG_NAME",
                "ORG_NAME",
                "Provider Organization Name (Legal Business Name)",
            ],
        )
        out["source_provider_type_desc"] = coalesce_columns(
            out,
            ["source_provider_type_desc", "SOURCE_PROVIDER_TYPE_DESC", "PROVIDER_TYPE_DESC"],
        )
        out["practice_address_1"] = coalesce_columns(
            out,
            [
                "practice_address_1",
                "PRACTICE_LOCATION_LINE_1",
                "PRACTICE_ADDR_LINE_1",
                "ADDR_LINE_1",
                "Provider First Line Business Practice Location Address",
                "Provider First Line Business Mailing Address",
            ],
        )
        out["practice_address_2"] = coalesce_columns(
            out,
            [
                "practice_address_2",
                "PRACTICE_LOCATION_LINE_2",
                "PRACTICE_ADDR_LINE_2",
                "ADDR_LINE_2",
                "Provider Second Line Business Practice Location Address",
                "Provider Second Line Business Mailing Address",
            ],
        )
        out["practice_city"] = coalesce_columns(
            out,
            [
                "practice_city",
                "PRACTICE_LOCATION_CITY",
                "CITY_NAME",
                "CITY",
                "Provider Business Practice Location Address City Name",
                "Provider Business Mailing Address City Name",
            ],
        )
        out["practice_state"] = coalesce_columns(
            out,
            [
                "practice_state",
                "PRACTICE_LOCATION_STATE",
                "STATE_CD",
                "STATE",
                "Provider Business Practice Location Address State Name",
                "Provider Business Mailing Address State Name",
            ],
        )
        out["practice_zip5"] = coalesce_columns(
            out,
            [
                "practice_zip5",
                "PRACTICE_LOCATION_ZIP",
                "ZIP_CD",
                "ZIP",
                "Provider Business Practice Location Address Postal Code",
                "Provider Business Mailing Address Postal Code",
            ],
        ).map(zip5)
        out["practice_phone"] = coalesce_columns(
            out,
            [
                "practice_phone",
                "PHONE",
                "Provider Business Practice Location Address Telephone Number",
                "Provider Business Mailing Address Telephone Number",
                "Authorized Official Telephone Number",
            ],
        ).map(lambda value: safe_phone(value) or "")
        return out

    def npi_enrichment(self, df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if is_mode_completed("npi_enrichment", metadata):
            LOGGER.info("npi_enrichment already completed. Skipping.")
            return df, metadata

        out = self._coalesce_org_core_fields(df)
        npi_path = self.find_latest_npi_orgs()
        if npi_path and npi_path.exists():
            available = parquet_columns(npi_path)
            projection = [
                "NPI",
                "Provider Organization Name (Legal Business Name)",
                "Provider Other Organization Name",
                "Provider First Line Business Practice Location Address",
                "Provider Second Line Business Practice Location Address",
                "Provider Business Practice Location Address City Name",
                "Provider Business Practice Location Address State Name",
                "Provider Business Practice Location Address Postal Code",
                "Provider Business Practice Location Address Telephone Number",
                "Provider First Line Business Mailing Address",
                "Provider Second Line Business Mailing Address",
                "Provider Business Mailing Address City Name",
                "Provider Business Mailing Address State Name",
                "Provider Business Mailing Address Postal Code",
                "Provider Business Mailing Address Telephone Number",
                "is_hospital",
                "org_entity_id",
            ] + [column for column in available if "Healthcare Provider Taxonomy Code_" in column or "Healthcare Provider Primary Taxonomy Switch_" in column]
            npi_df = read_projected_parquet(npi_path, projection).copy()
            rename_map = {column: f"npi__{column}" for column in npi_df.columns if column != "NPI"}
            npi_df = npi_df.rename(columns=rename_map)
            out = out.merge(npi_df, left_on="org_npi", right_on="NPI", how="left")
        else:
            LOGGER.warning("No NPI organizations parquet found. Continuing without NPI enrichment join.")

        out["org_name"] = coalesce_columns(
            out,
            ["org_name", "npi__Provider Organization Name (Legal Business Name)", "npi__Provider Other Organization Name"],
        )
        out["practice_address_1"] = coalesce_columns(
            out,
            ["practice_address_1", "npi__Provider First Line Business Practice Location Address", "npi__Provider First Line Business Mailing Address"],
        )
        out["practice_address_2"] = coalesce_columns(
            out,
            ["practice_address_2", "npi__Provider Second Line Business Practice Location Address", "npi__Provider Second Line Business Mailing Address"],
        )
        out["practice_city"] = coalesce_columns(
            out,
            ["practice_city", "npi__Provider Business Practice Location Address City Name", "npi__Provider Business Mailing Address City Name"],
        )
        out["practice_state"] = coalesce_columns(
            out,
            ["practice_state", "npi__Provider Business Practice Location Address State Name", "npi__Provider Business Mailing Address State Name"],
        )
        out["practice_zip5"] = coalesce_columns(
            out,
            ["practice_zip5", "npi__Provider Business Practice Location Address Postal Code", "npi__Provider Business Mailing Address Postal Code"],
        ).map(zip5)
        out["practice_phone"] = coalesce_columns(
            out,
            ["practice_phone", "npi__Provider Business Practice Location Address Telephone Number", "npi__Provider Business Mailing Address Telephone Number"],
        ).map(lambda value: safe_phone(value) or "")
        out["taxonomy_code_primary"] = resolve_primary_taxonomy_code(out)
        out["taxonomy_desc_primary"] = resolve_primary_taxonomy_description(out, self.taxonomy_lookup)
        if "npi__is_hospital" in out.columns:
            out["is_hospital"] = out["npi__is_hospital"].fillna(False).astype(bool)
        else:
            out["is_hospital"] = False
        out["org_entity_id"] = coalesce_columns(out, ["org_entity_id", "npi__org_entity_id"])
        out["clinic_name"] = coalesce_columns(out, ["clinic_name", "org_name"])
        out["clinic_display_name"] = out["clinic_name"]
        out["system_name"] = coalesce_columns(out, ["system_name", "org_name"])
        out["system_display_name"] = out["system_name"]
        out["is_clinic"] = True
        out["clinic_id"] = out.apply(
            lambda row: make_hash_id(
                "clinic",
                row.get("clinic_name"),
                row.get("practice_address_1"),
                row.get("practice_city"),
                row.get("practice_state"),
                row.get("practice_zip5"),
            ),
            axis=1,
        )
        out["system_id"] = out.apply(
            lambda row: make_hash_id(
                "system",
                row.get("org_entity_id"),
                row.get("system_name"),
                row.get("org_npi"),
                row.get("practice_state"),
            ),
            axis=1,
        )
        out["is_hospital"] = out.apply(is_hospital_from_row, axis=1)
        for column in [
            "hgi_facility_id",
            "hgi_facility_name",
            "hgi_match_confidence",
            "hgi_system_url",
            "hgi_location_url",
            "website",
            "website_source",
            "website_confidence",
            "provider_count",
        ]:
            if column not in out.columns:
                out[column] = 0 if column == "provider_count" else ""
        if "provider_count" in out.columns:
            out["provider_count"] = pd.to_numeric(out["provider_count"], errors="coerce").fillna(0).astype("int64")

        metadata = add_mode_completion(
            metadata,
            "npi_enrichment",
            time.time() - start,
            len(out),
            [
                "clinic_id",
                "clinic_name",
                "system_id",
                "system_name",
                "taxonomy_code_primary",
                "taxonomy_desc_primary",
                "is_hospital",
                "org_entity_id",
            ],
        )
        return out, metadata

    def hgi_enrichment(self, df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if is_mode_completed("hgi_enrichment", metadata):
            LOGGER.info("hgi_enrichment already completed. Skipping.")
            return df, metadata

        out = df.copy()
        hgi_path = self.find_latest_hgi()
        if not hgi_path or not hgi_path.exists():
            LOGGER.warning("No HGI processed parquet found. Skipping HGI enrichment.")
            return out, metadata

        hgi_df = pd.read_parquet(hgi_path).copy()
        hgi_df["hgi_zip5"] = hgi_df["ZIP Code"].astype("string").fillna("").map(zip5)
        hgi_df["hgi_phone"] = hgi_df["Telephone Number"].astype("string").fillna("").map(lambda value: safe_phone(value) or "")
        hgi_df["hgi_name_norm"] = hgi_df["Facility Name"].astype("string").fillna("").map(normalize_text)
        hgi_df["hgi_addr_norm"] = hgi_df["Address"].astype("string").fillna("").map(normalize_text)
        hgi_df["hgi_state_norm"] = hgi_df["State"].astype("string").fillna("").map(normalize_text)
        hgi_buckets = {
            key: bucket.copy()
            for key, bucket in hgi_df.groupby(["hgi_state_norm", "hgi_zip5"], dropna=False)
        }

        def match_row(row: pd.Series) -> tuple[str, str, str, str, str]:
            if not bool(row.get("is_hospital")):
                return ("", "", "", "", "")
            state = normalize_text(row.get("practice_state"))
            postal = row.get("practice_zip5") or ""
            candidates = hgi_buckets.get((state, postal))
            if candidates is None or candidates.empty:
                candidates = hgi_df[hgi_df["hgi_state_norm"] == state]
            if candidates.empty:
                return ("", "", "", "", "")
            row_name = row.get("system_name") or row.get("clinic_name") or ""
            row_addr = combine_address(row.get("practice_address_1"), row.get("practice_address_2"))
            row_phone = row.get("practice_phone") or ""
            best = None
            best_score = 0.0
            for _, candidate in candidates.iterrows():
                score = 0.65 * name_similarity(row_name, candidate.get("Facility Name", ""))
                score += 0.25 * address_similarity(row_addr, candidate.get("Address", ""))
                if row_phone and candidate.get("hgi_phone") and row_phone == candidate.get("hgi_phone"):
                    score += 0.10
                if score > best_score:
                    best_score = score
                    best = candidate
            if best is None or best_score < 0.55:
                return ("", "", "", "", "")
            return (
                str(best.get("Facility ID", "") or ""),
                str(best.get("Facility Name", "") or ""),
                str(best.get("system_url", "") or ""),
                str(best.get("location_url", "") or ""),
                f"{best_score:.3f}",
            )

        matched = out.apply(match_row, axis=1)
        out["hgi_facility_id"] = matched.map(lambda item: item[0])
        out["hgi_facility_name"] = matched.map(lambda item: item[1])
        out["hgi_system_url"] = matched.map(lambda item: item[2])
        out["hgi_location_url"] = matched.map(lambda item: item[3])
        out["hgi_match_confidence"] = matched.map(lambda item: item[4])

        metadata = add_mode_completion(
            metadata,
            "hgi_enrichment",
            time.time() - start,
            len(out),
            ["hgi_facility_id", "hgi_facility_name", "hgi_system_url", "hgi_location_url", "hgi_match_confidence"],
        )
        return out, metadata

    def provider_count(self, df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if is_mode_completed("provider_count", metadata):
            LOGGER.info("provider_count already completed. Skipping.")
            return df, metadata

        out = df.copy()
        individuals_path = self.find_latest_processed_individuals()
        if not individuals_path or not individuals_path.exists():
            LOGGER.warning("No processed PPEF individuals parquet found. provider_count will remain 0.")
            out["provider_count"] = pd.to_numeric(out.get("provider_count", 0), errors="coerce").fillna(0).astype("int64")
            return out, metadata

        ind_df = pd.read_parquet(individuals_path, columns=["provider_id", "mapped_clinic_id"]).copy()
        ind_df["provider_id"] = ind_df["provider_id"].astype("string").fillna("").str.strip()
        ind_df["mapped_clinic_id"] = ind_df["mapped_clinic_id"].astype("string").fillna("").str.strip()
        ind_df = ind_df[(ind_df["provider_id"] != "") & (ind_df["mapped_clinic_id"] != "")]
        counts = ind_df.groupby("mapped_clinic_id")["provider_id"].nunique()
        out["provider_count"] = out["clinic_id"].astype("string").map(counts).fillna(0).astype("int64")

        metadata = add_mode_completion(
            metadata,
            "provider_count",
            time.time() - start,
            len(out),
            ["provider_count"],
        )
        return out, metadata

    def _overpass_match(self, overpass: dict, row: pd.Series) -> tuple[str, float]:
        postal5 = row.get("practice_zip5") or ""
        name = row.get("system_name") or row.get("clinic_name") or ""
        address = combine_address(row.get("practice_address_1"), row.get("practice_address_2"))
        city = normalize_text(row.get("practice_city"))
        state = normalize_text(row.get("practice_state"))
        best_url = ""
        best_score = 0.0
        for element in (overpass or {}).get("elements", []):
            tags = element.get("tags") or {}
            osm_name = tags.get("name") or tags.get("official_name") or ""
            if not osm_name:
                continue
            osm_state = normalize_text(tags.get("addr:state") or "")
            osm_city = normalize_text(tags.get("addr:city") or "")
            osm_post = zip5(tags.get("addr:postcode"))
            if state and osm_state and state != osm_state:
                continue
            if city and osm_city and city != osm_city:
                continue
            if postal5 and osm_post and postal5 != osm_post:
                continue
            score = 0.70 * name_similarity(name, osm_name)
            score += 0.25 * address_similarity(
                address,
                combine_address(tags.get("addr:housenumber"), tags.get("addr:street")),
            )
            if postal5 and osm_post and postal5 == osm_post:
                score += 0.05
            if score > best_score:
                best_score = score
                best_url = (
                    tags.get("website")
                    or tags.get("contact:website")
                    or tags.get("url")
                    or tags.get("operator:website")
                    or tags.get("brand:website")
                    or ""
                )
        return (root_url(best_url) or "", best_score)

    def website_mapping(self, df: pd.DataFrame, metadata: dict, offline_only: bool = False) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if is_mode_completed("website_mapping", metadata):
            LOGGER.info("website_mapping already completed. Skipping.")
            return df, metadata

        out = df.copy()
        for column in ["website", "website_source", "website_confidence", "hgi_system_url", "hgi_location_url", "hgi_match_confidence", "is_hospital"]:
            if column not in out.columns:
                out[column] = ""
        out["website"] = out["website"].astype("string").fillna("").str.strip()
        out["website_source"] = out["website_source"].astype("string").fillna("").str.strip()
        out["website_confidence"] = out["website_confidence"].astype("string").fillna("").str.strip()

        hgi_mask = out["website"].eq("") & (
            out["hgi_system_url"].astype("string").fillna("").str.strip().ne("")
            | out["hgi_location_url"].astype("string").fillna("").str.strip().ne("")
        )
        out.loc[hgi_mask, "website"] = (
            out.loc[hgi_mask, "hgi_system_url"].astype("string").fillna("").map(root_url).fillna("")
        )
        missing_hgi = hgi_mask & out["website"].eq("")
        out.loc[missing_hgi, "website"] = (
            out.loc[missing_hgi, "hgi_location_url"].astype("string").fillna("").map(root_url).fillna("")
        )
        out.loc[hgi_mask & out["website"].ne(""), "website_source"] = "hgi"
        out.loc[hgi_mask & out["website"].ne(""), "website_confidence"] = out.loc[hgi_mask & out["website"].ne(""), "hgi_match_confidence"]

        overpass = load_overpass(self.data_dir)
        need_lookup = out["website"].eq("") & out["is_hospital"].fillna(False).astype(bool)
        if overpass:
            for idx, row in out.loc[need_lookup].iterrows():
                url, score = self._overpass_match(overpass, row)
                if url and score >= 0.55:
                    out.at[idx, "website"] = url
                    out.at[idx, "website_source"] = "overpass"
                    out.at[idx, "website_confidence"] = f"{score:.3f}"

        need_lookup = out["website"].eq("") & out["is_hospital"].fillna(False).astype(bool)
        if need_lookup.any():
            date_str = metadata.get("date") or datetime.now().strftime("%Y%m%d")
            client = NominatimClient(cache_path=self.hash_dir / f"osm_website_cache_pecos_{date_str}.json", offline_only=offline_only)
            for idx, row in out.loc[need_lookup].iterrows():
                result = client.lookup_website(
                    name=row.get("system_name") or row.get("clinic_name") or "",
                    address=combine_address(row.get("practice_address_1"), row.get("practice_address_2")),
                    city=row.get("practice_city") or "",
                    state=row.get("practice_state") or "",
                    postal=row.get("practice_zip5") or "",
                    min_importance=0.25,
                )
                if not result or not result.get("url"):
                    continue
                out.at[idx, "website"] = root_url(result.get("url")) or ""
                out.at[idx, "website_source"] = "nominatim"
                out.at[idx, "website_confidence"] = str(result.get("confidence", 0.6))

        metadata = add_mode_completion(
            metadata,
            "website_mapping",
            time.time() - start,
            len(out),
            ["website", "website_source", "website_confidence"],
        )
        return out, metadata


def main():
    parser = argparse.ArgumentParser(description="Process PECOS/PPEF org/location rows.")
    parser.add_argument("mode", nargs="?", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--input-parquet", default=None)
    parser.add_argument("--offline-only", action="store_true")
    args = parser.parse_args()

    mode = args.mode or _mode_prompt()
    valid_modes = {"npi_enrichment", "hgi_enrichment", "provider_count", "website_mapping", "all"}
    if mode not in valid_modes:
        raise SystemExit(f"Invalid mode: {mode}")

    processor = OrgProcessor(Path(args.data_dir))
    base_path = Path(args.input_parquet) if args.input_parquet else processor.find_latest_base()
    if base_path is None or not base_path.exists():
        raise SystemExit("No pecos_orgs parquet found. Run scripts/pecos_dump.py first.")

    data_date = extract_date_from_filename(base_path)
    metadata_path = processor.metadata_path(data_date)
    metadata = load_metadata(metadata_path, data_date)
    df = processor.load_working_frame(base_path, data_date)

    if mode in {"hgi_enrichment", "provider_count", "website_mapping"} and "clinic_id" not in df.columns:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"npi_enrichment", "all"}:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"hgi_enrichment", "all"}:
        df, metadata = processor.hgi_enrichment(df, metadata)
    if mode in {"provider_count", "all"}:
        df, metadata = processor.provider_count(df, metadata)
    if mode in {"website_mapping", "all"}:
        df, metadata = processor.website_mapping(df, metadata, offline_only=args.offline_only)

    metadata["total_records"] = int(len(df))
    output_path = processor.output_path(data_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_metadata(metadata, metadata_path)
    LOGGER.info("Saved processed orgs: %s (%s rows, %s columns)", output_path, f"{len(df):,}", len(df.columns))


if __name__ == "__main__":
    main()
