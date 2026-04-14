#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from provider_pipeline_common import (
    add_mode_completion,
    address_similarity,
    build_logger,
    coalesce_columns,
    combine_address,
    ensure_dirs,
    extract_date_from_filename,
    find_column_by_tokens,
    find_latest_parquet,
    first_existing_column,
    load_metadata,
    load_taxonomy_lookup,
    name_similarity,
    parquet_columns,
    read_projected_parquet,
    resolve_primary_taxonomy_code,
    resolve_primary_taxonomy_description,
    safe_phone,
    save_metadata,
    zip5,
)

LOGGER = build_logger("process_individuals")


def _mode_prompt() -> str:
    print("\n" + "=" * 80)
    print("PROCESS INDIVIDUALS (PPEF)")
    print("=" * 80)
    print("Modes:")
    print(" 1. npi_enrichment")
    print(" 2. clinic_mapping")
    print(" 3. all")
    value = input("\nMode [1/2/3 or mode name]: ").strip().lower()
    return {"1": "npi_enrichment", "2": "clinic_mapping", "3": "all"}.get(value, value or "all")


def _completed(mode: str, metadata: dict) -> bool:
    return any(item.get("mode") == mode for item in metadata.get("processing_modes_completed", []))


class IndividualProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.parquet_dir = data_dir / "parquet"
        self.processed_dir = data_dir / "processed_data"
        ensure_dirs(data_dir)
        self.taxonomy_lookup = load_taxonomy_lookup(data_dir)

    def find_latest_base(self) -> Optional[Path]:
        return find_latest_parquet(self.parquet_dir, "ppef_individuals_*.parquet")

    def find_latest_npi_individuals(self) -> Optional[Path]:
        return find_latest_parquet(self.processed_dir, "npi_individuals_processed_*.parquet") or find_latest_parquet(
            self.parquet_dir, "npi_individuals_*.parquet"
        )

    def find_latest_processed_orgs(self) -> Optional[Path]:
        return find_latest_parquet(self.processed_dir, "pecos_orgs_processed_*.parquet")

    def find_related_parquet(self, include_patterns: list[str], exclude_patterns: list[str] | None = None) -> Optional[Path]:
        exclude_patterns = exclude_patterns or []
        matches = []
        for path in sorted(self.parquet_dir.glob("*.parquet")):
            if all(re.search(pattern, path.name, re.I) for pattern in include_patterns) and not any(
                re.search(pattern, path.name, re.I) for pattern in exclude_patterns
            ):
                matches.append(path)
        return matches[-1] if matches else None

    def output_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individuals_processed_{date_str}.parquet"

    def links_output_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individual_affiliation_links_{date_str}.parquet"

    def metadata_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individuals_processed_{date_str}_metadata.json"

    def load_working_frame(self, base_path: Path, date_str: str) -> pd.DataFrame:
        processed = self.output_path(date_str)
        if processed.exists():
            LOGGER.info("Loading existing processed individuals parquet: %s", processed)
            return pd.read_parquet(processed)
        LOGGER.info("Loading base individuals parquet: %s", base_path)
        return pd.read_parquet(base_path)

    def npi_enrichment(self, df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if _completed("npi_enrichment", metadata):
            LOGGER.info("npi_enrichment already completed. Skipping.")
            return df, metadata

        out = df.copy()
        out["enrollment_id"] = coalesce_columns(out, ["enrollment_id", "ENRLMT_ID"])
        out["npi"] = coalesce_columns(out, ["npi", "NPI"])
        out["provider_id"] = out["npi"].where(out["npi"] != "", out["enrollment_id"])

        npi_path = self.find_latest_npi_individuals()
        if npi_path and npi_path.exists():
            available = parquet_columns(npi_path)
            projection = [
                "NPI",
                "Provider First Name",
                "Provider Middle Name",
                "Provider Last Name (Legal Name)",
                "Provider First Line Business Practice Location Address",
                "Provider Second Line Business Practice Location Address",
                "Provider Business Practice Location Address City Name",
                "Provider Business Practice Location Address State Name",
                "Provider Business Practice Location Address Postal Code",
                "Provider Business Practice Location Address Telephone Number",
            ] + [c for c in available if "Healthcare Provider Taxonomy Code_" in c or "Healthcare Provider Primary Taxonomy Switch_" in c]
            npi_df = read_projected_parquet(npi_path, projection).copy()
            npi_df = npi_df.rename(columns={c: f"npi__{c}" for c in npi_df.columns if c != "NPI"})
            out = out.merge(npi_df, left_on="npi", right_on="NPI", how="left")
        else:
            LOGGER.warning("No NPI individuals parquet found. Continuing with PPEF-only fields.")

        out["first_name"] = coalesce_columns(out, ["first_name", "FIRST_NAME", "npi__Provider First Name"])
        out["middle_name"] = coalesce_columns(out, ["middle_name", "MDL_NAME", "npi__Provider Middle Name"])
        out["last_name"] = coalesce_columns(out, ["last_name", "LAST_NAME", "npi__Provider Last Name (Legal Name)"])
        out["provider_full_name"] = coalesce_columns(out, ["provider_full_name", "PROVIDER_FULL_NAME"])
        missing = out["provider_full_name"].eq("")
        out.loc[missing, "provider_full_name"] = (
            out.loc[missing, "first_name"].astype("string").fillna("")
            + " "
            + out.loc[missing, "middle_name"].astype("string").fillna("")
            + " "
            + out.loc[missing, "last_name"].astype("string").fillna("")
        ).str.replace(r"\s+", " ", regex=True).str.strip()
        out["provider_practice_address_1"] = coalesce_columns(
            out, ["provider_practice_address_1", "npi__Provider First Line Business Practice Location Address"]
        )
        out["provider_practice_address_2"] = coalesce_columns(
            out, ["provider_practice_address_2", "npi__Provider Second Line Business Practice Location Address"]
        )
        out["provider_practice_city"] = coalesce_columns(
            out, ["provider_practice_city", "npi__Provider Business Practice Location Address City Name"]
        )
        out["provider_practice_state"] = coalesce_columns(
            out, ["provider_practice_state", "npi__Provider Business Practice Location Address State Name"]
        )
        out["provider_practice_zip5"] = coalesce_columns(
            out, ["provider_practice_zip5", "npi__Provider Business Practice Location Address Postal Code"]
        ).map(zip5)
        out["provider_practice_phone"] = coalesce_columns(
            out, ["provider_practice_phone", "npi__Provider Business Practice Location Address Telephone Number"]
        ).map(lambda v: safe_phone(v) or "")
        out["taxonomy_code_primary"] = resolve_primary_taxonomy_code(out)
        out["taxonomy_desc_primary"] = resolve_primary_taxonomy_description(out, self.taxonomy_lookup)
        for column in [
            "mapped_clinic_id",
            "mapped_clinic_name",
            "mapped_system_id",
            "mapped_system_name",
            "mapping_confidence",
            "mapping_method",
            "active_reassignment_count",
            "most_recent_reassignment_date",
            "longest_reassignment_days",
            "primary_affiliation_start_date",
            "primary_affiliation_end_date",
            "billing_affiliation_recency_rank",
            "shared_address_provider_count",
        ]:
            if column not in out.columns:
                out[column] = ""

        metadata = add_mode_completion(
            metadata,
            "npi_enrichment",
            time.time() - start,
            len(out),
            ["provider_id", "npi", "enrollment_id", "provider_full_name", "taxonomy_code_primary", "taxonomy_desc_primary"],
        )
        return out, metadata

    @staticmethod
    def _address_key(address1, city, state, postal) -> str:
        return "|".join([combine_address(address1).lower().strip(), str(city or "").strip().lower(), str(state or "").strip().lower(), zip5(postal)])

    def _resolve_reassignment_schema(self, df: pd.DataFrame) -> dict[str, Optional[str]]:
        return {
            "provider_npi": first_existing_column(df, ["INDIVIDUAL_NPI", "INDVL_NPI", "REASGN_BNFTS_INDVL_NPI", "NPI"])
            or find_column_by_tokens(df, required_tokens=["npi"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org", "recv"]),
            "provider_enrollment_id": first_existing_column(df, ["INDIVIDUAL_ENROLLMENT_ID", "INDVL_ENRLMT_ID", "ENRLMT_ID"])
            or find_column_by_tokens(df, required_tokens=["enrlmt"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org", "recv"]),
            "org_npi": first_existing_column(df, ["ORG_NPI", "RCVG_NPI", "REASSIGN_TO_NPI"])
            or find_column_by_tokens(df, required_tokens=["npi"], any_tokens=["org", "recv", "reassign"], exclude_tokens=["individual", "indvl", "provider"]),
            "org_enrollment_id": first_existing_column(df, ["ORG_ENROLLMENT_ID", "ORG_ENRLMT_ID", "RCVG_ENRLMT_ID", "REASSIGN_TO_ENRLMT_ID"])
            or find_column_by_tokens(df, required_tokens=["enrlmt"], any_tokens=["org", "recv", "reassign"], exclude_tokens=["individual", "indvl", "provider"]),
            "start_date": first_existing_column(df, ["EFF_DT", "START_DT", "BGN_DT", "BEGIN_DT"]) or find_column_by_tokens(df, required_tokens=["dt"], any_tokens=["eff", "start", "begin"]),
            "end_date": first_existing_column(df, ["END_DT", "TERM_DT", "TERMINATION_DT"]) or find_column_by_tokens(df, required_tokens=["dt"], any_tokens=["end", "term"]),
            "org_name": first_existing_column(df, ["ORG_NAME", "REASSIGN_TO_ORG_NAME"]) or find_column_by_tokens(df, required_tokens=["org", "name"]),
            "address1": first_existing_column(df, ["ADDR_LINE_1", "ADDRESS_LINE_1"]) or find_column_by_tokens(df, required_tokens=["address"], any_tokens=["line", "street"]),
            "address2": first_existing_column(df, ["ADDR_LINE_2", "ADDRESS_LINE_2"]),
            "city": first_existing_column(df, ["CITY", "CITY_NAME"]) or find_column_by_tokens(df, required_tokens=["city"]),
            "state": first_existing_column(df, ["STATE", "STATE_CD"]) or find_column_by_tokens(df, required_tokens=["state"]),
            "zip": first_existing_column(df, ["ZIP", "ZIP_CD", "POSTAL_CODE"]) or find_column_by_tokens(df, required_tokens=["zip"]),
            "phone": first_existing_column(df, ["PHONE", "TELEPHONE"]) or find_column_by_tokens(df, required_tokens=["phone"]),
        }

    def _resolve_practice_schema(self, df: pd.DataFrame) -> dict[str, Optional[str]]:
        return {
            "provider_npi": first_existing_column(df, ["NPI", "INDIVIDUAL_NPI", "INDVL_NPI"])
            or find_column_by_tokens(df, required_tokens=["npi"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org"]),
            "provider_enrollment_id": first_existing_column(df, ["ENRLMT_ID", "INDVL_ENRLMT_ID", "INDIVIDUAL_ENROLLMENT_ID"])
            or find_column_by_tokens(df, required_tokens=["enrlmt"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org"]),
            "org_npi": first_existing_column(df, ["ORG_NPI"]),
            "org_enrollment_id": first_existing_column(df, ["ORG_ENRLMT_ID", "ORG_ENROLLMENT_ID"]),
            "org_name": first_existing_column(df, ["ORG_NAME", "PRACTICE_NAME"]) or find_column_by_tokens(df, required_tokens=["name"], any_tokens=["org", "practice"]),
            "address1": first_existing_column(df, ["PRACTICE_LOCATION_LINE_1", "ADDR_LINE_1", "ADDRESS_LINE_1"]) or find_column_by_tokens(df, required_tokens=["address"], any_tokens=["line", "street"]),
            "address2": first_existing_column(df, ["PRACTICE_LOCATION_LINE_2", "ADDR_LINE_2", "ADDRESS_LINE_2"]),
            "city": first_existing_column(df, ["CITY", "CITY_NAME"]) or find_column_by_tokens(df, required_tokens=["city"]),
            "state": first_existing_column(df, ["STATE", "STATE_CD"]) or find_column_by_tokens(df, required_tokens=["state"]),
            "zip": first_existing_column(df, ["ZIP", "ZIP_CD", "POSTAL_CODE"]) or find_column_by_tokens(df, required_tokens=["zip"]),
            "phone": first_existing_column(df, ["PHONE", "TELEPHONE"]) or find_column_by_tokens(df, required_tokens=["phone"]),
            "start_date": first_existing_column(df, ["EFF_DT", "START_DT", "BEGIN_DT"]),
            "end_date": first_existing_column(df, ["END_DT", "TERM_DT"]),
        }

    def _build_org_lookups(self, org_df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
        lookups = {"by_org_npi": {}, "by_enrollment": {}, "by_address": {}, "by_phone_zip": {}}
        for idx, row in org_df.iterrows():
            org_npi = str(row.get("org_npi") or "").strip()
            enrollment = str(row.get("source_enrollment_id") or "").strip()
            address_key = self._address_key(row.get("practice_address_1"), row.get("practice_city"), row.get("practice_state"), row.get("practice_zip5"))
            phone = str(row.get("practice_phone") or "").strip()
            postal = str(row.get("practice_zip5") or "").strip()
            if org_npi:
                lookups["by_org_npi"].setdefault(org_npi, []).append(idx)
            if enrollment:
                lookups["by_enrollment"].setdefault(enrollment, []).append(idx)
            if address_key:
                lookups["by_address"].setdefault(address_key, []).append(idx)
            if phone and postal:
                lookups["by_phone_zip"].setdefault(f"{phone}|{postal}", []).append(idx)
        return lookups

    def _lookup_provider_idx(self, row: pd.Series, schema: dict[str, Optional[str]], npi_map: dict[str, int], enrollment_map: dict[str, int]) -> Optional[int]:
        npi = str(row.get(schema["provider_npi"]) or "").strip() if schema.get("provider_npi") else ""
        enrollment = str(row.get(schema["provider_enrollment_id"]) or "").strip() if schema.get("provider_enrollment_id") else ""
        if npi and npi in npi_map:
            return npi_map[npi]
        if enrollment and enrollment in enrollment_map:
            return enrollment_map[enrollment]
        return None

    def _lookup_org_candidates(self, row: pd.Series, schema: dict[str, Optional[str]], lookups: dict[str, dict[str, list[int]]], org_df: pd.DataFrame) -> tuple[list[int], str]:
        indexes: list[int] = []
        method = ""
        org_npi = str(row.get(schema["org_npi"]) or "").strip() if schema.get("org_npi") else ""
        enrollment = str(row.get(schema["org_enrollment_id"]) or "").strip() if schema.get("org_enrollment_id") else ""
        address_key = self._address_key(row.get(schema["address1"]) if schema.get("address1") else "", row.get(schema["city"]) if schema.get("city") else "", row.get(schema["state"]) if schema.get("state") else "", row.get(schema["zip"]) if schema.get("zip") else "")
        phone = safe_phone(row.get(schema["phone"])) if schema.get("phone") else None
        postal = zip5(row.get(schema["zip"])) if schema.get("zip") else ""
        org_name = str(row.get(schema["org_name"]) or "").strip() if schema.get("org_name") else ""
        if org_npi and org_npi in lookups["by_org_npi"]:
            indexes.extend(lookups["by_org_npi"][org_npi]); method = "org_npi"
        if enrollment and enrollment in lookups["by_enrollment"]:
            indexes.extend(lookups["by_enrollment"][enrollment]); method = method or "org_enrollment_id"
        if not indexes and address_key and address_key in lookups["by_address"]:
            indexes.extend(lookups["by_address"][address_key]); method = "address"
        if not indexes and phone and postal and f"{phone}|{postal}" in lookups["by_phone_zip"]:
            indexes.extend(lookups["by_phone_zip"][f"{phone}|{postal}"]); method = "phone_zip"
        if not indexes and org_name and postal:
            state = str(row.get(schema["state"]) or "").strip().lower() if schema.get("state") else ""
            bucket = org_df[(org_df["practice_zip5"] == postal) & (org_df["practice_state"].astype("string").fillna("").str.lower() == state)]
            if not bucket.empty:
                scores = bucket["clinic_name"].astype("string").fillna("").map(lambda v: name_similarity(v, org_name))
                hits = bucket.index[scores >= 0.65].tolist()
                if hits:
                    indexes.extend(hits); method = "name_zip"
        return list(dict.fromkeys(indexes)), method

    def _score_candidate(self, provider_row: pd.Series, org_row: pd.Series, raw_row: pd.Series, schema: dict[str, Optional[str]], relationship_source: str, method: str) -> dict[str, object]:
        today = pd.Timestamp.today().normalize()
        start_date = pd.to_datetime(raw_row.get(schema["start_date"]) if schema.get("start_date") else None, errors="coerce")
        end_date = pd.to_datetime(raw_row.get(schema["end_date"]) if schema.get("end_date") else None, errors="coerce")
        active = pd.isna(end_date) or end_date >= today
        ref_date = start_date if active and not pd.isna(start_date) else end_date if not pd.isna(end_date) else start_date
        recency_days = max((today - ref_date.normalize()).days, 0) if pd.notna(ref_date) else 99999
        duration_days = max(((today if pd.isna(end_date) else end_date.normalize()) - start_date.normalize()).days, 0) if pd.notna(start_date) else 0
        raw_address = combine_address(raw_row.get(schema["address1"]) if schema.get("address1") else "", raw_row.get(schema["address2"]) if schema.get("address2") else "")
        provider_address = combine_address(provider_row.get("provider_practice_address_1"), provider_row.get("provider_practice_address_2"))
        org_address = combine_address(org_row.get("practice_address_1"), org_row.get("practice_address_2"))
        base_address = raw_address or provider_address
        address_score = address_similarity(base_address, org_address) if base_address and org_address else 0.0
        raw_phone = safe_phone(raw_row.get(schema["phone"])) if schema.get("phone") else None
        provider_phone = safe_phone(provider_row.get("provider_practice_phone"))
        org_phone = safe_phone(org_row.get("practice_phone"))
        phone_match = bool(org_phone and ((raw_phone and raw_phone == org_phone) or (provider_phone and provider_phone == org_phone)))
        org_name = raw_row.get(schema["org_name"]) if schema.get("org_name") else ""
        name_score = max(name_similarity(org_name, org_row.get("clinic_name")), name_similarity(org_name, org_row.get("system_name")))
        provider_taxonomy = str(provider_row.get("taxonomy_desc_primary") or "")
        org_taxonomy = str(org_row.get("taxonomy_desc_primary") or "")
        taxonomy_score = name_similarity(provider_taxonomy, org_taxonomy) if provider_taxonomy and org_taxonomy else 0.0
        recency_score = 2.5 if active else max(0.0, 2.0 - min(recency_days, 3650) / 365.0)
        continuity_score = min(duration_days / 365.0, 2.5)
        total = recency_score + continuity_score + (address_score * 2.0) + (1.25 if phone_match else 0.0) + (name_score * 1.5) + taxonomy_score
        if method == "org_npi":
            total += 4.0
        elif method == "org_enrollment_id":
            total += 3.5
        elif method == "address":
            total += 2.0
        elif method == "phone_zip":
            total += 1.5
        elif method == "name_zip":
            total += 1.25
        if relationship_source == "reassignment":
            total += 1.5
        if active:
            total += 3.0
        return {
            "relationship_start_date": start_date,
            "relationship_end_date": end_date,
            "is_active_relationship": bool(active),
            "recency_score": recency_score,
            "continuity_score": continuity_score,
            "address_match_score": address_score,
            "taxonomy_match_score": taxonomy_score,
            "total_affiliation_score": total,
            "raw_address_key": self._address_key(raw_row.get(schema["address1"]) if schema.get("address1") else "", raw_row.get(schema["city"]) if schema.get("city") else "", raw_row.get(schema["state"]) if schema.get("state") else "", raw_row.get(schema["zip"]) if schema.get("zip") else ""),
            "relationship_duration_days": duration_days,
            "raw_recency_days": recency_days,
        }

    def _build_candidates(self, relation_df: pd.DataFrame, schema: dict[str, Optional[str]], relationship_source: str, individuals_df: pd.DataFrame, org_df: pd.DataFrame, provider_npi_map: dict[str, int], provider_enrollment_map: dict[str, int], org_lookups: dict[str, dict[str, list[int]]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if relation_df is None or relation_df.empty:
            return rows
        for _, raw_row in relation_df.iterrows():
            provider_idx = self._lookup_provider_idx(raw_row, schema, provider_npi_map, provider_enrollment_map)
            if provider_idx is None:
                continue
            org_indexes, method = self._lookup_org_candidates(raw_row, schema, org_lookups, org_df)
            if not org_indexes:
                continue
            provider_row = individuals_df.loc[provider_idx]
            for org_idx in org_indexes:
                org_row = org_df.loc[org_idx]
                bits = self._score_candidate(provider_row, org_row, raw_row, schema, relationship_source, method)
                rows.append({
                    "provider_id": provider_row.get("provider_id", ""),
                    "clinic_id": org_row.get("clinic_id", ""),
                    "clinic_name": org_row.get("clinic_name", ""),
                    "system_id": org_row.get("system_id", ""),
                    "system_name": org_row.get("system_name", ""),
                    "relationship_source": relationship_source,
                    "relationship_start_date": bits["relationship_start_date"],
                    "relationship_end_date": bits["relationship_end_date"],
                    "is_active_relationship": bits["is_active_relationship"],
                    "recency_score": bits["recency_score"],
                    "continuity_score": bits["continuity_score"],
                    "address_match_score": bits["address_match_score"],
                    "taxonomy_match_score": bits["taxonomy_match_score"],
                    "total_affiliation_score": bits["total_affiliation_score"],
                    "selected_as_primary": False,
                    "mapping_method": f"{relationship_source}:{method}" if method else relationship_source,
                    "shared_address_provider_count": 0,
                    "billing_affiliation_recency_rank": bits["raw_recency_days"],
                    "relationship_duration_days": bits["relationship_duration_days"],
                    "raw_address_key": bits["raw_address_key"],
                })
        return rows

    def clinic_mapping(self, df: pd.DataFrame, metadata: dict, date_str: str) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        if _completed("clinic_mapping", metadata):
            LOGGER.info("clinic_mapping already completed. Skipping.")
            return df, metadata
        out = df.copy()
        orgs_path = self.find_latest_processed_orgs()
        if not orgs_path or not orgs_path.exists():
            LOGGER.warning("No processed orgs parquet found. Run scripts/process_orgs.py first.")
            return out, metadata

        org_df = pd.read_parquet(orgs_path, columns=["clinic_id", "clinic_name", "system_id", "system_name", "org_npi", "source_enrollment_id", "practice_address_1", "practice_address_2", "practice_city", "practice_state", "practice_zip5", "practice_phone", "taxonomy_desc_primary"]).copy()
        org_df["practice_phone"] = org_df["practice_phone"].astype("string").fillna("").map(lambda v: safe_phone(v) or "")
        org_df["practice_zip5"] = org_df["practice_zip5"].astype("string").fillna("").map(zip5)

        provider_npi_map = {str(v).strip(): idx for idx, v in out["npi"].astype("string").fillna("").items() if str(v).strip()}
        provider_enrollment_map = {str(v).strip(): idx for idx, v in out["enrollment_id"].astype("string").fillna("").items() if str(v).strip()}
        org_lookups = self._build_org_lookups(org_df)

        candidate_rows: list[dict[str, object]] = []
        reassignment_path = self.find_related_parquet(["REASSIGN"], ["ppef_individuals", "pecos_orgs"])
        if reassignment_path and reassignment_path.exists():
            rel_df = pd.read_parquet(reassignment_path)
            candidate_rows.extend(self._build_candidates(rel_df, self._resolve_reassignment_schema(rel_df), "reassignment", out, org_df, provider_npi_map, provider_enrollment_map, org_lookups))
        else:
            LOGGER.warning("No reassignment parquet found.")
        practice_path = self.find_related_parquet(["PRACTICE|LOCATION"], ["ppef_individuals", "pecos_orgs"])
        if practice_path and practice_path.exists():
            prac_df = pd.read_parquet(practice_path)
            candidate_rows.extend(self._build_candidates(prac_df, self._resolve_practice_schema(prac_df), "practice_location", out, org_df, provider_npi_map, provider_enrollment_map, org_lookups))
        else:
            LOGGER.warning("No practice-location parquet found.")

        if not candidate_rows:
            self.links_output_path(date_str).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["provider_id", "clinic_id", "system_id", "relationship_source", "relationship_start_date", "relationship_end_date", "is_active_relationship", "recency_score", "continuity_score", "address_match_score", "taxonomy_match_score", "total_affiliation_score", "selected_as_primary"]).to_parquet(self.links_output_path(date_str), index=False)
            metadata = add_mode_completion(metadata, "clinic_mapping", time.time() - start, len(out), ["mapped_clinic_id", "mapped_clinic_name", "mapped_system_id", "mapped_system_name"])
            return out, metadata

        links_df = pd.DataFrame(candidate_rows)
        shared_counts = links_df.loc[links_df["relationship_source"] == "practice_location"].groupby("raw_address_key")["provider_id"].nunique()
        links_df["shared_address_provider_count"] = links_df["raw_address_key"].map(shared_counts).fillna(0).astype("int64")
        links_df["total_affiliation_score"] = links_df["total_affiliation_score"] + ((links_df["shared_address_provider_count"] - 1).clip(lower=0, upper=10) / 10.0)
        links_df = links_df.sort_values(by=["provider_id", "total_affiliation_score", "is_active_relationship", "recency_score", "continuity_score", "relationship_source"], ascending=[True, False, False, False, False, True])
        primary_idx = links_df.groupby("provider_id").head(1).index
        links_df.loc[primary_idx, "selected_as_primary"] = True
        links_df["billing_affiliation_recency_rank"] = links_df.groupby("provider_id")["billing_affiliation_recency_rank"].rank(method="dense")
        best = links_df.loc[links_df["selected_as_primary"]].copy()
        best["mapping_confidence"] = best["total_affiliation_score"].map(lambda v: f"{min(float(v) / 12.0, 1.0):.3f}")
        reassignment_summary = links_df.loc[links_df["relationship_source"] == "reassignment"].groupby("provider_id").agg(active_reassignment_count=("is_active_relationship", "sum"), most_recent_reassignment_date=("relationship_start_date", "max"), longest_reassignment_days=("relationship_duration_days", "max"))

        out = out.merge(best[["provider_id", "clinic_id", "clinic_name", "system_id", "system_name", "mapping_confidence", "mapping_method", "relationship_start_date", "relationship_end_date", "billing_affiliation_recency_rank", "shared_address_provider_count"]].rename(columns={"clinic_id": "mapped_clinic_id", "clinic_name": "mapped_clinic_name", "system_id": "mapped_system_id", "system_name": "mapped_system_name", "relationship_start_date": "primary_affiliation_start_date", "relationship_end_date": "primary_affiliation_end_date"}), on="provider_id", how="left")
        out = out.merge(reassignment_summary, on="provider_id", how="left")
        for column in ["mapped_clinic_id", "mapped_clinic_name", "mapped_system_id", "mapped_system_name", "mapping_confidence", "mapping_method", "primary_affiliation_start_date", "primary_affiliation_end_date", "most_recent_reassignment_date"]:
            out[column] = out[column].astype("string").fillna("")
        for column in ["active_reassignment_count", "longest_reassignment_days", "shared_address_provider_count", "billing_affiliation_recency_rank"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")

        links_df = links_df.drop(columns=["raw_address_key"], errors="ignore")
        self.links_output_path(date_str).parent.mkdir(parents=True, exist_ok=True)
        links_df.to_parquet(self.links_output_path(date_str), index=False)
        metadata = add_mode_completion(metadata, "clinic_mapping", time.time() - start, len(out), ["mapped_clinic_id", "mapped_clinic_name", "mapped_system_id", "mapped_system_name", "mapping_confidence", "mapping_method"])
        return out, metadata


def main():
    parser = argparse.ArgumentParser(description="Process PPEF individuals into clinic/system mappings.")
    parser.add_argument("mode", nargs="?", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--input-parquet", default=None)
    args = parser.parse_args()
    mode = args.mode or _mode_prompt()
    if mode not in {"npi_enrichment", "clinic_mapping", "all"}:
        raise SystemExit(f"Invalid mode: {mode}")

    processor = IndividualProcessor(Path(args.data_dir))
    base_path = Path(args.input_parquet) if args.input_parquet else processor.find_latest_base()
    if base_path is None or not base_path.exists():
        raise SystemExit("No ppef_individuals parquet found. Run scripts/ppef_dump.py first.")

    date_str = extract_date_from_filename(base_path)
    metadata_path = processor.metadata_path(date_str)
    metadata = load_metadata(metadata_path, date_str)
    df = processor.load_working_frame(base_path, date_str)
    if mode == "clinic_mapping" and "provider_id" not in df.columns:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"npi_enrichment", "all"}:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"clinic_mapping", "all"}:
        df, metadata = processor.clinic_mapping(df, metadata, date_str)

    metadata["total_records"] = int(len(df))
    output = processor.output_path(date_str)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    save_metadata(metadata, metadata_path)
    LOGGER.info("Saved processed individuals: %s (%s rows, %s columns)", output, f"{len(df):,}", len(df.columns))


if __name__ == "__main__":
    main()
