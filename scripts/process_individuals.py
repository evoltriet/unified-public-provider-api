#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from provider_pipeline_common import (
    add_mode_completion,
    address_similarity,
    build_logger,
    canonical_entity_name,
    coalesce_columns,
    combine_address,
    ensure_dirs,
    extract_date_from_filename,
    find_column_by_tokens,
    find_latest_parquet,
    first_existing_column,
    load_metadata,
    load_specialty_rollup,
    load_taxonomy_lookup,
    name_similarity,
    normalize_address_line1,
    normalize_text,
    parquet_columns,
    read_projected_parquet,
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
    print(" 4. quality_exports")
    value = input("\nMode [1/2/3/4 or mode name]: ").strip().lower()
    return {"1": "npi_enrichment", "2": "clinic_mapping", "3": "all", "4": "quality_exports"}.get(value, value or "all")


def _completed(mode: str, metadata: dict) -> bool:
    return any(item.get("mode") == mode for item in metadata.get("processing_modes_completed", []))


class IndividualProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.parquet_dir = data_dir / "parquet"
        self.processed_dir = data_dir / "processed_data"
        ensure_dirs(data_dir)
        self.taxonomy_lookup = load_taxonomy_lookup(data_dir)
        self.specialty_rollup = load_specialty_rollup(data_dir)

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

    def candidates_output_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individual_affiliation_candidates_{date_str}.parquet"

    def quality_report_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_pecos_mapping_quality_{date_str}.csv"

    def quality_report_json_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_pecos_mapping_quality_{date_str}.json"

    def system_quality_report_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_pecos_mapping_quality_systems_{date_str}.csv"

    def field_provenance_summary_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_pecos_field_provenance_summary_{date_str}.csv"

    def specialty_rollup_audit_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_specialty_rollup_audit_{date_str}.csv"

    def unresolved_taxonomy_audit_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_unresolved_taxonomy_audit_{date_str}.csv"

    def comparison_readiness_summary_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_comparison_readiness_summary_{date_str}.csv"

    def system_alias_summary_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_pecos_system_alias_summary_{date_str}.csv"

    def comparison_baseline_output_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individuals_comparison_baseline_{date_str}.parquet"

    def metadata_path(self, date_str: str) -> Path:
        return self.processed_dir / f"ppef_individuals_processed_{date_str}_metadata.json"

    def processed_orgs_signature(self) -> str:
        orgs_path = self.find_latest_processed_orgs()
        if not orgs_path or not orgs_path.exists():
            return ""
        stat = orgs_path.stat()
        return f"{orgs_path.name}:{stat.st_size}:{stat.st_mtime_ns}"

    def load_working_frame(self, base_path: Path, date_str: str) -> pd.DataFrame:
        processed = self.output_path(date_str)
        if processed.exists():
            LOGGER.info("Loading existing processed individuals parquet: %s", processed)
            return pd.read_parquet(processed)
        LOGGER.info("Loading base individuals parquet: %s", base_path)
        return pd.read_parquet(base_path)

    @staticmethod
    def _snapshot_date(path: Optional[Path]) -> str:
        if path is None:
            return ""
        try:
            return extract_date_from_filename(path)
        except Exception:
            return ""

    @staticmethod
    def _has_value(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        has_value = pd.Series(False, index=df.index)
        for column in columns:
            if column in df.columns:
                values = df.loc[:, column]
                if isinstance(values, pd.DataFrame):
                    column_has_value = pd.Series(False, index=df.index)
                    for column_index in range(values.shape[1]):
                        column_has_value = (
                            column_has_value
                            | values.iloc[:, column_index].astype("string").fillna("").str.strip().ne("")
                        )
                else:
                    column_has_value = values.astype("string").fillna("").str.strip().ne("")
                has_value = has_value | column_has_value
        return has_value

    @staticmethod
    def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
        if column not in df.columns:
            return pd.Series(default, index=df.index)
        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(default).astype(bool)
        normalized = values.astype("string").fillna("").str.strip().str.lower()
        return normalized.isin(["1", "true", "t", "yes", "y"])

    @staticmethod
    def _taxonomy_code_columns(df: pd.DataFrame) -> list[str]:
        columns = [
            column
            for column in df.columns
            if "Healthcare Provider Taxonomy Code_" in column or column == "taxonomy_code_primary"
        ]
        return list(dict.fromkeys(columns))

    @staticmethod
    def _primary_switch_for_code_column(df: pd.DataFrame, code_column: str) -> Optional[str]:
        match = re.search(r"Healthcare Provider Taxonomy Code_(\d+)$", code_column)
        if not match:
            return None
        suffix = match.group(1)
        candidates = [
            code_column.replace(f"Healthcare Provider Taxonomy Code_{suffix}", f"Healthcare Provider Primary Taxonomy Switch_{suffix}"),
            f"Healthcare Provider Primary Taxonomy Switch_{suffix}",
            f"npi__Healthcare Provider Primary Taxonomy Switch_{suffix}",
        ]
        return first_existing_column(df, candidates)

    def _resolve_primary_taxonomy_code_any(self, df: pd.DataFrame) -> pd.Series:
        taxonomy_cols = self._taxonomy_code_columns(df)
        if not taxonomy_cols:
            return pd.Series("", index=df.index, dtype="string")

        code = pd.Series("", index=df.index, dtype="string")
        for code_column in taxonomy_cols:
            switch_column = self._primary_switch_for_code_column(df, code_column)
            if not switch_column:
                continue
            switch = df[switch_column].astype("string").fillna("").str.upper().str.strip()
            values = df[code_column].astype("string").fillna("").str.strip()
            code = code.where(code != "", values.where(switch == "Y", ""))

        fallback = coalesce_columns(df, taxonomy_cols)
        if (code == "").all():
            return fallback.fillna("")
        return code.where(code != "", fallback).fillna("")

    def _build_taxonomy_code_all(self, df: pd.DataFrame, primary_code: pd.Series) -> pd.Series:
        # Keep this intentionally bounded; the comparison baseline only needs a stable audit field, not every raw code.
        taxonomy_cols = [column for column in self._taxonomy_code_columns(df) if column != "taxonomy_code_primary"][:5]
        out = primary_code.astype("string").fillna("").str.strip()
        for column in taxonomy_cols:
            values = df[column].astype("string").fillna("").str.strip()
            mask = values.ne("") & out.eq("")
            out = out.where(~mask, values)
        return out.fillna("")

    def _specialty_lookup_for_type(self, source_types: list[str]) -> pd.DataFrame:
        if self.specialty_rollup.empty:
            return pd.DataFrame()
        source_type_set = {item.lower() for item in source_types}
        work = self.specialty_rollup[
            self.specialty_rollup["source_type"].isin(source_type_set) | self.specialty_rollup["source_type"].eq("")
        ].copy()
        if work.empty:
            return work
        return work.drop_duplicates("source_key", keep="first").set_index("source_key")

    def _ensure_specialty_rollup_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        primary_code = self._resolve_primary_taxonomy_code_any(out)
        out["taxonomy_code_primary"] = primary_code.fillna("")
        direct_desc = coalesce_columns(
            out,
            [
                "Healthcare Provider Primary Taxonomy Description",
                "npi__Healthcare Provider Primary Taxonomy Description",
                "taxonomy_desc_primary",
                "taxonomy_desc",
            ],
        )
        mapped_desc = primary_code.map(lambda code: self.taxonomy_lookup.get(str(code or "").strip(), str(code or "").strip()))
        primary_desc = direct_desc.where(direct_desc.ne(""), mapped_desc).astype("string").fillna("").str.strip()

        code_lookup = self._specialty_lookup_for_type(["taxonomy_code"])
        desc_lookup = self._specialty_lookup_for_type(["taxonomy_desc"])

        code_key = primary_code.map(normalize_text)
        desc_key = primary_desc.map(normalize_text)
        code_like_desc = primary_desc.astype("string").fillna("").str.strip().str.upper().str.match(
            r"^(?=.*\d)[0-9A-Z]{10}$",
            na=False,
        )
        code_like_desc_key = primary_desc.where(code_like_desc, "").map(normalize_text)

        def mapped_from(lookup: pd.DataFrame, key: pd.Series, column: str) -> pd.Series:
            if lookup.empty or column not in lookup.columns:
                return pd.Series("", index=out.index, dtype="string")
            return key.map(lookup[column]).fillna("").astype("string")

        code_specialty = mapped_from(code_lookup, code_key, "normalized_specialty")
        desc_specialty = mapped_from(desc_lookup, desc_key, "normalized_specialty")
        code_desc_specialty = mapped_from(code_lookup, code_like_desc_key, "normalized_specialty")

        code_group = mapped_from(code_lookup, code_key, "specialty_group")
        desc_group = mapped_from(desc_lookup, desc_key, "specialty_group")
        code_desc_group = mapped_from(code_lookup, code_like_desc_key, "specialty_group")

        code_provider_type = mapped_from(code_lookup, code_key, "provider_type_group")
        desc_provider_type = mapped_from(desc_lookup, desc_key, "provider_type_group")
        code_desc_provider_type = mapped_from(code_lookup, code_like_desc_key, "provider_type_group")

        normalized = code_specialty.where(code_specialty.ne(""), desc_specialty)
        normalized = normalized.where(normalized.ne(""), code_desc_specialty)
        normalized = normalized.where(normalized.ne(""), primary_desc)
        group = code_group.where(code_group.ne(""), desc_group)
        group = group.where(group.ne(""), code_desc_group)
        group = group.where(group.ne(""), normalized)
        provider_type = code_provider_type.where(code_provider_type.ne(""), desc_provider_type)
        provider_type = provider_type.where(provider_type.ne(""), code_desc_provider_type)

        source = pd.Series("", index=out.index, dtype="string")
        source = source.where(code_specialty.eq(""), "taxonomy_code_rollup")
        source = source.where((source.ne("")) | desc_specialty.eq(""), "taxonomy_desc_rollup")
        source = source.where((source.ne("")) | code_desc_specialty.eq(""), "taxonomy_desc_code_rollup")
        source = source.where(source.ne(""), "taxonomy_desc_primary")

        code_rollup_mask = code_specialty.ne("")
        code_desc_mask = code_like_desc & code_desc_specialty.ne("")
        primary_desc = primary_desc.where(~code_rollup_mask, code_specialty)
        primary_desc = primary_desc.where(~code_desc_mask, code_desc_specialty)

        out["taxonomy_desc_primary"] = primary_desc.fillna("")
        out["taxonomy_codes_all"] = self._build_taxonomy_code_all(out, primary_code)
        out["taxonomy_descs_all"] = out["taxonomy_codes_all"].map(
            lambda value: "|".join(
                dict.fromkeys(
                    self.taxonomy_lookup.get(item.strip(), item.strip())
                    for item in str(value or "").split("|")
                    if item.strip()
                )
            )
        )
        out["specialty_normalized"] = normalized.fillna("").astype("string").str.strip()
        out["specialty_group"] = group.fillna("").astype("string").str.strip()
        out["provider_type_group"] = provider_type.fillna("").astype("string").str.strip()
        out["specialty_rollup_source"] = source.fillna("").astype("string").str.strip()
        return out

    @staticmethod
    def _combine_two_unique(left: pd.Series, right: pd.Series, sep: str = "|") -> pd.Series:
        left = left.astype("string").fillna("").str.strip()
        right = right.astype("string").fillna("").str.strip()
        both = left.ne("") & right.ne("") & left.ne(right)
        out = left.where(left.ne(""), right)
        out = out.where(~both, left + sep + right)
        return out.fillna("")

    def _address_key_series(self, df: pd.DataFrame, prefix: str) -> pd.Series:
        address1 = f"{prefix}_address_1"
        address2 = f"{prefix}_address_2"
        city = f"{prefix}_city"
        state = f"{prefix}_state"
        zip_col = f"{prefix}_zip5"
        if address1 not in df.columns and city not in df.columns and state not in df.columns and zip_col not in df.columns:
            return pd.Series("", index=df.index, dtype="string")
        line1 = (
            self._normalize_address_line1_series(df[address1])
            if address1 in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        line2 = (
            self._normalize_address_line1_series(df[address2])
            if address2 in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        city_value = self._normalize_series(df[city]) if city in df.columns else pd.Series("", index=df.index, dtype="string")
        state_value = (
            df[state].astype("string").fillna("").str.upper().str.strip()
            if state in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        zip_value = df[zip_col].map(zip5) if zip_col in df.columns else pd.Series("", index=df.index, dtype="string")
        key = line1 + "|" + line2 + "|" + city_value + "|" + state_value + "|" + zip_value
        has_value = line1.ne("") | line2.ne("") | city_value.ne("") | state_value.ne("") | zip_value.ne("")
        return key.where(has_value, "").fillna("")

    def _zip_state_key_series(self, df: pd.DataFrame, prefix: str) -> pd.Series:
        state = f"{prefix}_state"
        zip_col = f"{prefix}_zip5"
        if state not in df.columns and zip_col not in df.columns:
            return pd.Series("", index=df.index, dtype="string")
        state_value = (
            df[state].astype("string").fillna("").str.upper().str.strip()
            if state in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        zip_value = df[zip_col].map(zip5) if zip_col in df.columns else pd.Series("", index=df.index, dtype="string")
        key = state_value + "|" + zip_value
        return key.where(state_value.ne("") | zip_value.ne(""), "").fillna("")

    def _address_line1_key_series(self, df: pd.DataFrame, prefix: str) -> pd.Series:
        address1 = f"{prefix}_address_1"
        if address1 not in df.columns:
            return pd.Series("", index=df.index, dtype="string")
        return self._normalize_address_line1_series(df[address1]).fillna("")

    def _city_state_key_series(self, df: pd.DataFrame, prefix: str) -> pd.Series:
        city = f"{prefix}_city"
        state = f"{prefix}_state"
        if city not in df.columns and state not in df.columns:
            return pd.Series("", index=df.index, dtype="string")
        city_value = self._normalize_series(df[city]) if city in df.columns else pd.Series("", index=df.index, dtype="string")
        state_value = (
            df[state].astype("string").fillna("").str.upper().str.strip()
            if state in df.columns
            else pd.Series("", index=df.index, dtype="string")
        )
        key = city_value + "|" + state_value
        return key.where(city_value.ne("") | state_value.ne(""), "").fillna("")

    def _street_number_zip_key_series(self, df: pd.DataFrame, prefix: str) -> pd.Series:
        line1 = self._address_line1_key_series(df, prefix)
        zip_col = f"{prefix}_zip5"
        zip_value = df[zip_col].map(zip5) if zip_col in df.columns else pd.Series("", index=df.index, dtype="string")
        street_number = line1.str.extract(r"^(\d+)", expand=False).fillna("").astype("string")
        key = street_number + "|" + zip_value
        return key.where(street_number.ne("") | zip_value.ne(""), "").fillna("")

    def _ensure_comparison_fields(
        self,
        df: pd.DataFrame,
        *,
        refresh_specialty: bool = True,
        include_mapped_contact: bool = True,
    ) -> pd.DataFrame:
        out = self._ensure_specialty_rollup_fields(df) if refresh_specialty else df.copy()

        provider_phone = (
            out["provider_practice_phone"].map(lambda value: safe_phone(value) or "")
            if "provider_practice_phone" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        mapped_phone = (
            out["mapped_practice_phone"].map(lambda value: safe_phone(value) or "")
            if include_mapped_contact and "mapped_practice_phone" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_phone_set"] = self._combine_two_unique(provider_phone, mapped_phone)

        provider_address = self._address_key_series(out, "provider_practice")
        mapped_address = (
            self._address_key_series(out, "mapped_practice")
            if include_mapped_contact
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_address_set"] = self._combine_two_unique(provider_address, mapped_address, sep="||")

        provider_zip_state = self._zip_state_key_series(out, "provider_practice")
        mapped_zip_state = (
            self._zip_state_key_series(out, "mapped_practice")
            if include_mapped_contact
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_zip_state_set"] = self._combine_two_unique(provider_zip_state, mapped_zip_state)

        provider_line1 = self._address_line1_key_series(out, "provider_practice")
        mapped_line1 = (
            self._address_line1_key_series(out, "mapped_practice")
            if include_mapped_contact
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_address_line1_set"] = self._combine_two_unique(provider_line1, mapped_line1)

        provider_city_state = self._city_state_key_series(out, "provider_practice")
        mapped_city_state = (
            self._city_state_key_series(out, "mapped_practice")
            if include_mapped_contact
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_city_state_set"] = self._combine_two_unique(provider_city_state, mapped_city_state)

        provider_street_number_zip = self._street_number_zip_key_series(out, "provider_practice")
        mapped_street_number_zip = (
            self._street_number_zip_key_series(out, "mapped_practice")
            if include_mapped_contact
            else pd.Series("", index=out.index, dtype="string")
        )
        out["comparison_street_number_zip_set"] = self._combine_two_unique(
            provider_street_number_zip,
            mapped_street_number_zip,
        )

        mapped_system = self._has_value(out, ["mapped_system_id", "mapped_system_name"])
        mapped_clinic = self._has_value(out, ["mapped_clinic_id", "mapped_clinic_name"])
        specialty = self._has_value(out, ["specialty_normalized", "specialty_group", "taxonomy_desc_primary"])
        contact = self._has_value(out, ["comparison_phone_set", "comparison_address_set"])
        tier = pd.Series("low", index=out.index, dtype="string")
        tier = tier.where(mapped_system, "unmapped")
        medium = mapped_system & specialty & contact
        tier = tier.where(~medium, "medium")
        mapping_tier = (
            out["mapping_confidence_tier"].astype("string").fillna("").str.strip()
            if "mapping_confidence_tier" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        ambiguous = (
            out["primary_is_ambiguous"].fillna(False).astype(bool)
            if "primary_is_ambiguous" in out.columns
            else pd.Series(False, index=out.index)
        )
        high = medium & mapped_clinic & mapping_tier.isin(["high", "medium"]) & ~ambiguous
        tier = tier.where(~high, "high")
        out["comparison_readiness_tier"] = tier.fillna("low")
        return out

    def _ensure_freshness_and_provenance(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        out = df.copy()
        orgs_path = self.find_latest_processed_orgs()
        npi_path = self.find_latest_npi_individuals()
        out["ppef_snapshot_date"] = date_str
        out["pecos_snapshot_date"] = self._snapshot_date(orgs_path)
        out["npi_registry_snapshot_date"] = self._snapshot_date(npi_path)

        npi_available = bool(npi_path and npi_path.exists())
        out["provider_identity_source"] = ""
        if "provider_id" in out.columns:
            has_provider_id = out["provider_id"].astype("string").fillna("").str.strip().ne("")
            out.loc[has_provider_id, "provider_identity_source"] = "ppef_enrollment"
        if "npi" in out.columns:
            has_npi = out["npi"].astype("string").fillna("").str.strip().ne("")
            out.loc[has_npi, "provider_identity_source"] = "ppef_enrollment_npi"

        out["specialty_source"] = ""
        specialty_mask = self._has_value(out, ["taxonomy_code_primary", "taxonomy_desc_primary"])
        out.loc[specialty_mask, "specialty_source"] = "npi_registry" if npi_available else "existing_processed"

        out["phone_source"] = ""
        phone_mask = self._has_value(out, ["provider_practice_phone"])
        out.loc[phone_mask, "phone_source"] = "npi_registry" if npi_available else "existing_processed"

        out["address_source"] = ""
        address_mask = self._has_value(
            out,
            [
                "provider_practice_address_1",
                "provider_practice_address_2",
                "provider_practice_city",
                "provider_practice_state",
                "provider_practice_zip5",
            ],
        )
        out.loc[address_mask, "address_source"] = "npi_registry" if npi_available else "existing_processed"

        out["clinic_source"] = ""
        clinic_mask = self._has_value(out, ["mapped_clinic_id", "mapped_clinic_name"])
        out.loc[clinic_mask, "clinic_source"] = "ppef_reassignment_pecos_primary"

        out["system_source"] = ""
        system_mask = self._has_value(out, ["mapped_system_id", "mapped_system_name"])
        out.loc[system_mask, "system_source"] = "ppef_reassignment_pecos_primary"
        return out

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
        out = self._ensure_specialty_rollup_fields(out)
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
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "candidate_clinic_count",
            "candidate_system_count",
            "primary_is_tied",
            "primary_is_ambiguous",
            "mapping_confidence_tier",
            "primary_selection_reason",
        ]:
            if column not in out.columns:
                out[column] = ""
        out = self._ensure_freshness_and_provenance(out, metadata.get("date") or "")
        out = self._ensure_comparison_fields(out)

        metadata = add_mode_completion(
            metadata,
            "npi_enrichment",
            time.time() - start,
            len(out),
            [
                "provider_id",
                "npi",
                "enrollment_id",
                "provider_full_name",
                "taxonomy_code_primary",
                "taxonomy_desc_primary",
                "taxonomy_codes_all",
                "taxonomy_descs_all",
                "specialty_normalized",
                "specialty_group",
                "provider_type_group",
                "specialty_rollup_source",
                "comparison_phone_set",
                "comparison_address_set",
                "comparison_zip_state_set",
                "comparison_address_line1_set",
                "comparison_city_state_set",
                "comparison_street_number_zip_set",
                "comparison_readiness_tier",
                "ppef_snapshot_date",
                "pecos_snapshot_date",
                "npi_registry_snapshot_date",
                "specialty_source",
                "phone_source",
                "address_source",
            ],
        )
        return out, metadata

    @staticmethod
    def _address_key(address1, city, state, postal) -> str:
        return "|".join([combine_address(address1).lower().strip(), str(city or "").strip().lower(), str(state or "").strip().lower(), zip5(postal)])

    @staticmethod
    def _row_completeness(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        score = pd.Series(0, index=df.index, dtype="int64")
        for column in columns:
            if column not in df.columns:
                continue
            score = score + df[column].astype("string").fillna("").str.strip().ne("").astype("int64")
        return score

    @staticmethod
    def _first_non_empty(series: pd.Series) -> str:
        values = series.astype("string").fillna("").str.strip()
        values = values[values != ""]
        return str(values.iloc[0]) if not values.empty else ""

    @staticmethod
    def _normalize_series(series: pd.Series) -> pd.Series:
        return (
            series.astype("string")
            .fillna("")
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    @staticmethod
    def _normalize_address_line1_series(series: pd.Series) -> pd.Series:
        # Keep this vectorized because clinic mapping evaluates millions of candidate edges.
        normalized = (
            series.astype("string")
            .fillna("")
            .str.lower()
            .str.replace(r"\b(?:apt|apartment|bldg|building|dept|department|fl|floor|lot|rm|room|ste|suite|unit)\b\.?\s*[a-z0-9-]*", " ", regex=True)
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\bavenue\b", "ave", regex=True)
            .str.replace(r"\bboulevard\b", "blvd", regex=True)
            .str.replace(r"\bcircle\b", "cir", regex=True)
            .str.replace(r"\bcourt\b", "ct", regex=True)
            .str.replace(r"\bdrive\b", "dr", regex=True)
            .str.replace(r"\bhighway\b", "hwy", regex=True)
            .str.replace(r"\blane\b", "ln", regex=True)
            .str.replace(r"\bparkway\b", "pkwy", regex=True)
            .str.replace(r"\bplace\b", "pl", regex=True)
            .str.replace(r"\broad\b", "rd", regex=True)
            .str.replace(r"\bsquare\b", "sq", regex=True)
            .str.replace(r"\bstreet\b", "st", regex=True)
            .str.replace(r"\bterrace\b", "ter", regex=True)
            .str.replace(r"\bnorth\b", "n", regex=True)
            .str.replace(r"\bsouth\b", "s", regex=True)
            .str.replace(r"\beast\b", "e", regex=True)
            .str.replace(r"\bwest\b", "w", regex=True)
            .str.replace(r"\bnortheast\b", "ne", regex=True)
            .str.replace(r"\bnorthwest\b", "nw", regex=True)
            .str.replace(r"\bsoutheast\b", "se", regex=True)
            .str.replace(r"\bsouthwest\b", "sw", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return normalized

    def _resolve_reassignment_schema(self, df: pd.DataFrame) -> dict[str, Optional[str]]:
        return {
            "provider_npi": first_existing_column(df, ["INDIVIDUAL_NPI", "INDVL_NPI", "REASGN_BNFTS_INDVL_NPI", "NPI"])
            or find_column_by_tokens(df, required_tokens=["npi"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org", "recv"]),
            "provider_enrollment_id": first_existing_column(df, ["REASGN_BNFT_ENRLMT_ID", "INDIVIDUAL_ENROLLMENT_ID", "INDVL_ENRLMT_ID", "ENRLMT_ID"])
            or find_column_by_tokens(df, required_tokens=["enrlmt"], any_tokens=["individual", "indvl", "provider"], exclude_tokens=["org", "recv"]),
            "org_npi": first_existing_column(df, ["ORG_NPI", "RCVG_NPI", "REASSIGN_TO_NPI"])
            or find_column_by_tokens(df, required_tokens=["npi"], any_tokens=["org", "recv", "reassign"], exclude_tokens=["individual", "indvl", "provider"]),
            "org_enrollment_id": first_existing_column(df, ["RCV_BNFT_ENRLMT_ID", "ORG_ENROLLMENT_ID", "ORG_ENRLMT_ID", "RCVG_ENRLMT_ID", "REASSIGN_TO_ENRLMT_ID"])
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

    @staticmethod
    def _empty_links_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "provider_id",
                "clinic_id",
                "clinic_name",
                "system_id",
                "system_name",
                "relationship_source",
                "relationship_start_date",
                "relationship_end_date",
                "is_active_relationship",
                "recency_score",
                "continuity_score",
                "address_match_score",
                "taxonomy_match_score",
                "total_affiliation_score",
                "selected_as_primary",
                "mapping_method",
                "shared_address_provider_count",
                "billing_affiliation_recency_rank",
                "relationship_duration_days",
                "raw_address_key",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
                "system_rollup_quality_warning",
                "clinic_rollup_quality_warning",
                "signal_blocked_generic_system",
                "blocked_generic_penalty",
            ]
        )

    def _load_practice_support(self, practice_path: Optional[Path]) -> pd.DataFrame:
        if practice_path is None or not practice_path.exists():
            return pd.DataFrame(
                columns=[
                    "enrollment_id",
                    "practice_support_city",
                    "practice_support_state",
                    "practice_support_zip5",
                ]
            )
        practice_df = pd.read_parquet(practice_path, columns=["ENRLMT_ID", "CITY_NAME", "STATE_CD", "ZIP_CD"]).copy()
        practice_df = practice_df.rename(
            columns={
                "ENRLMT_ID": "enrollment_id",
                "CITY_NAME": "practice_support_city",
                "STATE_CD": "practice_support_state",
                "ZIP_CD": "practice_support_zip5",
            }
        )
        for column in ["enrollment_id", "practice_support_city", "practice_support_state", "practice_support_zip5"]:
            practice_df[column] = practice_df[column].astype("string").fillna("").str.strip()
        practice_df["practice_support_zip5"] = practice_df["practice_support_zip5"].map(zip5)
        practice_df = practice_df[practice_df["enrollment_id"] != ""]
        if practice_df.empty:
            return practice_df
        practice_df["support_completeness"] = (
            practice_df["practice_support_city"].ne("").astype("int64")
            + practice_df["practice_support_state"].ne("").astype("int64")
            + practice_df["practice_support_zip5"].ne("").astype("int64")
        )
        practice_df = (
            practice_df.sort_values(
                by=["support_completeness", "practice_support_zip5", "practice_support_city", "practice_support_state"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["enrollment_id"], keep="first")
            .drop(columns=["support_completeness"], errors="ignore")
            .copy()
        )
        return practice_df

    def _build_org_enrollment_bridge(self, org_df: pd.DataFrame) -> pd.DataFrame:
        work = org_df.copy()
        for column in [
            "source_enrollment_id",
            "clinic_id",
            "clinic_name",
            "system_id",
            "system_name",
            "org_npi",
            "org_entity_id",
            "practice_address_1",
            "practice_address_2",
            "practice_city",
            "practice_state",
            "practice_zip5",
            "practice_phone",
            "taxonomy_desc_primary",
            "system_alias_action",
            "system_alias_reason",
            "system_rollup_quality_warning",
            "clinic_rollup_quality_warning",
        ]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        if "system_alias_blocked_generic" not in work.columns:
            work["system_alias_blocked_generic"] = False
        work["system_alias_blocked_generic"] = work["system_alias_blocked_generic"].fillna(False).astype(bool)
        if "is_hospital" not in work.columns:
            work["is_hospital"] = False
        work["is_hospital"] = work["is_hospital"].fillna(False).astype(bool)
        work["practice_zip5"] = work["practice_zip5"].map(zip5)
        work["practice_phone"] = work["practice_phone"].map(lambda v: safe_phone(v) or "")
        work["clinic_name_canonical"] = work["clinic_name"].map(canonical_entity_name)
        work["system_name_canonical"] = work["system_name"].map(canonical_entity_name)
        work["org_row_completeness"] = self._row_completeness(
            work,
            [
                "clinic_name",
                "system_name",
                "practice_address_1",
                "practice_city",
                "practice_state",
                "practice_zip5",
                "practice_phone",
                "org_npi",
                "org_entity_id",
                "taxonomy_desc_primary",
            ],
        )
        work["clinic_name_len"] = work["clinic_name"].str.len().fillna(0).astype("int64")
        work["system_name_len"] = work["system_name"].str.len().fillna(0).astype("int64")

        clinic_best = (
            work.sort_values(
                by=["org_row_completeness", "clinic_name_len", "is_hospital", "clinic_name"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["clinic_id"], keep="first")
            .copy()
        )
        system_best = (
            work.sort_values(
                by=["org_row_completeness", "system_name_len", "is_hospital", "system_name"],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["system_id"], keep="first")
            .copy()
        )
        clinic_rep = clinic_best[
            [
                "clinic_id",
                "clinic_name",
                "clinic_name_canonical",
                "practice_address_1",
                "practice_address_2",
                "practice_city",
                "practice_state",
                "practice_zip5",
                "practice_phone",
                "taxonomy_desc_primary",
                "is_hospital",
                "clinic_rollup_quality_warning",
            ]
        ].rename(
            columns={
                "clinic_name": "clinic_name_rep",
                "clinic_name_canonical": "clinic_name_canonical_rep",
                "practice_address_1": "practice_address_1_rep",
                "practice_address_2": "practice_address_2_rep",
                "practice_city": "practice_city_rep",
                "practice_state": "practice_state_rep",
                "practice_zip5": "practice_zip5_rep",
                "practice_phone": "practice_phone_rep",
                "taxonomy_desc_primary": "taxonomy_desc_primary_rep",
                "is_hospital": "is_hospital_rep",
                "clinic_rollup_quality_warning": "clinic_rollup_quality_warning_rep",
            }
        )
        system_rep = system_best[
            [
                "system_id",
                "system_name",
                "system_name_canonical",
                "org_entity_id",
                "org_npi",
                "is_hospital",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
                "system_rollup_quality_warning",
            ]
        ].rename(
            columns={
                "system_name": "system_name_rep",
                "system_name_canonical": "system_name_canonical_rep",
                "org_entity_id": "org_entity_id_rep",
                "org_npi": "org_npi_rep",
                "is_hospital": "is_hospital_system_rep",
                "system_alias_action": "system_alias_action_rep",
                "system_alias_reason": "system_alias_reason_rep",
                "system_alias_blocked_generic": "system_alias_blocked_generic_rep",
                "system_rollup_quality_warning": "system_rollup_quality_warning_rep",
            }
        )
        bridge = work[
            [
                "source_enrollment_id",
                "clinic_id",
                "system_id",
            ]
        ].drop_duplicates(subset=["source_enrollment_id"], keep="first")
        bridge = bridge.merge(clinic_rep, on="clinic_id", how="left")
        bridge = bridge.merge(system_rep, on="system_id", how="left")
        bridge = bridge.rename(
            columns={
                "clinic_name_rep": "clinic_name",
                "clinic_name_canonical_rep": "clinic_name_canonical",
                "practice_address_1_rep": "practice_address_1",
                "practice_address_2_rep": "practice_address_2",
                "practice_city_rep": "practice_city",
                "practice_state_rep": "practice_state",
                "practice_zip5_rep": "practice_zip5",
                "practice_phone_rep": "practice_phone",
                "taxonomy_desc_primary_rep": "org_taxonomy_desc_primary",
                "clinic_rollup_quality_warning_rep": "clinic_rollup_quality_warning",
                "system_name_rep": "system_name",
                "system_name_canonical_rep": "system_name_canonical",
                "org_entity_id_rep": "org_entity_id",
                "org_npi_rep": "org_npi",
                "system_alias_action_rep": "system_alias_action",
                "system_alias_reason_rep": "system_alias_reason",
                "system_alias_blocked_generic_rep": "system_alias_blocked_generic",
                "system_rollup_quality_warning_rep": "system_rollup_quality_warning",
            }
        )
        bridge["is_hospital"] = (
            bridge["is_hospital_rep"].fillna(False).astype(bool)
            | bridge["is_hospital_system_rep"].fillna(False).astype(bool)
        )
        bridge = bridge.drop(columns=["is_hospital_rep", "is_hospital_system_rep"], errors="ignore")
        for column in bridge.columns:
            if column in {"is_hospital", "system_alias_blocked_generic"}:
                continue
            bridge[column] = bridge[column].astype("string").fillna("").str.strip()
        bridge["system_alias_blocked_generic"] = bridge["system_alias_blocked_generic"].fillna(False).astype(bool)
        bridge["practice_zip5"] = bridge["practice_zip5"].map(zip5)
        bridge["practice_phone"] = bridge["practice_phone"].map(lambda v: safe_phone(v) or "")
        return bridge[bridge["source_enrollment_id"] != ""].copy()

    @staticmethod
    def _confidence_tier(signal_count: int, margin: float, candidate_system_count: int, top_score_tied: bool) -> str:
        if candidate_system_count == 1 and signal_count >= 1:
            return "high"
        if signal_count >= 4 and margin >= 1.0 and not top_score_tied:
            return "high"
        if signal_count >= 2 and margin >= 0.5:
            return "medium"
        if signal_count >= 1 and candidate_system_count <= 2 and not top_score_tied:
            return "medium"
        return "low"

    @staticmethod
    def _selection_reason(row: pd.Series) -> str:
        reasons: list[str] = ["reassignment"]
        if bool(row.get("signal_state_zip_match")):
            reasons.append("state_zip")
        if bool(row.get("signal_practice_support_state_zip_match")):
            reasons.append("practice_support_zip")
        if bool(row.get("signal_address_line1_match")):
            reasons.append("address")
        if bool(row.get("signal_phone_match")):
            reasons.append("phone")
        if bool(row.get("signal_org_name_strong")):
            reasons.append("org_name")
        if bool(row.get("signal_city_state_match")):
            reasons.append("city_state")
        if bool(row.get("signal_repeat_edge")):
            reasons.append("repeat_edge")
        return "+".join(reasons)

    def _write_mapping_quality_report(
        self,
        date_str: str,
        processed_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        best_df: pd.DataFrame,
    ) -> None:
        report_path = self.quality_report_path(date_str)
        json_path = self.quality_report_json_path(date_str)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        selected = best_df.copy()
        provider_count = int(processed_df["provider_id"].astype("string").fillna("").str.strip().replace("", pd.NA).nunique())
        mapped_provider_count = int(selected["provider_id"].astype("string").fillna("").str.strip().replace("", pd.NA).nunique())
        mapped_provider_pct = round((mapped_provider_count / provider_count) * 100, 2) if provider_count else 0.0

        tier_counts = selected["mapping_confidence_tier"].fillna("").replace("", "unknown").value_counts().to_dict()
        high_medium = int(tier_counts.get("high", 0) + tier_counts.get("medium", 0))
        high_medium_pct = round((high_medium / mapped_provider_count) * 100, 2) if mapped_provider_count else 0.0
        tied_count = int(selected["primary_is_tied"].fillna(False).astype(bool).sum())
        ambiguous_count = int(selected["primary_is_ambiguous"].fillna(False).astype(bool).sum())
        tied_pct = round((tied_count / mapped_provider_count) * 100, 2) if mapped_provider_count else 0.0
        ambiguous_pct = round((ambiguous_count / mapped_provider_count) * 100, 2) if mapped_provider_count else 0.0

        signal_columns = [
            "signal_state_zip_match",
            "signal_practice_support_state_zip_match",
            "signal_city_state_match",
            "signal_practice_support_city_state_match",
            "signal_address_line1_match",
            "signal_phone_match",
            "signal_rare_phone_match",
            "signal_org_name_strong",
            "signal_repeat_edge",
        ]
        signal_counts = {
            column: int(selected[column].fillna(False).astype(bool).sum())
            for column in signal_columns
            if column in selected.columns
        }
        warnings = []
        if mapped_provider_pct < 87.0:
            warnings.append("mapped_unique_provider_coverage_below_87_pct")
        if tied_pct > 15.0:
            warnings.append("tied_primary_rate_above_15_pct")
        if high_medium_pct < 70.0:
            warnings.append("high_medium_confidence_below_70_pct")
        if signal_counts.get("signal_org_name_strong", 0) == 0:
            warnings.append("org_name_signal_zero_or_unavailable")

        rows = [
            {"metric": "provider_count", "value": provider_count},
            {"metric": "mapped_provider_count", "value": mapped_provider_count},
            {"metric": "mapped_provider_pct", "value": mapped_provider_pct},
            {"metric": "candidate_row_count", "value": int(len(candidate_df))},
            {"metric": "candidate_provider_count", "value": int(candidate_df["provider_id"].nunique())},
            {"metric": "high_medium_confidence_pct", "value": high_medium_pct},
            {"metric": "tied_primary_count", "value": tied_count},
            {"metric": "tied_primary_pct", "value": tied_pct},
            {"metric": "ambiguous_primary_count", "value": ambiguous_count},
            {"metric": "ambiguous_primary_pct", "value": ambiguous_pct},
        ]
        rows.extend({"metric": f"confidence_tier_{tier}", "value": int(count)} for tier, count in sorted(tier_counts.items()))
        rows.extend({"metric": column, "value": count} for column, count in sorted(signal_counts.items()))
        rows.extend({"metric": f"warning_{warning}", "value": 1} for warning in warnings)
        pd.DataFrame(rows).to_csv(report_path, index=False)

        top_ambiguous = (
            selected[selected["primary_is_ambiguous"].fillna(False).astype(bool)]
            .groupby(["system_id", "system_name"], dropna=False)["provider_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(25)
            .reset_index(name="ambiguous_provider_count")
        )
        top_low_confidence = (
            selected[selected["mapping_confidence_tier"].fillna("").eq("low")]
            .groupby(["system_id", "system_name"], dropna=False)["provider_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(25)
            .reset_index(name="low_confidence_provider_count")
        )
        payload = {
            "date": date_str,
            "metrics": {row["metric"]: row["value"] for row in rows if not str(row["metric"]).startswith("warning_")},
            "warnings": warnings,
            "top_ambiguous_systems": top_ambiguous.to_dict(orient="records"),
            "top_low_confidence_systems": top_low_confidence.to_dict(orient="records"),
            "notes": [
                "ORG_NAME is empty in the current PPEF individual snapshot; org-name signals require a populated source field.",
                "Reassignment extracts currently provide enrollment IDs only, so temporal recency/continuity remains unavailable.",
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2, default=str))
        LOGGER.info("Wrote PPEF/PECOS mapping quality report: %s", report_path)

    def _write_system_quality_report(self, date_str: str, selected_df: pd.DataFrame) -> None:
        path = self.system_quality_report_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "system_id",
            "system_name",
            "mapped_provider_count",
            "high_confidence_count",
            "medium_confidence_count",
            "low_confidence_count",
            "tied_primary_count",
            "ambiguous_primary_count",
            "avg_mapping_score",
            "avg_mapping_margin_score",
            "avg_mapping_signal_count",
            "avg_candidate_clinic_count",
            "avg_candidate_system_count",
            "signal_state_zip_match_count",
            "signal_address_line1_match_count",
            "signal_phone_match_count",
            "signal_org_name_strong_count",
            "blocked_generic_system_count",
            "avg_blocked_generic_penalty",
            "system_alias_action",
            "system_alias_blocked_generic",
            "system_rollup_quality_warning",
            "review_priority_score",
        ]
        if selected_df.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return

        work = selected_df.copy()
        for column in ["system_id", "system_name", "mapping_confidence_tier"]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        for column in [
            "primary_is_tied",
            "primary_is_ambiguous",
            "signal_state_zip_match",
            "signal_address_line1_match",
            "signal_phone_match",
            "signal_org_name_strong",
            "system_alias_blocked_generic",
        ]:
            if column not in work.columns:
                work[column] = False
            work[column] = work[column].fillna(False).astype(bool)
        for column in ["system_alias_action", "system_rollup_quality_warning"]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        for column in [
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "candidate_clinic_count",
            "candidate_system_count",
            "blocked_generic_penalty",
        ]:
            if column not in work.columns:
                work[column] = 0
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
        work["is_high"] = work["mapping_confidence_tier"].eq("high")
        work["is_medium"] = work["mapping_confidence_tier"].eq("medium")
        work["is_low"] = work["mapping_confidence_tier"].eq("low")

        grouped = (
            work.groupby(["system_id", "system_name"], dropna=False)
            .agg(
                mapped_provider_count=("provider_id", "nunique"),
                high_confidence_count=("is_high", "sum"),
                medium_confidence_count=("is_medium", "sum"),
                low_confidence_count=("is_low", "sum"),
                tied_primary_count=("primary_is_tied", "sum"),
                ambiguous_primary_count=("primary_is_ambiguous", "sum"),
                avg_mapping_score=("mapping_score", "mean"),
                avg_mapping_margin_score=("mapping_margin_score", "mean"),
                avg_mapping_signal_count=("mapping_signal_count", "mean"),
                avg_candidate_clinic_count=("candidate_clinic_count", "mean"),
                avg_candidate_system_count=("candidate_system_count", "mean"),
                signal_state_zip_match_count=("signal_state_zip_match", "sum"),
                signal_address_line1_match_count=("signal_address_line1_match", "sum"),
                signal_phone_match_count=("signal_phone_match", "sum"),
                signal_org_name_strong_count=("signal_org_name_strong", "sum"),
                blocked_generic_system_count=("system_alias_blocked_generic", "sum"),
                avg_blocked_generic_penalty=("blocked_generic_penalty", "mean"),
                system_alias_action=("system_alias_action", lambda s: " | ".join(s[s != ""].drop_duplicates().head(5))),
                system_alias_blocked_generic=("system_alias_blocked_generic", "max"),
                system_rollup_quality_warning=("system_rollup_quality_warning", lambda s: " | ".join(s[s != ""].drop_duplicates().head(5))),
            )
            .reset_index()
        )
        grouped["review_priority_score"] = (
            grouped["low_confidence_count"].astype("int64")
            + grouped["ambiguous_primary_count"].astype("int64") * 2
            + grouped["tied_primary_count"].astype("int64")
            + grouped["blocked_generic_system_count"].astype("int64") * 3
        )
        grouped = grouped.sort_values(
            ["review_priority_score", "mapped_provider_count", "system_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        for column in grouped.columns:
            if column.startswith("avg_"):
                grouped[column] = grouped[column].round(3)
        grouped[columns].to_csv(path, index=False)
        LOGGER.info("Wrote PPEF/PECOS system quality report: %s", path)

    def _write_field_provenance_summary(self, date_str: str, processed_df: pd.DataFrame) -> None:
        path = self.field_provenance_summary_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        total_providers = int(processed_df["provider_id"].astype("string").fillna("").str.strip().replace("", pd.NA).nunique())
        field_specs = [
            ("provider_identity", "provider_identity_source", ["provider_id", "npi", "enrollment_id"]),
            ("specialty", "specialty_source", ["taxonomy_code_primary", "taxonomy_desc_primary"]),
            ("phone", "phone_source", ["provider_practice_phone"]),
            (
                "address",
                "address_source",
                [
                    "provider_practice_address_1",
                    "provider_practice_city",
                    "provider_practice_state",
                    "provider_practice_zip5",
                ],
            ),
            ("clinic", "clinic_source", ["mapped_clinic_id", "mapped_clinic_name"]),
            ("system", "system_source", ["mapped_system_id", "mapped_system_name"]),
        ]
        rows = []
        for field_name, source_column, value_columns in field_specs:
            selected_columns = []
            for column in ["provider_id", source_column, *value_columns]:
                if column in processed_df.columns and column not in selected_columns:
                    selected_columns.append(column)
            work = processed_df[selected_columns].copy()
            if source_column not in work.columns:
                work[source_column] = ""
            work[source_column] = work[source_column].astype("string").fillna("").str.strip().replace("", "missing")
            work["_field_populated"] = self._has_value(work, value_columns)
            grouped = (
                work.groupby(source_column, dropna=False)
                .agg(
                    row_count=("provider_id", "size"),
                    provider_count=("provider_id", lambda values: values.astype("string").fillna("").str.strip().replace("", pd.NA).nunique()),
                    populated_row_count=("_field_populated", "sum"),
                )
                .reset_index()
                .rename(columns={source_column: "source"})
            )
            grouped["field_name"] = field_name
            grouped["provider_pct_of_total"] = grouped["provider_count"].map(
                lambda value: round((float(value) / total_providers) * 100, 2) if total_providers else 0.0
            )
            rows.append(grouped[["field_name", "source", "row_count", "provider_count", "populated_row_count", "provider_pct_of_total"]])
        summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        summary.to_csv(path, index=False)
        LOGGER.info("Wrote PPEF/PECOS field provenance summary: %s", path)

    def _write_specialty_rollup_audit(self, date_str: str, processed_df: pd.DataFrame) -> None:
        path = self.specialty_rollup_audit_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "taxonomy_code_primary",
            "taxonomy_desc_primary",
            "specialty_normalized",
            "specialty_group",
            "provider_type_group",
            "specialty_rollup_source",
            "row_count",
            "provider_count",
        ]
        if processed_df.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        work = processed_df.copy()
        for column in columns[:-2]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        grouped = (
            work.groupby(columns[:-2], dropna=False)
            .agg(
                row_count=("provider_id", "size"),
                provider_count=("provider_id", lambda values: values.astype("string").fillna("").str.strip().replace("", pd.NA).nunique()),
            )
            .reset_index()
            .sort_values(["provider_count", "row_count"], ascending=[False, False], kind="mergesort")
        )
        grouped[columns].to_csv(path, index=False)
        LOGGER.info("Wrote PPEF specialty rollup audit: %s", path)

    def _write_unresolved_taxonomy_audit(self, date_str: str, processed_df: pd.DataFrame) -> None:
        path = self.unresolved_taxonomy_audit_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "taxonomy_code_primary",
            "taxonomy_desc_primary",
            "specialty_normalized",
            "specialty_group",
            "provider_type_group",
            "specialty_rollup_source",
            "unresolved_reason",
            "row_count",
            "provider_count",
            "sample_provider_names",
        ]
        if processed_df.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return

        work = processed_df.copy()
        for column in columns[:6]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        if "provider_full_name" not in work.columns:
            work["provider_full_name"] = ""
        work["provider_full_name"] = work["provider_full_name"].astype("string").fillna("").str.strip()

        code_like = r"^(?=.*\d)[0-9A-Z]{10}$"
        unresolved_desc = work["taxonomy_desc_primary"].str.upper().str.match(code_like, na=False)
        unresolved_specialty = work["specialty_normalized"].str.upper().str.match(code_like, na=False)
        unresolved_group = work["specialty_group"].str.upper().str.match(code_like, na=False)
        mask = work["taxonomy_code_primary"].ne("") & (unresolved_desc | unresolved_specialty | unresolved_group)
        work = work.loc[mask].copy()
        if work.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            LOGGER.info("Wrote PPEF unresolved taxonomy audit: %s", path)
            return

        work["unresolved_reason"] = ""
        work.loc[unresolved_desc, "unresolved_reason"] = "taxonomy_desc_primary_code_like"
        work.loc[unresolved_specialty, "unresolved_reason"] = (
            work.loc[unresolved_specialty, "unresolved_reason"].where(
                work.loc[unresolved_specialty, "unresolved_reason"].eq(""),
                work.loc[unresolved_specialty, "unresolved_reason"] + "|",
            )
            + "specialty_normalized_code_like"
        )
        work.loc[unresolved_group, "unresolved_reason"] = (
            work.loc[unresolved_group, "unresolved_reason"].where(
                work.loc[unresolved_group, "unresolved_reason"].eq(""),
                work.loc[unresolved_group, "unresolved_reason"] + "|",
            )
            + "specialty_group_code_like"
        )

        def sample_names(values: pd.Series) -> str:
            sample = values.astype("string").fillna("").str.strip()
            sample = sample[sample.ne("")].drop_duplicates().head(5)
            return " | ".join(sample.tolist())

        grouped = (
            work.groupby(columns[:7], dropna=False)
            .agg(
                row_count=("provider_id", "size"),
                provider_count=("provider_id", lambda values: values.astype("string").fillna("").str.strip().replace("", pd.NA).nunique()),
                sample_provider_names=("provider_full_name", sample_names),
            )
            .reset_index()
            .sort_values(["provider_count", "row_count"], ascending=[False, False], kind="mergesort")
        )
        grouped[columns].to_csv(path, index=False)
        LOGGER.info("Wrote PPEF unresolved taxonomy audit: %s", path)

    def _write_comparison_readiness_summary(self, date_str: str, processed_df: pd.DataFrame) -> None:
        path = self.comparison_readiness_summary_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "comparison_readiness_tier",
            "mapping_confidence_tier",
            "row_count",
            "provider_count",
            "mapped_system_count",
            "mapped_clinic_count",
            "specialty_count",
            "phone_count",
            "address_count",
        ]
        if processed_df.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        work = processed_df.copy()
        for column in ["comparison_readiness_tier", "mapping_confidence_tier"]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip().replace("", "unknown")
        work["_mapped_system"] = self._has_value(work, ["mapped_system_id", "mapped_system_name"])
        work["_mapped_clinic"] = self._has_value(work, ["mapped_clinic_id", "mapped_clinic_name"])
        work["_specialty"] = self._has_value(work, ["specialty_normalized", "specialty_group", "taxonomy_desc_primary"])
        work["_phone"] = self._has_value(work, ["comparison_phone_set", "provider_practice_phone"])
        work["_address"] = self._has_value(work, ["comparison_address_set", "provider_practice_address_1", "provider_practice_zip5"])
        grouped = (
            work.groupby(["comparison_readiness_tier", "mapping_confidence_tier"], dropna=False)
            .agg(
                row_count=("provider_id", "size"),
                provider_count=("provider_id", lambda values: values.astype("string").fillna("").str.strip().replace("", pd.NA).nunique()),
                mapped_system_count=("_mapped_system", "sum"),
                mapped_clinic_count=("_mapped_clinic", "sum"),
                specialty_count=("_specialty", "sum"),
                phone_count=("_phone", "sum"),
                address_count=("_address", "sum"),
            )
            .reset_index()
            .sort_values(["comparison_readiness_tier", "provider_count"], ascending=[True, False], kind="mergesort")
        )
        grouped[columns].to_csv(path, index=False)
        LOGGER.info("Wrote PPEF comparison readiness summary: %s", path)

    def _write_system_alias_summary(self, date_str: str) -> None:
        path = self.system_alias_summary_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "system_alias_action",
            "system_alias_blocked_generic",
            "system_alias_reason",
            "system_count",
            "org_row_count",
            "top_system_names",
        ]
        orgs_path = self.find_latest_processed_orgs()
        if not orgs_path or not orgs_path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        available = parquet_columns(orgs_path)
        read_columns = [
            column
            for column in [
                "system_id",
                "system_name",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
            ]
            if column in available
        ]
        if not read_columns or "system_alias_action" not in read_columns:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            return
        orgs = pd.read_parquet(orgs_path, columns=read_columns).copy()
        for column in ["system_id", "system_name", "system_alias_action", "system_alias_reason"]:
            if column not in orgs.columns:
                orgs[column] = ""
            orgs[column] = orgs[column].astype("string").fillna("").str.strip()
        if "system_alias_blocked_generic" not in orgs.columns:
            orgs["system_alias_blocked_generic"] = False
        orgs["system_alias_blocked_generic"] = orgs["system_alias_blocked_generic"].fillna(False).astype(bool)
        orgs["system_alias_action"] = orgs["system_alias_action"].replace("", "none")
        grouped = (
            orgs.groupby(["system_alias_action", "system_alias_blocked_generic", "system_alias_reason"], dropna=False)
            .agg(
                system_count=("system_id", lambda values: values.astype("string").fillna("").str.strip().replace("", pd.NA).nunique()),
                org_row_count=("system_id", "size"),
                top_system_names=("system_name", lambda values: " | ".join(values.astype("string").fillna("").str.strip().replace("", pd.NA).dropna().value_counts().head(5).index)),
            )
            .reset_index()
            .sort_values(["org_row_count", "system_count"], ascending=[False, False], kind="mergesort")
        )
        grouped[columns].to_csv(path, index=False)
        LOGGER.info("Wrote PPEF/PECOS system alias summary: %s", path)

    @staticmethod
    def _comparison_baseline_columns() -> list[str]:
        return [
            "baseline_view",
            "provider_id",
            "npi",
            "enrollment_id",
            "provider_full_name",
            "first_name",
            "middle_name",
            "last_name",
            "taxonomy_code_primary",
            "taxonomy_desc_primary",
            "taxonomy_codes_all",
            "taxonomy_descs_all",
            "specialty_normalized",
            "specialty_group",
            "provider_type_group",
            "specialty_rollup_source",
            "provider_practice_address_1",
            "provider_practice_address_2",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
            "provider_practice_phone",
            "mapped_clinic_id",
            "mapped_clinic_name",
            "mapped_system_id",
            "mapped_system_name",
            "mapped_practice_address_1",
            "mapped_practice_address_2",
            "mapped_practice_city",
            "mapped_practice_state",
            "mapped_practice_zip5",
            "mapped_practice_phone",
            "comparison_phone_set",
            "comparison_address_set",
            "comparison_zip_state_set",
            "comparison_address_line1_set",
            "comparison_city_state_set",
            "comparison_street_number_zip_set",
            "comparison_readiness_tier",
            "cms_current_active_flag",
            "cms_currentness_tier",
            "cms_currentness_reason",
            "cms_active_as_of_date",
            "baseline_current_active_flag",
            "baseline_currentness_tier",
            "baseline_currentness_reason",
            "baseline_active_as_of_date",
            "mapping_confidence",
            "mapping_confidence_tier",
            "mapping_method",
            "primary_selection_reason",
            "candidate_selection_reason",
            "system_alias_action",
            "system_alias_reason",
            "system_alias_blocked_generic",
            "system_rollup_quality_warning",
            "clinic_rollup_quality_warning",
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "blocked_generic_penalty",
            "candidate_clinic_count",
            "candidate_system_count",
            "candidate_rank",
            "selected_as_primary",
            "primary_is_tied",
            "primary_is_ambiguous",
            "relationship_source",
            "relationship_start_date",
            "relationship_end_date",
            "is_active_relationship",
            "ppef_snapshot_date",
            "pecos_snapshot_date",
            "npi_registry_snapshot_date",
            "provider_identity_source",
            "specialty_source",
            "phone_source",
            "address_source",
            "clinic_source",
            "system_source",
            "signal_state_zip_match",
            "signal_practice_support_state_zip_match",
            "signal_city_state_match",
            "signal_practice_support_city_state_match",
            "signal_address_line1_match",
            "signal_phone_match",
            "signal_rare_phone_match",
            "signal_org_name_strong",
            "signal_repeat_edge",
            "signal_blocked_generic_system",
        ]

    @staticmethod
    def _normalize_comparison_baseline_frame(frame: pd.DataFrame) -> pd.DataFrame:
        columns = IndividualProcessor._comparison_baseline_columns()
        numeric_columns = {
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "blocked_generic_penalty",
            "candidate_clinic_count",
            "candidate_system_count",
            "candidate_rank",
        }
        float_columns = {"mapping_score", "mapping_margin_score", "blocked_generic_penalty"}
        bool_columns = {
            "selected_as_primary",
            "primary_is_tied",
            "primary_is_ambiguous",
            "is_active_relationship",
            "signal_state_zip_match",
            "signal_practice_support_state_zip_match",
            "signal_city_state_match",
            "signal_practice_support_city_state_match",
            "signal_address_line1_match",
            "signal_phone_match",
            "signal_rare_phone_match",
            "signal_org_name_strong",
            "signal_repeat_edge",
            "signal_blocked_generic_system",
            "system_alias_blocked_generic",
            "cms_current_active_flag",
            "baseline_current_active_flag",
        }
        out = frame.copy()
        for column in columns:
            if column not in out.columns:
                out[column] = False if column in bool_columns else 0 if column in numeric_columns else ""
        for column in numeric_columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
            if column in float_columns:
                out[column] = out[column].astype("float64")
        for column in bool_columns:
            out[column] = out[column].fillna(False).astype(bool)
        for column in columns:
            if column not in numeric_columns and column not in bool_columns:
                out[column] = out[column].astype("string").fillna("").astype(str)
        return out[columns]

    def _ensure_currentness_fields(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if out.empty:
            for column in [
                "cms_current_active_flag",
                "cms_currentness_tier",
                "cms_currentness_reason",
                "cms_active_as_of_date",
                "baseline_current_active_flag",
                "baseline_currentness_tier",
                "baseline_currentness_reason",
                "baseline_active_as_of_date",
            ]:
                out[column] = False if column.endswith("_flag") else ""
            return out

        snapshot = (
            out["ppef_snapshot_date"].astype("string").fillna("").str.strip()
            if "ppef_snapshot_date" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        out["cms_active_as_of_date"] = snapshot.where(snapshot.ne(""), datetime.today().strftime("%Y%m%d"))

        active_relationship = (
            self._bool_series(out, "is_active_relationship")
            if "is_active_relationship" in out.columns
            else pd.Series(False, index=out.index)
        )
        relationship_source = (
            out["relationship_source"].astype("string").fillna("").str.strip()
            if "relationship_source" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        relationship_start = (
            out["relationship_start_date"].astype("string").fillna("").str.strip()
            if "relationship_start_date" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        relationship_end = (
            out["relationship_end_date"].astype("string").fillna("").str.strip()
            if "relationship_end_date" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        has_relationship_evidence = (
            active_relationship
            | relationship_source.ne("")
            | relationship_start.ne("")
            | relationship_end.ne("")
        )

        mapped_system = self._has_value(out, ["mapped_system_id", "mapped_system_name"])
        mapped_clinic = self._has_value(out, ["mapped_clinic_id", "mapped_clinic_name"])
        has_identity = self._has_value(out, ["provider_id", "npi", "enrollment_id"])
        mapping_tier = (
            out["mapping_confidence_tier"].astype("string").fillna("").str.strip().str.lower()
            if "mapping_confidence_tier" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        readiness = (
            out["comparison_readiness_tier"].astype("string").fillna("").str.strip().str.lower()
            if "comparison_readiness_tier" in out.columns
            else pd.Series("", index=out.index, dtype="string")
        )
        ambiguous = (
            self._bool_series(out, "primary_is_ambiguous")
            if "primary_is_ambiguous" in out.columns
            else pd.Series(False, index=out.index)
        )
        blocked_generic = (
            self._bool_series(out, "system_alias_blocked_generic")
            if "system_alias_blocked_generic" in out.columns
            else pd.Series(False, index=out.index)
        )
        if "signal_blocked_generic_system" in out.columns:
            blocked_generic = blocked_generic | self._bool_series(out, "signal_blocked_generic_system")

        active_mapped = active_relationship & mapped_system
        active_primary = active_mapped & mapped_clinic
        current_active_primary = (
            active_primary
            & has_identity
            & mapping_tier.isin(["high", "medium"])
            & readiness.isin(["high", "medium"])
            & ~ambiguous
            & ~blocked_generic
        )

        tier = pd.Series("unknown_currentness", index=out.index, dtype="string")
        reason = pd.Series("missing_relationship_evidence", index=out.index, dtype="string")
        inactive = has_relationship_evidence & ~active_relationship
        tier.loc[inactive] = "historical_or_inactive"
        reason.loc[inactive] = "relationship_inactive_or_ended"

        active_unmapped = active_relationship & ~mapped_system
        tier.loc[active_unmapped] = "active_unmapped"
        reason.loc[active_unmapped] = "active_relationship_without_mapped_system"

        tier.loc[active_mapped] = "active_mapped_system"
        reason.loc[active_mapped] = "active_relationship_with_mapped_system"

        tier.loc[active_primary] = "active_mapped_primary"
        reason.loc[active_primary] = "active_relationship_with_mapped_system_and_clinic"

        low_confidence_primary = active_primary & ~current_active_primary
        reason.loc[low_confidence_primary] = "active_primary_but_low_confidence_ambiguous_or_low_readiness"

        tier.loc[current_active_primary] = "current_active_primary"
        reason.loc[current_active_primary] = "active_relationship_high_confidence_primary_mapping"

        out["cms_current_active_flag"] = current_active_primary.astype(bool)
        out["cms_currentness_tier"] = tier
        out["cms_currentness_reason"] = reason
        out["baseline_current_active_flag"] = out["cms_current_active_flag"]
        out["baseline_currentness_tier"] = out["cms_currentness_tier"]
        out["baseline_currentness_reason"] = out["cms_currentness_reason"]
        out["baseline_active_as_of_date"] = out["cms_active_as_of_date"]
        return out

    def _primary_comparison_view(self, processed_df: pd.DataFrame, candidate_df: pd.DataFrame, view_name: str) -> pd.DataFrame:
        primary_columns = [
            "provider_id",
            "npi",
            "enrollment_id",
            "provider_full_name",
            "first_name",
            "middle_name",
            "last_name",
            "taxonomy_code_primary",
            "taxonomy_desc_primary",
            "taxonomy_codes_all",
            "taxonomy_descs_all",
            "specialty_normalized",
            "specialty_group",
            "provider_type_group",
            "specialty_rollup_source",
            "provider_practice_address_1",
            "provider_practice_address_2",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
            "provider_practice_phone",
            "mapped_clinic_id",
            "mapped_clinic_name",
            "mapped_system_id",
            "mapped_system_name",
            "mapping_confidence",
            "mapping_confidence_tier",
            "mapping_method",
            "primary_selection_reason",
            "system_alias_action",
            "system_alias_reason",
            "system_alias_blocked_generic",
            "system_rollup_quality_warning",
            "clinic_rollup_quality_warning",
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "blocked_generic_penalty",
            "candidate_clinic_count",
            "candidate_system_count",
            "primary_is_tied",
            "primary_is_ambiguous",
            "ppef_snapshot_date",
            "pecos_snapshot_date",
            "npi_registry_snapshot_date",
            "provider_identity_source",
            "specialty_source",
            "phone_source",
            "address_source",
            "clinic_source",
            "system_source",
            "comparison_phone_set",
            "comparison_address_set",
            "comparison_zip_state_set",
            "comparison_address_line1_set",
            "comparison_city_state_set",
            "comparison_street_number_zip_set",
            "comparison_readiness_tier",
        ]
        base = processed_df[[column for column in primary_columns if column in processed_df.columns]].copy()
        mapped = self._has_value(base, ["mapped_clinic_id", "mapped_system_id"])
        base = base.loc[mapped].copy()
        if view_name == "clean_primary":
            tier = base["mapping_confidence_tier"].astype("string").fillna("").str.strip()
            ambiguous = self._bool_series(base, "primary_is_ambiguous")
            base = base.loc[tier.isin(["high", "medium"]) & ~ambiguous].copy()

        if not base.empty and "provider_id" in base.columns:
            base["_baseline_row_completeness"] = self._row_completeness(
                base,
                [
                    "npi",
                    "enrollment_id",
                    "provider_full_name",
                    "taxonomy_desc_primary",
                    "provider_practice_address_1",
                    "provider_practice_city",
                    "provider_practice_state",
                    "provider_practice_zip5",
                    "provider_practice_phone",
                    "mapped_clinic_name",
                    "mapped_system_name",
                ],
            )
            mapping_score = base["mapping_score"] if "mapping_score" in base.columns else pd.Series(0, index=base.index)
            signal_count = (
                base["mapping_signal_count"] if "mapping_signal_count" in base.columns else pd.Series(0, index=base.index)
            )
            base["_baseline_mapping_score"] = pd.to_numeric(mapping_score, errors="coerce").fillna(0.0)
            base["_baseline_signal_count"] = pd.to_numeric(signal_count, errors="coerce").fillna(0)
            base = (
                base.sort_values(
                    by=[
                        "provider_id",
                        "_baseline_row_completeness",
                        "_baseline_mapping_score",
                        "_baseline_signal_count",
                        "enrollment_id",
                    ],
                    ascending=[True, False, False, False, True],
                    kind="mergesort",
                )
                .drop_duplicates(subset=["provider_id"], keep="first")
                .drop(
                    columns=[
                        "_baseline_row_completeness",
                        "_baseline_mapping_score",
                        "_baseline_signal_count",
                    ],
                    errors="ignore",
                )
                .copy()
            )

        if not candidate_df.empty and "selected_as_primary" in candidate_df.columns:
            selected = candidate_df[self._bool_series(candidate_df, "selected_as_primary")].copy()
            location_cols = [
                "provider_id",
                "practice_address_1",
                "practice_address_2",
                "practice_city",
                "practice_state",
                "practice_zip5",
                "practice_phone",
                "relationship_source",
                "relationship_start_date",
                "relationship_end_date",
                "is_active_relationship",
                "candidate_rank",
                "candidate_selection_reason",
                "signal_state_zip_match",
                "signal_practice_support_state_zip_match",
                "signal_city_state_match",
                "signal_practice_support_city_state_match",
                "signal_address_line1_match",
                "signal_phone_match",
                "signal_rare_phone_match",
                "signal_org_name_strong",
                "signal_repeat_edge",
                "signal_blocked_generic_system",
                "blocked_generic_penalty",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
                "system_rollup_quality_warning",
                "clinic_rollup_quality_warning",
            ]
            selected = selected[[column for column in location_cols if column in selected.columns]].drop_duplicates("provider_id")
            selected = selected.rename(
                columns={
                    "practice_address_1": "mapped_practice_address_1",
                    "practice_address_2": "mapped_practice_address_2",
                    "practice_city": "mapped_practice_city",
                    "practice_state": "mapped_practice_state",
                    "practice_zip5": "mapped_practice_zip5",
                    "practice_phone": "mapped_practice_phone",
                }
            )
            base = base.merge(selected, on="provider_id", how="left")

        base = self._ensure_comparison_fields(base, refresh_specialty=False, include_mapped_contact=True)
        base["baseline_view"] = view_name
        base["selected_as_primary"] = True
        base["candidate_rank"] = 1
        base["candidate_selection_reason"] = base.get("candidate_selection_reason", base.get("mapping_method", ""))
        base["relationship_source"] = base.get("relationship_source", "reassignment")
        if "is_active_relationship" in base.columns:
            base["is_active_relationship"] = self._bool_series(base, "is_active_relationship")
        else:
            base["is_active_relationship"] = False
        base = self._ensure_currentness_fields(base)
        if view_name == "current_active_primary":
            base = base.loc[self._bool_series(base, "baseline_current_active_flag")].copy()
        return self._normalize_comparison_baseline_frame(base)

    def _candidate_graph_comparison_view(self, processed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> pd.DataFrame:
        if candidate_df.empty:
            return self._normalize_comparison_baseline_frame(pd.DataFrame())
        provider_columns = [
            "provider_id",
            "npi",
            "enrollment_id",
            "provider_full_name",
            "first_name",
            "middle_name",
            "last_name",
            "taxonomy_code_primary",
            "taxonomy_desc_primary",
            "taxonomy_codes_all",
            "taxonomy_descs_all",
            "specialty_normalized",
            "specialty_group",
            "provider_type_group",
            "specialty_rollup_source",
            "provider_practice_address_1",
            "provider_practice_address_2",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
            "provider_practice_phone",
            "ppef_snapshot_date",
            "pecos_snapshot_date",
            "npi_registry_snapshot_date",
            "provider_identity_source",
            "specialty_source",
            "phone_source",
            "address_source",
            "comparison_phone_set",
            "comparison_address_set",
            "comparison_zip_state_set",
            "comparison_address_line1_set",
            "comparison_city_state_set",
            "comparison_street_number_zip_set",
            "comparison_readiness_tier",
        ]
        provider_info = processed_df[[column for column in provider_columns if column in processed_df.columns]].drop_duplicates("provider_id").copy()
        graph = candidate_df.merge(provider_info, on="provider_id", how="left")
        graph = graph.rename(
            columns={
                "clinic_id": "mapped_clinic_id",
                "clinic_name": "mapped_clinic_name",
                "system_id": "mapped_system_id",
                "system_name": "mapped_system_name",
                "practice_address_1": "mapped_practice_address_1",
                "practice_address_2": "mapped_practice_address_2",
                "practice_city": "mapped_practice_city",
                "practice_state": "mapped_practice_state",
                "practice_zip5": "mapped_practice_zip5",
                "practice_phone": "mapped_practice_phone",
            }
        )
        graph["baseline_view"] = "candidate_graph"
        graph["clinic_source"] = "ppef_reassignment_pecos_candidate"
        graph["system_source"] = "ppef_reassignment_pecos_candidate"
        graph = self._ensure_comparison_fields(graph, refresh_specialty=False, include_mapped_contact=False)
        graph = self._ensure_currentness_fields(graph)
        return self._normalize_comparison_baseline_frame(graph)

    def _write_comparison_baseline_exports(self, date_str: str, processed_df: pd.DataFrame, candidate_df: pd.DataFrame) -> None:
        path = self.comparison_baseline_output_path(date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        view_builders = [
            ("all_mapped", lambda: self._primary_comparison_view(processed_df, candidate_df, "all_mapped")),
            ("clean_primary", lambda: self._primary_comparison_view(processed_df, candidate_df, "clean_primary")),
            (
                "current_active_primary",
                lambda: self._primary_comparison_view(processed_df, candidate_df, "current_active_primary"),
            ),
            ("candidate_graph", lambda: self._candidate_graph_comparison_view(processed_df, candidate_df)),
        ]
        writer: Optional[pq.ParquetWriter] = None
        try:
            for view_name, build_view in view_builders:
                LOGGER.info("Building PPEF comparison baseline view '%s'", view_name)
                view = build_view()
                LOGGER.info("Writing PPEF comparison baseline view '%s': %s rows", view_name, f"{len(view):,}")
                table = pa.Table.from_pandas(view, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
                del table
                del view
        finally:
            if writer is not None:
                writer.close()
        LOGGER.info("Wrote PPEF comparison baseline export: %s", path)

    def quality_exports(self, df: pd.DataFrame, metadata: dict, date_str: str) -> tuple[pd.DataFrame, dict]:
        start = time.time()
        out = self._ensure_freshness_and_provenance(df, date_str)
        out = self._ensure_comparison_fields(out)
        candidates_path = self.candidates_output_path(date_str)
        if candidates_path.exists():
            candidate_df = pd.read_parquet(candidates_path)
        else:
            LOGGER.warning("No PPEF affiliation candidates parquet found for quality exports: %s", candidates_path)
            candidate_df = self._empty_links_frame()
        if "selected_as_primary" in candidate_df.columns:
            best_df = candidate_df[candidate_df["selected_as_primary"].fillna(False).astype(bool)].copy()
        else:
            best_df = pd.DataFrame()
        self._write_mapping_quality_report(date_str, out, candidate_df, best_df)
        self._write_system_quality_report(date_str, best_df)
        self._write_field_provenance_summary(date_str, out)
        self._write_specialty_rollup_audit(date_str, out)
        self._write_unresolved_taxonomy_audit(date_str, out)
        self._write_comparison_readiness_summary(date_str, out)
        self._write_system_alias_summary(date_str)
        self._write_comparison_baseline_exports(date_str, out, candidate_df)
        metadata = add_mode_completion(
            metadata,
            "quality_exports",
            time.time() - start,
            len(out),
            [
                "ppef_individuals_comparison_baseline",
                "ppef_pecos_mapping_quality_systems",
                "ppef_pecos_field_provenance_summary",
                "ppef_specialty_rollup_audit",
                "ppef_unresolved_taxonomy_audit",
                "ppef_comparison_readiness_summary",
                "ppef_pecos_system_alias_summary",
            ],
        )
        return out, metadata

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

    def _build_reassignment_links_fast(
        self,
        reassignment_path: Path,
        schema: dict[str, Optional[str]],
        individuals_df: pd.DataFrame,
        org_df: pd.DataFrame,
    ) -> pd.DataFrame:
        provider_key = schema.get("provider_enrollment_id") or schema.get("provider_npi")
        org_key = schema.get("org_enrollment_id") or schema.get("org_npi")
        if not provider_key or not org_key:
            return self._empty_links_frame()

        rel_columns = [provider_key, org_key]
        if schema.get("start_date"):
            rel_columns.append(schema["start_date"])
        if schema.get("end_date"):
            rel_columns.append(schema["end_date"])
        rel_columns = list(dict.fromkeys(rel_columns))
        rel_df = pd.read_parquet(reassignment_path, columns=rel_columns).copy()

        rel_df = rel_df.rename(columns={provider_key: "_provider_join_key", org_key: "_org_join_key"})
        rel_df["_provider_join_key"] = rel_df["_provider_join_key"].astype("string").fillna("").str.strip()
        rel_df["_org_join_key"] = rel_df["_org_join_key"].astype("string").fillna("").str.strip()
        rel_df = rel_df[(rel_df["_provider_join_key"] != "") & (rel_df["_org_join_key"] != "")]
        if rel_df.empty:
            return self._empty_links_frame()

        rel_df = rel_df.drop_duplicates()

        if schema.get("provider_enrollment_id"):
            provider_frame = individuals_df[
                [
                    "provider_id",
                    "enrollment_id",
                ]
            ].copy()
            provider_frame = provider_frame.rename(columns={"enrollment_id": "_provider_join_key"})
        else:
            provider_frame = individuals_df[
                [
                    "provider_id",
                    "npi",
                ]
            ].copy()
            provider_frame = provider_frame.rename(columns={"npi": "_provider_join_key"})

        provider_frame["_provider_join_key"] = provider_frame["_provider_join_key"].astype("string").fillna("").str.strip()
        provider_frame = provider_frame[provider_frame["_provider_join_key"] != ""].drop_duplicates(subset=["_provider_join_key", "provider_id"])

        if schema.get("org_enrollment_id"):
            org_frame = org_df[
                [
                    "clinic_id",
                    "clinic_name",
                    "system_id",
                    "system_name",
                    "source_enrollment_id",
                ]
            ].copy()
            org_frame = org_frame.rename(columns={"source_enrollment_id": "_org_join_key"})
            method = "org_enrollment_id"
            method_bonus = 3.5
        else:
            org_frame = org_df[
                [
                    "clinic_id",
                    "clinic_name",
                    "system_id",
                    "system_name",
                    "org_npi",
                ]
            ].copy()
            org_frame = org_frame.rename(columns={"org_npi": "_org_join_key"})
            method = "org_npi"
            method_bonus = 4.0

        org_frame["_org_join_key"] = org_frame["_org_join_key"].astype("string").fillna("").str.strip()
        org_frame = org_frame[org_frame["_org_join_key"] != ""].drop_duplicates()

        merged = rel_df.merge(provider_frame, on="_provider_join_key", how="inner")
        if merged.empty:
            return self._empty_links_frame()
        merged = merged.merge(org_frame, on="_org_join_key", how="inner")
        if merged.empty:
            return self._empty_links_frame()

        today = pd.Timestamp.today().normalize()
        start_col = schema.get("start_date")
        end_col = schema.get("end_date")
        start_dates = pd.to_datetime(merged[start_col], errors="coerce") if start_col and start_col in merged.columns else pd.Series(pd.NaT, index=merged.index)
        end_dates = pd.to_datetime(merged[end_col], errors="coerce") if end_col and end_col in merged.columns else pd.Series(pd.NaT, index=merged.index)
        active = end_dates.isna() | (end_dates.dt.normalize() >= today)
        ref_dates = start_dates.where(active, end_dates)
        recency_days = pd.Series(99999, index=merged.index, dtype="int64")
        has_ref = ref_dates.notna()
        if has_ref.any():
            recency_days.loc[has_ref] = (today - ref_dates.loc[has_ref].dt.normalize()).dt.days.clip(lower=0).astype("int64")
        continuity_days = pd.Series(0, index=merged.index, dtype="int64")
        has_start = start_dates.notna()
        if has_start.any():
            end_for_duration = end_dates.where(end_dates.notna(), today)
            continuity_days.loc[has_start] = (
                end_for_duration.loc[has_start].dt.normalize() - start_dates.loc[has_start].dt.normalize()
            ).dt.days.clip(lower=0).astype("int64")

        recency_score = pd.Series(2.5, index=merged.index, dtype="float64")
        inactive = ~active
        if inactive.any():
            recency_score.loc[inactive] = (2.0 - (recency_days.loc[inactive].clip(upper=3650) / 365.0)).clip(lower=0.0)
        continuity_score = (continuity_days / 365.0).clip(upper=2.5).astype("float64")
        total_affiliation_score = recency_score + continuity_score + method_bonus + 1.5 + active.astype("int64") * 3.0

        links_df = pd.DataFrame(
            {
                "provider_id": merged["provider_id"],
                "clinic_id": merged["clinic_id"].astype("string").fillna(""),
                "clinic_name": merged["clinic_name"].astype("string").fillna(""),
                "system_id": merged["system_id"].astype("string").fillna(""),
                "system_name": merged["system_name"].astype("string").fillna(""),
                "relationship_source": "reassignment",
                "relationship_start_date": start_dates,
                "relationship_end_date": end_dates,
                "is_active_relationship": active.astype(bool),
                "recency_score": recency_score,
                "continuity_score": continuity_score,
                "address_match_score": 0.0,
                "taxonomy_match_score": 0.0,
                "total_affiliation_score": total_affiliation_score,
                "selected_as_primary": False,
                "mapping_method": f"reassignment:{method}",
                "shared_address_provider_count": 0,
                "billing_affiliation_recency_rank": recency_days,
                "relationship_duration_days": continuity_days,
                "raw_address_key": "",
            }
        )
        return links_df.drop_duplicates(
            subset=[
                "provider_id",
                "clinic_id",
                "system_id",
                "relationship_source",
                "mapping_method",
                "relationship_start_date",
                "relationship_end_date",
            ]
        )

    def _build_candidates(self, relation_df: pd.DataFrame, schema: dict[str, Optional[str]], relationship_source: str, individuals_df: pd.DataFrame, org_df: pd.DataFrame, provider_npi_map: dict[str, int], provider_enrollment_map: dict[str, int], org_lookups: dict[str, dict[str, list[int]]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if relation_df is None or relation_df.empty:
            return rows
        if not schema.get("provider_npi") and not schema.get("provider_enrollment_id"):
            LOGGER.warning(
                "Skipping %s candidate build because provider identifiers could not be resolved from columns=%s",
                relationship_source,
                list(relation_df.columns),
            )
            return rows
        if (
            not schema.get("org_npi")
            and not schema.get("org_enrollment_id")
            and not schema.get("address1")
            and not schema.get("phone")
            and not schema.get("org_name")
        ):
            LOGGER.warning(
                "Skipping %s candidate build because org identifiers could not be resolved from columns=%s",
                relationship_source,
                list(relation_df.columns),
            )
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

        for column in [
            "mapped_clinic_id",
            "mapped_clinic_name",
            "mapped_system_id",
            "mapped_system_name",
            "mapping_confidence",
            "mapping_method",
            "primary_affiliation_start_date",
            "primary_affiliation_end_date",
            "billing_affiliation_recency_rank",
            "shared_address_provider_count",
            "active_reassignment_count",
            "most_recent_reassignment_date",
            "longest_reassignment_days",
            "mapping_score",
            "mapping_margin_score",
            "mapping_signal_count",
            "candidate_clinic_count",
            "candidate_system_count",
            "primary_is_tied",
            "primary_is_ambiguous",
            "mapping_confidence_tier",
            "primary_selection_reason",
        ]:
            if column not in out.columns:
                out[column] = ""

        org_df = pd.read_parquet(
            orgs_path,
            columns=[
                "clinic_id",
                "clinic_name",
                "system_id",
                "system_name",
                "org_npi",
                "org_entity_id",
                "source_enrollment_id",
                "practice_address_1",
                "practice_address_2",
                "practice_city",
                "practice_state",
                "practice_zip5",
                "practice_phone",
                "taxonomy_desc_primary",
                "is_hospital",
            ],
        ).copy()
        org_bridge = self._build_org_enrollment_bridge(org_df)

        reassignment_path = self.find_related_parquet(["REASSIGN"], ["ppef_individuals", "pecos_orgs"])
        practice_path = self.find_related_parquet(["PRACTICE|LOCATION"], ["ppef_individuals", "pecos_orgs"])
        practice_support = self._load_practice_support(practice_path)
        if practice_path and practice_path.exists():
            LOGGER.info("Loaded practice-location support: %s (%s provider rows)", practice_path, f"{len(practice_support):,}")

        if reassignment_path and reassignment_path.exists():
            rel_df = pd.read_parquet(reassignment_path)
            reassignment_schema = self._resolve_reassignment_schema(rel_df)
            LOGGER.info(
                "Loaded reassignment parquet: %s (%s rows) schema=%s",
                reassignment_path,
                f"{len(rel_df):,}",
                reassignment_schema,
            )
        else:
            LOGGER.warning("No reassignment parquet found.")

        if reassignment_path is None or not reassignment_path.exists():
            self.links_output_path(date_str).parent.mkdir(parents=True, exist_ok=True)
            LOGGER.warning("No affiliation candidates were generated from the available relation files.")
            self._empty_links_frame().to_parquet(self.links_output_path(date_str), index=False)
            metadata = add_mode_completion(metadata, "clinic_mapping", time.time() - start, len(out), ["mapped_clinic_id", "mapped_clinic_name", "mapped_system_id", "mapped_system_name"])
            return out, metadata

        provider_cols = [
            "provider_id",
            "enrollment_id",
            "npi",
            "provider_practice_address_1",
            "provider_practice_address_2",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
            "provider_practice_phone",
            "taxonomy_desc_primary",
            "ORG_NAME",
            "STATE_CD",
        ]
        provider_frame = out[[column for column in provider_cols if column in out.columns]].copy()
        for column in provider_cols:
            if column not in provider_frame.columns:
                provider_frame[column] = ""
            provider_frame[column] = provider_frame[column].astype("string").fillna("").str.strip()
        provider_frame["provider_practice_zip5"] = provider_frame["provider_practice_zip5"].map(zip5)
        provider_frame["provider_practice_phone"] = provider_frame["provider_practice_phone"].map(lambda v: safe_phone(v) or "")
        if not practice_support.empty:
            provider_frame = provider_frame.merge(practice_support, on="enrollment_id", how="left")
        else:
            for column in ["practice_support_city", "practice_support_state", "practice_support_zip5"]:
                provider_frame[column] = ""
        provider_frame["effective_practice_city"] = coalesce_columns(
            provider_frame,
            ["provider_practice_city", "practice_support_city"],
        )
        provider_frame["effective_practice_state"] = coalesce_columns(
            provider_frame,
            ["provider_practice_state", "practice_support_state", "STATE_CD"],
        )
        provider_frame["effective_practice_zip5"] = coalesce_columns(
            provider_frame,
            ["provider_practice_zip5", "practice_support_zip5"],
        ).map(zip5)
        provider_frame["provider_org_name"] = coalesce_columns(provider_frame, ["ORG_NAME"])
        provider_frame["provider_org_name_canonical"] = provider_frame["provider_org_name"].map(canonical_entity_name)
        provider_frame["provider_practice_address_1_norm"] = self._normalize_address_line1_series(provider_frame["provider_practice_address_1"])
        provider_frame["effective_city_norm"] = self._normalize_series(provider_frame["effective_practice_city"])
        provider_frame["effective_state_norm"] = self._normalize_series(provider_frame["effective_practice_state"])
        provider_frame["effective_zip5"] = provider_frame["effective_practice_zip5"].map(zip5)
        provider_frame["practice_support_city_norm"] = self._normalize_series(provider_frame["practice_support_city"])
        provider_frame["practice_support_state_norm"] = self._normalize_series(provider_frame["practice_support_state"])
        provider_frame["practice_support_zip5_norm"] = provider_frame["practice_support_zip5"].map(zip5)
        provider_frame["provider_row_completeness"] = self._row_completeness(
            provider_frame,
            [
                "provider_practice_address_1",
                "effective_practice_city",
                "effective_practice_state",
                "effective_practice_zip5",
                "provider_practice_phone",
                "provider_org_name",
                "taxonomy_desc_primary",
            ],
        )
        provider_frame = provider_frame[provider_frame["provider_id"] != ""]
        provider_frame = (
            provider_frame.sort_values(
                by=["provider_row_completeness", "provider_practice_address_1", "provider_org_name"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["provider_id", "enrollment_id"], keep="first")
            .copy()
        )
        LOGGER.info("Prepared provider support frame for clinic mapping: %s unique provider/enrollment rows", f"{len(provider_frame):,}")

        provider_key = reassignment_schema.get("provider_enrollment_id") or reassignment_schema.get("provider_npi")
        org_key = reassignment_schema.get("org_enrollment_id") or reassignment_schema.get("org_npi")
        if not provider_key or not org_key:
            raise RuntimeError(f"Could not resolve reassignment join keys from {reassignment_path}: {reassignment_schema}")
        rel_df = pd.read_parquet(reassignment_path, columns=[provider_key, org_key]).copy()
        rel_df = rel_df.rename(columns={provider_key: "_provider_join_key", org_key: "_org_join_key"})
        rel_df["_provider_join_key"] = rel_df["_provider_join_key"].astype("string").fillna("").str.strip()
        rel_df["_org_join_key"] = rel_df["_org_join_key"].astype("string").fillna("").str.strip()
        rel_df = rel_df[(rel_df["_provider_join_key"] != "") & (rel_df["_org_join_key"] != "")].drop_duplicates()

        merged = rel_df.merge(
            provider_frame,
            left_on="_provider_join_key",
            right_on="enrollment_id" if reassignment_schema.get("provider_enrollment_id") else "npi",
            how="inner",
        )
        merged = merged.merge(
            org_bridge,
            left_on="_org_join_key",
            right_on="source_enrollment_id",
            how="inner",
        )
        LOGGER.info("Built raw provider-org reassignment join rows: %s", f"{len(merged):,}")
        if merged.empty:
            self.links_output_path(date_str).parent.mkdir(parents=True, exist_ok=True)
            LOGGER.warning("Reassignment join produced 0 provider-org candidates.")
            self._empty_links_frame().to_parquet(self.links_output_path(date_str), index=False)
            metadata = add_mode_completion(metadata, "clinic_mapping", time.time() - start, len(out), ["mapped_clinic_id", "mapped_clinic_name", "mapped_system_id", "mapped_system_name"])
            return out, metadata

        merged["org_practice_state_norm"] = self._normalize_series(merged["practice_state"])
        merged["org_practice_city_norm"] = self._normalize_series(merged["practice_city"])
        merged["org_practice_zip5"] = merged["practice_zip5"].map(zip5)
        merged["org_practice_address_1_norm"] = self._normalize_address_line1_series(merged["practice_address_1"])
        merged["org_practice_phone_norm"] = merged["practice_phone"].map(lambda v: safe_phone(v) or "")
        phone_clinic_counts = (
            org_bridge[org_bridge["practice_phone"].astype("string").fillna("").str.strip().ne("")]
            .groupby("practice_phone")["clinic_id"]
            .nunique()
        )
        merged["signal_state_match"] = (
            merged["effective_state_norm"].ne("")
            & merged["org_practice_state_norm"].ne("")
            & merged["effective_state_norm"].eq(merged["org_practice_state_norm"])
        )
        merged["signal_state_zip_match"] = (
            merged["signal_state_match"]
            & merged["effective_zip5"].ne("")
            & merged["org_practice_zip5"].ne("")
            & merged["effective_zip5"].eq(merged["org_practice_zip5"])
        )
        merged["signal_city_state_match"] = (
            merged["signal_state_match"]
            & merged["effective_city_norm"].ne("")
            & merged["org_practice_city_norm"].ne("")
            & merged["effective_city_norm"].eq(merged["org_practice_city_norm"])
        )
        merged["signal_practice_support_state_zip_match"] = (
            merged["practice_support_state_norm"].ne("")
            & merged["practice_support_zip5_norm"].ne("")
            & merged["org_practice_state_norm"].ne("")
            & merged["org_practice_zip5"].ne("")
            & merged["practice_support_state_norm"].eq(merged["org_practice_state_norm"])
            & merged["practice_support_zip5_norm"].eq(merged["org_practice_zip5"])
        )
        merged["signal_practice_support_city_state_match"] = (
            merged["practice_support_state_norm"].ne("")
            & merged["practice_support_city_norm"].ne("")
            & merged["org_practice_state_norm"].ne("")
            & merged["org_practice_city_norm"].ne("")
            & merged["practice_support_state_norm"].eq(merged["org_practice_state_norm"])
            & merged["practice_support_city_norm"].eq(merged["org_practice_city_norm"])
        )
        merged["signal_address_line1_match"] = (
            merged["provider_practice_address_1_norm"].ne("")
            & merged["org_practice_address_1_norm"].ne("")
            & merged["provider_practice_address_1_norm"].eq(merged["org_practice_address_1_norm"])
        )
        merged["signal_phone_match"] = (
            merged["provider_practice_phone"].ne("")
            & merged["org_practice_phone_norm"].ne("")
            & merged["provider_practice_phone"].eq(merged["org_practice_phone_norm"])
        )

        candidate_df = (
            merged.groupby(["provider_id", "clinic_id", "system_id"], as_index=False)
            .agg(
                clinic_name=("clinic_name", "first"),
                clinic_name_canonical=("clinic_name_canonical", "first"),
                system_name=("system_name", "first"),
                system_name_canonical=("system_name_canonical", "first"),
                org_npi=("org_npi", "first"),
                org_entity_id=("org_entity_id", "first"),
                practice_address_1=("practice_address_1", "first"),
                practice_address_2=("practice_address_2", "first"),
                practice_city=("practice_city", "first"),
                practice_state=("practice_state", "first"),
                practice_zip5=("practice_zip5", "first"),
                practice_phone=("practice_phone", "first"),
                org_taxonomy_desc_primary=("org_taxonomy_desc_primary", "first"),
                is_hospital=("is_hospital", "max"),
                provider_org_name_canonical=("provider_org_name_canonical", "first"),
                edge_row_count=("_org_join_key", "size"),
                org_enrollment_count=("_org_join_key", "nunique"),
                provider_enrollment_count=("enrollment_id", "nunique"),
                signal_state_match=("signal_state_match", "max"),
                signal_state_zip_match=("signal_state_zip_match", "max"),
                signal_city_state_match=("signal_city_state_match", "max"),
                signal_practice_support_state_zip_match=("signal_practice_support_state_zip_match", "max"),
                signal_practice_support_city_state_match=("signal_practice_support_city_state_match", "max"),
                signal_address_line1_match=("signal_address_line1_match", "max"),
                signal_phone_match=("signal_phone_match", "max"),
                system_alias_action=("system_alias_action", "first"),
                system_alias_reason=("system_alias_reason", "first"),
                system_alias_blocked_generic=("system_alias_blocked_generic", "max"),
                system_rollup_quality_warning=("system_rollup_quality_warning", "first"),
                clinic_rollup_quality_warning=("clinic_rollup_quality_warning", "first"),
            )
            .copy()
        )
        candidate_df["matched_phone_clinic_count"] = (
            candidate_df["practice_phone"].astype("string").fillna("").str.strip().map(phone_clinic_counts).fillna(0).astype("int64")
        )
        candidate_df["signal_rare_phone_match"] = candidate_df["signal_phone_match"] & candidate_df["matched_phone_clinic_count"].between(1, 5)
        LOGGER.info("Collapsed reassignment graph to provider-clinic candidates: %s", f"{len(candidate_df):,}")
        candidate_df["provider_org_name_prefix"] = candidate_df["provider_org_name_canonical"].astype("string").fillna("").str.slice(0, 12)
        candidate_df["clinic_name_prefix"] = candidate_df["clinic_name_canonical"].astype("string").fillna("").str.slice(0, 12)
        candidate_df["system_name_prefix"] = candidate_df["system_name_canonical"].astype("string").fillna("").str.slice(0, 12)
        candidate_df["signal_org_name_exact"] = (
            candidate_df["provider_org_name_canonical"].ne("")
            & (
                candidate_df["provider_org_name_canonical"].eq(candidate_df["clinic_name_canonical"])
                | candidate_df["provider_org_name_canonical"].eq(candidate_df["system_name_canonical"])
            )
        )
        candidate_df["signal_org_name_prefix"] = (
            candidate_df["provider_org_name_prefix"].ne("")
            & (
                candidate_df["provider_org_name_prefix"].eq(candidate_df["clinic_name_prefix"])
                | candidate_df["provider_org_name_prefix"].eq(candidate_df["system_name_prefix"])
            )
        )
        candidate_df["signal_org_name_strong"] = candidate_df["signal_org_name_exact"] | candidate_df["signal_org_name_prefix"]
        candidate_df["org_name_similarity"] = candidate_df["signal_org_name_exact"].astype("float64") + candidate_df["signal_org_name_prefix"].astype("float64") * 0.5
        candidate_df["signal_repeat_edge"] = candidate_df["edge_row_count"] > 1
        candidate_df["signal_blocked_generic_system"] = candidate_df["system_alias_blocked_generic"].fillna(False).astype(bool)
        candidate_df["mapping_signal_count"] = (
            candidate_df["signal_state_zip_match"].astype("int64")
            + candidate_df["signal_practice_support_state_zip_match"].astype("int64")
            + candidate_df["signal_address_line1_match"].astype("int64")
            + candidate_df["signal_phone_match"].astype("int64")
            + candidate_df["signal_city_state_match"].astype("int64")
            + candidate_df["signal_practice_support_city_state_match"].astype("int64")
            + candidate_df["signal_org_name_strong"].astype("int64")
            + candidate_df["signal_repeat_edge"].astype("int64")
        )
        candidate_df["relationship_source"] = "reassignment"
        candidate_df["relationship_start_date"] = pd.NaT
        candidate_df["relationship_end_date"] = pd.NaT
        candidate_df["is_active_relationship"] = True
        candidate_df["recency_score"] = 0.0
        candidate_df["continuity_score"] = 0.0
        candidate_df["address_match_score"] = candidate_df["signal_address_line1_match"].astype("float64")
        candidate_df["taxonomy_match_score"] = 0.0
        candidate_df["shared_address_provider_count"] = 0
        candidate_df["relationship_duration_days"] = 0
        candidate_df["raw_address_key"] = ""
        direct_evidence = (
            candidate_df["signal_state_zip_match"].fillna(False).astype(bool)
            | candidate_df["signal_address_line1_match"].fillna(False).astype(bool)
            | candidate_df["signal_rare_phone_match"].fillna(False).astype(bool)
            | (
                candidate_df["signal_org_name_exact"].fillna(False).astype(bool)
                & candidate_df["signal_city_state_match"].fillna(False).astype(bool)
            )
        )
        candidate_df["blocked_generic_penalty"] = 0.0
        weak_blocked = candidate_df["signal_blocked_generic_system"] & ~direct_evidence
        supported_blocked = candidate_df["signal_blocked_generic_system"] & direct_evidence
        candidate_df.loc[weak_blocked, "blocked_generic_penalty"] = 4.0
        candidate_df.loc[supported_blocked, "blocked_generic_penalty"] = 1.0
        candidate_df["total_affiliation_score"] = (
            5.0
            + candidate_df["signal_state_zip_match"].astype("float64") * 4.0
            + candidate_df["signal_practice_support_state_zip_match"].astype("float64") * 0.75
            + (candidate_df["signal_state_match"] & ~candidate_df["signal_state_zip_match"]).astype("float64") * 1.0
            + candidate_df["signal_city_state_match"].astype("float64") * 1.5
            + candidate_df["signal_practice_support_city_state_match"].astype("float64") * 0.5
            + candidate_df["signal_address_line1_match"].astype("float64") * 3.0
            + candidate_df["signal_phone_match"].astype("float64") * 0.75
            + candidate_df["signal_rare_phone_match"].astype("float64") * 1.75
            + candidate_df["signal_org_name_exact"].astype("float64") * 2.0
            + (candidate_df["signal_org_name_strong"] & ~candidate_df["signal_org_name_exact"]).astype("float64") * 1.0
            + (candidate_df["edge_row_count"] - 1).clip(lower=0, upper=4).astype("float64") * 0.5
            + (candidate_df["provider_enrollment_count"] - 1).clip(lower=0, upper=2).astype("float64") * 0.25
            - candidate_df["blocked_generic_penalty"]
        )
        candidate_df = candidate_df.sort_values(
            by=[
                "provider_id",
                "total_affiliation_score",
                "mapping_signal_count",
                "system_alias_blocked_generic",
                "signal_address_line1_match",
                "signal_phone_match",
                "edge_row_count",
                "system_id",
                "clinic_id",
            ],
            ascending=[True, False, False, True, False, False, False, True, True],
            kind="mergesort",
        ).copy()
        candidate_df["candidate_rank"] = candidate_df.groupby("provider_id").cumcount() + 1
        candidate_df["selected_as_primary"] = candidate_df["candidate_rank"].eq(1)
        candidate_df["top_score"] = candidate_df.groupby("provider_id")["total_affiliation_score"].transform("max")
        candidate_df["is_top_score"] = candidate_df["total_affiliation_score"].eq(candidate_df["top_score"])
        candidate_df["top_score_count"] = candidate_df.groupby("provider_id")["is_top_score"].transform("sum")
        candidate_df["next_score"] = candidate_df.groupby("provider_id")["total_affiliation_score"].shift(-1)
        provider_summary = (
            candidate_df.groupby("provider_id", as_index=False)
            .agg(
                candidate_clinic_count=("clinic_id", "nunique"),
                candidate_system_count=("system_id", "nunique"),
                active_reassignment_count=("edge_row_count", "sum"),
            )
        )
        candidate_df = candidate_df.merge(provider_summary, on="provider_id", how="left")
        best = candidate_df.loc[candidate_df["selected_as_primary"]].copy()
        best["mapping_score"] = best["total_affiliation_score"].astype("float64")
        best["mapping_margin_score"] = (
            best["total_affiliation_score"] - best["next_score"].fillna(0.0)
        ).clip(lower=0.0)
        best["primary_is_tied"] = best["top_score_count"].fillna(0).astype("int64") > 1
        best["primary_is_ambiguous"] = (
            (best["candidate_system_count"].fillna(0).astype("int64") > 1)
            & (
                best["primary_is_tied"]
                | (best["mapping_signal_count"].fillna(0).astype("int64") < 2)
                | (best["mapping_margin_score"] < 1.0)
            )
        )
        best["mapping_confidence_tier"] = [
            self._confidence_tier(int(signal_count), float(margin), int(system_count), bool(tied))
            for signal_count, margin, system_count, tied in zip(
                best["mapping_signal_count"].fillna(0),
                best["mapping_margin_score"].fillna(0.0),
                best["candidate_system_count"].fillna(0),
                best["primary_is_tied"].fillna(False),
            )
        ]
        confidence_values = []
        for tier, signal_count, margin, edge_count in zip(
            best["mapping_confidence_tier"],
            best["mapping_signal_count"].fillna(0),
            best["mapping_margin_score"].fillna(0.0),
            best["edge_row_count"].fillna(1),
        ):
            confidence = 0.35 + (0.08 * float(signal_count)) + (0.04 * min(float(margin), 4.0)) + (0.04 * min(max(float(edge_count) - 1.0, 0.0), 3.0))
            if tier == "high":
                confidence = max(confidence, 0.82)
            elif tier == "medium":
                confidence = min(max(confidence, 0.60), 0.81)
            else:
                confidence = min(confidence, 0.59)
            confidence_values.append(f"{min(confidence, 0.99):.3f}")
        best["mapping_confidence"] = confidence_values
        best["primary_selection_reason"] = "reassignment"
        for column, label in [
            ("signal_state_zip_match", "state_zip"),
            ("signal_practice_support_state_zip_match", "practice_support_zip"),
            ("signal_address_line1_match", "address"),
            ("signal_rare_phone_match", "rare_phone"),
            ("signal_phone_match", "phone"),
            ("signal_org_name_strong", "org_name"),
            ("signal_city_state_match", "city_state"),
            ("signal_practice_support_city_state_match", "practice_support_city"),
            ("signal_repeat_edge", "repeat_edge"),
            ("signal_blocked_generic_system", "blocked_generic_system"),
        ]:
            mask = best[column].fillna(False).astype(bool)
            best.loc[mask, "primary_selection_reason"] = best.loc[mask, "primary_selection_reason"] + f"+{label}"
        best["mapping_method"] = best["primary_selection_reason"]
        best["primary_affiliation_start_date"] = ""
        best["primary_affiliation_end_date"] = ""
        best["billing_affiliation_recency_rank"] = 1
        best["shared_address_provider_count"] = 0
        best["most_recent_reassignment_date"] = ""
        best["longest_reassignment_days"] = 0

        candidate_df["mapping_score"] = candidate_df["total_affiliation_score"].astype("float64")
        candidate_df["mapping_margin_score"] = (candidate_df["total_affiliation_score"] - candidate_df["next_score"].fillna(0.0)).clip(lower=0.0)
        candidate_df["primary_is_tied"] = candidate_df["top_score_count"].fillna(0).astype("int64") > 1
        best_by_provider = best.set_index("provider_id")
        provider_key_series = candidate_df["provider_id"].astype("string")
        candidate_df["primary_is_ambiguous"] = provider_key_series.map(best_by_provider["primary_is_ambiguous"]).fillna(False).astype(bool)
        candidate_df["mapping_confidence_tier"] = provider_key_series.map(best_by_provider["mapping_confidence_tier"]).fillna("")
        candidate_df["mapping_confidence"] = provider_key_series.map(best_by_provider["mapping_confidence"]).fillna("")
        candidate_df["primary_selection_reason"] = provider_key_series.map(best_by_provider["primary_selection_reason"]).fillna("")
        candidate_df["candidate_selection_reason"] = "reassignment"
        for column, label in [
            ("signal_state_zip_match", "state_zip"),
            ("signal_practice_support_state_zip_match", "practice_support_zip"),
            ("signal_address_line1_match", "address"),
            ("signal_rare_phone_match", "rare_phone"),
            ("signal_phone_match", "phone"),
            ("signal_org_name_strong", "org_name"),
            ("signal_city_state_match", "city_state"),
            ("signal_practice_support_city_state_match", "practice_support_city"),
            ("signal_repeat_edge", "repeat_edge"),
            ("signal_blocked_generic_system", "blocked_generic_system"),
        ]:
            mask = candidate_df[column].fillna(False).astype(bool)
            candidate_df.loc[mask, "candidate_selection_reason"] = candidate_df.loc[mask, "candidate_selection_reason"] + f"+{label}"
        candidate_df["mapping_method"] = candidate_df["candidate_selection_reason"]
        candidate_df["billing_affiliation_recency_rank"] = candidate_df["candidate_rank"].astype("int64")

        out = out.drop(
            columns=[
                "mapped_clinic_id",
                "mapped_clinic_name",
                "mapped_system_id",
                "mapped_system_name",
                "mapping_confidence",
                "mapping_method",
                "primary_affiliation_start_date",
                "primary_affiliation_end_date",
                "billing_affiliation_recency_rank",
                "shared_address_provider_count",
                "active_reassignment_count",
                "most_recent_reassignment_date",
                "longest_reassignment_days",
                "mapping_score",
                "mapping_margin_score",
                "mapping_signal_count",
                "candidate_clinic_count",
                "candidate_system_count",
                "primary_is_tied",
                "primary_is_ambiguous",
                "mapping_confidence_tier",
                "primary_selection_reason",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
                "system_rollup_quality_warning",
                "clinic_rollup_quality_warning",
                "signal_blocked_generic_system",
                "blocked_generic_penalty",
            ],
            errors="ignore",
        )
        out = out.merge(
            best[
                [
                    "provider_id",
                    "clinic_id",
                    "clinic_name",
                    "system_id",
                    "system_name",
                    "mapping_confidence",
                    "mapping_method",
                    "primary_affiliation_start_date",
                    "primary_affiliation_end_date",
                    "billing_affiliation_recency_rank",
                    "shared_address_provider_count",
                    "active_reassignment_count",
                    "most_recent_reassignment_date",
                    "longest_reassignment_days",
                    "mapping_score",
                    "mapping_margin_score",
                    "mapping_signal_count",
                    "candidate_clinic_count",
                    "candidate_system_count",
                    "primary_is_tied",
                    "primary_is_ambiguous",
                    "mapping_confidence_tier",
                    "primary_selection_reason",
                    "system_alias_action",
                    "system_alias_reason",
                    "system_alias_blocked_generic",
                    "system_rollup_quality_warning",
                    "clinic_rollup_quality_warning",
                    "signal_blocked_generic_system",
                    "blocked_generic_penalty",
                ]
            ].rename(
                columns={
                    "clinic_id": "mapped_clinic_id",
                    "clinic_name": "mapped_clinic_name",
                    "system_id": "mapped_system_id",
                    "system_name": "mapped_system_name",
                }
            ),
            on="provider_id",
            how="left",
        )
        for column in [
            "mapped_clinic_id",
            "mapped_clinic_name",
            "mapped_system_id",
            "mapped_system_name",
            "mapping_confidence",
            "mapping_method",
            "primary_affiliation_start_date",
            "primary_affiliation_end_date",
            "most_recent_reassignment_date",
            "mapping_confidence_tier",
            "primary_selection_reason",
            "system_alias_action",
            "system_alias_reason",
            "system_rollup_quality_warning",
            "clinic_rollup_quality_warning",
        ]:
            out[column] = out[column].astype("string").fillna("")
        for column in [
            "active_reassignment_count",
            "longest_reassignment_days",
            "shared_address_provider_count",
            "billing_affiliation_recency_rank",
            "mapping_signal_count",
            "candidate_clinic_count",
            "candidate_system_count",
        ]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")
        for column in ["mapping_score", "mapping_margin_score", "blocked_generic_penalty"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        for column in ["primary_is_tied", "primary_is_ambiguous", "system_alias_blocked_generic", "signal_blocked_generic_system"]:
            out[column] = out[column].fillna(False).astype(bool)
        out = self._ensure_freshness_and_provenance(out, date_str)
        out = self._ensure_comparison_fields(out)

        links_df = candidate_df.drop(columns=["raw_address_key"], errors="ignore")
        self.links_output_path(date_str).parent.mkdir(parents=True, exist_ok=True)
        links_df.to_parquet(self.links_output_path(date_str), index=False)
        links_df.to_parquet(self.candidates_output_path(date_str), index=False)
        self._write_mapping_quality_report(date_str, out, links_df, best)
        self._write_system_quality_report(date_str, best)
        self._write_field_provenance_summary(date_str, out)
        self._write_specialty_rollup_audit(date_str, out)
        self._write_unresolved_taxonomy_audit(date_str, out)
        self._write_comparison_readiness_summary(date_str, out)
        self._write_system_alias_summary(date_str)
        self._write_comparison_baseline_exports(date_str, out, links_df)
        metadata = add_mode_completion(
            metadata,
            "clinic_mapping",
            time.time() - start,
            len(out),
            [
                "mapped_clinic_id",
                "mapped_clinic_name",
                "mapped_system_id",
                "mapped_system_name",
                "mapping_confidence",
                "mapping_method",
                "mapping_score",
                "mapping_margin_score",
                "mapping_signal_count",
                "candidate_clinic_count",
                "candidate_system_count",
                "primary_is_tied",
                "primary_is_ambiguous",
                "mapping_confidence_tier",
                "primary_selection_reason",
                "ppef_snapshot_date",
                "pecos_snapshot_date",
                "npi_registry_snapshot_date",
                "specialty_source",
                "phone_source",
                "address_source",
                "clinic_source",
                "system_source",
                "taxonomy_codes_all",
                "taxonomy_descs_all",
                "specialty_normalized",
                "specialty_group",
                "provider_type_group",
                "specialty_rollup_source",
                "ppef_unresolved_taxonomy_audit",
                "comparison_phone_set",
                "comparison_address_set",
                "comparison_zip_state_set",
                "comparison_address_line1_set",
                "comparison_city_state_set",
                "comparison_street_number_zip_set",
                "comparison_readiness_tier",
            ],
        )
        return out, metadata


def main():
    parser = argparse.ArgumentParser(description="Process PPEF individuals into clinic/system mappings.")
    parser.add_argument("mode", nargs="?", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--input-parquet", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run the requested mode even if metadata marks it completed.")
    args = parser.parse_args()
    mode = args.mode or _mode_prompt()
    if mode not in {"npi_enrichment", "clinic_mapping", "quality_exports", "all"}:
        raise SystemExit(f"Invalid mode: {mode}")

    processor = IndividualProcessor(Path(args.data_dir))
    base_path = Path(args.input_parquet) if args.input_parquet else processor.find_latest_base()
    if base_path is None or not base_path.exists():
        raise SystemExit("No ppef_individuals parquet found. Run scripts/ppef_dump.py first.")

    date_str = extract_date_from_filename(base_path)
    metadata_path = processor.metadata_path(date_str)
    metadata = load_metadata(metadata_path, date_str)
    if args.force:
        modes_to_clear = {"npi_enrichment", "clinic_mapping", "quality_exports"} if mode == "all" else {mode}
        metadata["processing_modes_completed"] = [
            item for item in metadata.get("processing_modes_completed", []) if item.get("mode") not in modes_to_clear
        ]
        LOGGER.info("Force re-run requested. Cleared completion metadata for modes: %s", ", ".join(sorted(modes_to_clear)))
    orgs_signature = (
        processor.processed_orgs_signature()
        if mode in {"clinic_mapping", "quality_exports", "all"}
        else str(metadata.get("processed_orgs_signature") or "")
    )
    previous_orgs_signature = str(metadata.get("processed_orgs_signature") or "")
    if mode in {"clinic_mapping", "all"} and orgs_signature != previous_orgs_signature:
        LOGGER.info("Processed PECOS orgs changed. Re-running PPEF clinic mapping.")
        metadata["processing_modes_completed"] = [
            item for item in metadata.get("processing_modes_completed", []) if item.get("mode") != "clinic_mapping"
        ]
    df = processor.load_working_frame(base_path, date_str)
    if mode in {"clinic_mapping", "quality_exports"} and "provider_id" not in df.columns:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"npi_enrichment", "all"}:
        df, metadata = processor.npi_enrichment(df, metadata)
    if mode in {"clinic_mapping", "all"}:
        df, metadata = processor.clinic_mapping(df, metadata, date_str)
    if mode in {"quality_exports", "all"}:
        df, metadata = processor.quality_exports(df, metadata, date_str)

    metadata["total_records"] = int(len(df))
    if mode in {"clinic_mapping", "quality_exports", "all"}:
        metadata["processed_orgs_signature"] = orgs_signature
    output = processor.output_path(date_str)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    save_metadata(metadata, metadata_path)
    LOGGER.info("Saved processed individuals: %s (%s rows, %s columns)", output, f"{len(df):,}", len(df.columns))


if __name__ == "__main__":
    main()
