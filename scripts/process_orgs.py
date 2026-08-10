#!/usr/bin/env python3
"""
Process PECOS-style org/location rows into clinic/system entities.
"""

from __future__ import annotations

import argparse
import hashlib
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
    canonical_entity_name,
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
    normalize_address_line1,
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

    def clinic_rollup_path(self, data_date: str) -> Path:
        return self.processed_dir / f"pecos_clinic_rollup_{data_date}.parquet"

    def system_rollup_path(self, data_date: str) -> Path:
        return self.processed_dir / f"pecos_system_rollup_{data_date}.parquet"

    def alias_overrides_path(self) -> Path:
        return self.data_dir / "config" / "pecos_system_alias_overrides.csv"

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

    def _load_system_alias_overrides(self) -> pd.DataFrame:
        path = self.alias_overrides_path()
        columns = ["alias_name", "alias_canonical", "target_system_name", "target_system_id", "action", "status", "reason"]
        if not path.exists():
            return pd.DataFrame(columns=columns)
        aliases = pd.read_csv(path, dtype=str).fillna("")
        for column in columns:
            if column not in aliases.columns:
                aliases[column] = ""
            aliases[column] = aliases[column].astype("string").fillna("").str.strip()
        aliases["alias_canonical"] = aliases["alias_canonical"].where(
            aliases["alias_canonical"].ne(""),
            aliases["alias_name"].map(canonical_entity_name),
        )
        aliases["target_system_id"] = aliases["target_system_id"].astype("string").fillna("").str.strip()
        aliases = aliases[
            aliases["status"].str.lower().eq("approved")
            & aliases["action"].str.lower().isin(["merge_to_target", "block_generic_match"])
            & aliases["alias_canonical"].ne("")
        ].copy()
        if aliases.empty:
            return aliases
        aliases = aliases.drop_duplicates(subset=["alias_canonical", "action"], keep="first")
        LOGGER.info("Loaded %s approved PECOS system alias override rows from %s", f"{len(aliases):,}", path)
        return aliases

    def system_alias_override_signature(self) -> str:
        aliases = self._load_system_alias_overrides()
        if aliases.empty:
            return ""
        columns = ["alias_canonical", "target_system_name", "target_system_id", "action", "status"]
        payload = aliases[columns].sort_values(columns).to_csv(index=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _apply_system_alias_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        aliases = self._load_system_alias_overrides()
        out = df.copy()
        if "system_alias_action" not in out.columns:
            out["system_alias_action"] = ""
        if "system_alias_reason" not in out.columns:
            out["system_alias_reason"] = ""
        if "system_alias_blocked_generic" not in out.columns:
            out["system_alias_blocked_generic"] = False
        if aliases.empty:
            out["system_alias_blocked_generic"] = out["system_alias_blocked_generic"].fillna(False).astype(bool)
            return out

        system_key = out["system_name"].astype("string").fillna("").map(canonical_entity_name)
        merge_rows = aliases[aliases["action"].str.lower().eq("merge_to_target")].copy()
        merge_rows = merge_rows[merge_rows["target_system_id"].ne("") & merge_rows["target_system_name"].ne("")]
        if not merge_rows.empty:
            merge_map = merge_rows.drop_duplicates("alias_canonical").set_index("alias_canonical")
            matched = system_key.isin(merge_map.index)
            if matched.any():
                out.loc[matched, "system_alias_action"] = "merge_to_target"
                out.loc[matched, "system_alias_reason"] = system_key[matched].map(merge_map["reason"]).fillna("")
                out.loc[matched, "system_id"] = system_key[matched].map(merge_map["target_system_id"]).fillna(out.loc[matched, "system_id"])
                out.loc[matched, "system_name"] = system_key[matched].map(merge_map["target_system_name"]).fillna(out.loc[matched, "system_name"])
                out.loc[matched, "system_display_name"] = out.loc[matched, "system_name"]
                LOGGER.info("Applied merge_to_target PECOS system aliases to %s org rows", f"{int(matched.sum()):,}")

        block_rows = aliases[aliases["action"].str.lower().eq("block_generic_match")].copy()
        if not block_rows.empty:
            blocked_canonicals = set(block_rows["alias_canonical"])
            blocked = system_key.isin(blocked_canonicals)
            if blocked.any():
                block_map = block_rows.drop_duplicates("alias_canonical").set_index("alias_canonical")
                out.loc[blocked, "system_alias_action"] = out.loc[blocked, "system_alias_action"].where(
                    out.loc[blocked, "system_alias_action"].astype("string").fillna("").str.strip().ne(""),
                    "block_generic_match",
                )
                out.loc[blocked, "system_alias_reason"] = system_key[blocked].map(block_map["reason"]).fillna("")
                out.loc[blocked, "system_alias_blocked_generic"] = True
                LOGGER.info("Applied block_generic_match PECOS system aliases to %s org rows", f"{int(blocked.sum()):,}")

        out["system_alias_blocked_generic"] = out["system_alias_blocked_generic"].fillna(False).astype(bool)
        return out

    @staticmethod
    def _row_completeness(df: pd.DataFrame, columns: list[str]) -> pd.Series:
        score = pd.Series(0, index=df.index, dtype="int64")
        for column in columns:
            if column not in df.columns:
                continue
            score = score + df[column].astype("string").fillna("").str.strip().ne("").astype("int64")
        return score

    def _build_rollups(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        work = df.copy()
        for column in [
            "clinic_id",
            "clinic_name",
            "clinic_display_name",
            "system_id",
            "system_name",
            "system_display_name",
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
            "website",
            "website_source",
            "website_confidence",
            "system_alias_action",
            "system_alias_reason",
        ]:
            if column not in work.columns:
                work[column] = ""
            work[column] = work[column].astype("string").fillna("").str.strip()
        if "system_alias_blocked_generic" not in work.columns:
            work["system_alias_blocked_generic"] = False
        work["system_alias_blocked_generic"] = work["system_alias_blocked_generic"].fillna(False).astype(bool)
        for column in ["provider_count", "provider_count_entity"]:
            if column not in work.columns:
                work[column] = 0
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0).astype("int64")
        if "is_hospital" not in work.columns:
            work["is_hospital"] = False
        work["is_hospital"] = work["is_hospital"].fillna(False).astype(bool)
        work["clinic_name_canonical"] = work["clinic_name"].map(canonical_entity_name)
        work["system_name_canonical"] = work["system_name"].map(canonical_entity_name)
        work["clinic_generic_name_flag"] = work["clinic_name"].ne("") & (
            work["clinic_name_canonical"].eq("") | work["clinic_name_canonical"].str.split().str.len().fillna(0).le(1)
        )
        work["system_generic_name_flag"] = work["system_name"].ne("") & (
            work["system_name_canonical"].eq("") | work["system_name_canonical"].str.split().str.len().fillna(0).le(1)
        )
        work["system_generic_name_flag"] = work["system_generic_name_flag"] | work["system_alias_blocked_generic"]
        work["rollup_address_key"] = (
            work["practice_address_1"].map(normalize_address_line1)
            + "|"
            + work["practice_city"].map(normalize_text)
            + "|"
            + work["practice_state"].map(normalize_text)
            + "|"
            + work["practice_zip5"].map(zip5)
        )
        address_counts = (
            work[work["rollup_address_key"].str.replace("|", "", regex=False).ne("")]
            .groupby("rollup_address_key")["clinic_id"]
            .nunique()
        )
        work["duplicate_address_group_size"] = work["rollup_address_key"].map(address_counts).fillna(0).astype("int64")
        completeness_cols = [
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
            "website",
        ]
        work["row_completeness"] = self._row_completeness(work, completeness_cols)
        work["name_length"] = work["clinic_name"].str.len().fillna(0).astype("int64")

        clinic_stats = (
            work.groupby("clinic_id", dropna=False)
            .agg(
                clinic_row_count=("clinic_id", "size"),
                clinic_npi_count=("org_npi", lambda s: int(s[s != ""].nunique())),
                provider_count=("provider_count", "max"),
                provider_count_entity=("provider_count_entity", "max"),
                duplicate_address_group_size=("duplicate_address_group_size", "max"),
            )
            .reset_index()
        )
        clinic_best = (
            work.sort_values(
                by=["row_completeness", "is_hospital", "name_length", "provider_count", "clinic_name"],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["clinic_id"], keep="first")
            .copy()
        )
        clinic_rollup = clinic_best[
            [
                "clinic_id",
                "clinic_name",
                "clinic_display_name",
                "clinic_name_canonical",
                "system_id",
                "system_name",
                "system_display_name",
                "system_name_canonical",
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
                "website",
                "website_source",
                "website_confidence",
                "clinic_generic_name_flag",
            ]
        ].merge(clinic_stats, on="clinic_id", how="left")

        system_stats = (
            work.groupby("system_id", dropna=False)
            .agg(
                system_row_count=("system_id", "size"),
                system_npi_count=("org_npi", lambda s: int(s[s != ""].nunique())),
                provider_count_entity=("provider_count_entity", "max"),
                duplicate_address_group_size=("duplicate_address_group_size", "max"),
            )
            .reset_index()
        )
        system_best = (
            work.sort_values(
                by=["row_completeness", "is_hospital", "name_length", "provider_count_entity", "system_name"],
                ascending=[False, False, False, False, True],
                kind="mergesort",
            )
            .drop_duplicates(subset=["system_id"], keep="first")
            .copy()
        )
        system_rollup = system_best[
            [
                "system_id",
                "system_name",
                "system_display_name",
                "system_name_canonical",
                "org_entity_id",
                "org_npi",
                "practice_address_1",
                "practice_address_2",
                "practice_city",
                "practice_state",
                "practice_zip5",
                "practice_phone",
                "taxonomy_desc_primary",
                "is_hospital",
                "website",
                "website_source",
                "website_confidence",
                "system_generic_name_flag",
                "system_alias_action",
                "system_alias_reason",
                "system_alias_blocked_generic",
            ]
        ].merge(system_stats, on="system_id", how="left")
        return clinic_rollup, system_rollup

    @staticmethod
    def _add_rollup_quality_fields(rollup: pd.DataFrame, count_column: str, generic_column: str) -> pd.DataFrame:
        out = rollup.copy()
        out["provider_count_rank"] = pd.to_numeric(out[count_column], errors="coerce").fillna(0).rank(
            method="dense",
            ascending=False,
        ).astype("int64")
        out["rollup_quality_warning"] = ""
        generic_mask = out.get(generic_column, False)
        if not isinstance(generic_mask, pd.Series):
            generic_mask = pd.Series(False, index=out.index)
        generic_mask = generic_mask.fillna(False).astype(bool)
        duplicate_address_mask = pd.to_numeric(out.get("duplicate_address_group_size", 0), errors="coerce").fillna(0).gt(10)
        missing_name_mask = out[count_column].notna() & out.get("clinic_name", out.get("system_name", "")).astype("string").fillna("").str.strip().eq("")
        out.loc[generic_mask, "rollup_quality_warning"] = out.loc[generic_mask, "rollup_quality_warning"] + "generic_name;"
        out.loc[duplicate_address_mask, "rollup_quality_warning"] = out.loc[duplicate_address_mask, "rollup_quality_warning"] + "duplicate_address_cluster;"
        out.loc[missing_name_mask, "rollup_quality_warning"] = out.loc[missing_name_mask, "rollup_quality_warning"] + "missing_name;"
        out["rollup_quality_warning"] = out["rollup_quality_warning"].str.rstrip(";")
        return out

    def _apply_rollups_to_rows(
        self,
        df: pd.DataFrame,
        clinic_rollup: pd.DataFrame,
        system_rollup: pd.DataFrame,
    ) -> pd.DataFrame:
        out = df.copy()
        clinic_merge = clinic_rollup[
            [
                "clinic_id",
                "clinic_name",
                "clinic_display_name",
                "clinic_row_count",
                "clinic_npi_count",
                "provider_count",
                "rollup_quality_warning",
            ]
        ].rename(
            columns={
                "clinic_name": "_rollup_clinic_name",
                "clinic_display_name": "_rollup_clinic_display_name",
                "clinic_row_count": "_rollup_clinic_row_count",
                "clinic_npi_count": "_rollup_clinic_npi_count",
                "provider_count": "_rollup_provider_count",
                "rollup_quality_warning": "_rollup_clinic_quality_warning",
            }
        )
        system_merge = system_rollup[
            [
                "system_id",
                "system_name",
                "system_display_name",
                "system_row_count",
                "system_npi_count",
                "provider_count_entity",
                "rollup_quality_warning",
            ]
        ].rename(
            columns={
                "system_name": "_rollup_system_name",
                "system_display_name": "_rollup_system_display_name",
                "system_row_count": "_rollup_system_row_count",
                "system_npi_count": "_rollup_system_npi_count",
                "provider_count_entity": "_rollup_provider_count_entity",
                "rollup_quality_warning": "_rollup_system_quality_warning",
            }
        )
        out = out.merge(clinic_merge, on="clinic_id", how="left")
        out = out.merge(system_merge, on="system_id", how="left")
        out["clinic_name"] = coalesce_columns(out, ["_rollup_clinic_name", "clinic_name"])
        out["clinic_display_name"] = coalesce_columns(out, ["_rollup_clinic_display_name", "clinic_display_name", "clinic_name"])
        out["system_name"] = coalesce_columns(out, ["_rollup_system_name", "system_name"])
        out["system_display_name"] = coalesce_columns(out, ["_rollup_system_display_name", "system_display_name", "system_name"])
        out["clinic_row_count"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_clinic_row_count", "clinic_row_count"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["clinic_npi_count"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_clinic_npi_count", "clinic_npi_count"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["system_row_count"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_system_row_count", "system_row_count"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["system_npi_count"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_system_npi_count", "system_npi_count"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["provider_count"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_provider_count", "provider_count"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["provider_count_entity"] = pd.to_numeric(
            coalesce_columns(out, ["_rollup_provider_count_entity", "provider_count_entity"], default="0"),
            errors="coerce",
        ).fillna(0).astype("int64")
        out["clinic_rollup_quality_warning"] = coalesce_columns(
            out,
            ["_rollup_clinic_quality_warning", "clinic_rollup_quality_warning"],
        )
        out["system_rollup_quality_warning"] = coalesce_columns(
            out,
            ["_rollup_system_quality_warning", "system_rollup_quality_warning"],
        )
        return out.drop(columns=[column for column in out.columns if column.startswith("_rollup_")], errors="ignore")

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
                "entity",
                row.get("org_entity_id"),
            ),
            axis=1,
        )
        missing_entity = out["org_entity_id"].astype("string").fillna("").str.strip().eq("")
        out.loc[missing_entity, "system_id"] = out.loc[missing_entity].apply(
            lambda row: make_hash_id(
                "system",
                "fallback",
                row.get("system_name"),
                row.get("org_npi"),
                row.get("practice_state"),
            ),
            axis=1,
        )
        out["is_hospital"] = out.apply(is_hospital_from_row, axis=1)
        out = self._apply_system_alias_overrides(out)
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
            "provider_count_entity",
            "clinic_row_count",
            "system_row_count",
            "clinic_npi_count",
            "system_npi_count",
            "system_alias_action",
            "system_alias_reason",
            "clinic_rollup_quality_warning",
            "system_rollup_quality_warning",
        ]:
            if column not in out.columns:
                out[column] = 0 if column in {"provider_count", "provider_count_entity", "clinic_row_count", "system_row_count", "clinic_npi_count", "system_npi_count"} else ""
        if "system_alias_blocked_generic" not in out.columns:
            out["system_alias_blocked_generic"] = False
        out["system_alias_blocked_generic"] = out["system_alias_blocked_generic"].fillna(False).astype(bool)
        for column in ["provider_count", "provider_count_entity", "clinic_row_count", "system_row_count", "clinic_npi_count", "system_npi_count"]:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")

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
                "system_alias_action",
                "system_alias_blocked_generic",
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

        ind_df = pd.read_parquet(individuals_path, columns=["provider_id", "mapped_clinic_id", "mapped_system_id"]).copy()
        ind_df["provider_id"] = ind_df["provider_id"].astype("string").fillna("").str.strip()
        ind_df["mapped_clinic_id"] = ind_df["mapped_clinic_id"].astype("string").fillna("").str.strip()
        ind_df = ind_df[(ind_df["provider_id"] != "") & (ind_df["mapped_clinic_id"] != "")]
        clinic_counts = ind_df.groupby("mapped_clinic_id")["provider_id"].nunique()
        out["provider_count"] = out["clinic_id"].astype("string").map(clinic_counts).fillna(0).astype("int64")
        ind_system_df = ind_df[["provider_id", "mapped_system_id"]].copy()
        ind_system_df["provider_id"] = ind_system_df["provider_id"].astype("string").fillna("").str.strip()
        ind_system_df["mapped_system_id"] = ind_system_df["mapped_system_id"].astype("string").fillna("").str.strip()
        ind_system_df = ind_system_df[(ind_system_df["provider_id"] != "") & (ind_system_df["mapped_system_id"] != "")]
        system_counts = ind_system_df.groupby("mapped_system_id")["provider_id"].nunique()
        out["provider_count_entity"] = out["system_id"].astype("string").map(system_counts).fillna(0).astype("int64")
        clinic_rollup, system_rollup = self._build_rollups(out)
        clinic_rollup["provider_count"] = clinic_rollup["clinic_id"].astype("string").map(clinic_counts).fillna(0).astype("int64")
        system_rollup["provider_count_entity"] = system_rollup["system_id"].astype("string").map(system_counts).fillna(0).astype("int64")
        clinic_rollup = self._add_rollup_quality_fields(clinic_rollup, "provider_count", "clinic_generic_name_flag")
        system_rollup = self._add_rollup_quality_fields(system_rollup, "provider_count_entity", "system_generic_name_flag")
        out = self._apply_rollups_to_rows(out, clinic_rollup, system_rollup)
        clinic_key = out["clinic_id"].astype("string").fillna("").str.strip()
        system_key = out["system_id"].astype("string").fillna("").str.strip()
        clinic_rollup_indexed = clinic_rollup.copy()
        clinic_rollup_indexed.index = clinic_rollup_indexed["clinic_id"].astype("string").fillna("").str.strip()
        system_rollup_indexed = system_rollup.copy()
        system_rollup_indexed.index = system_rollup_indexed["system_id"].astype("string").fillna("").str.strip()
        out["clinic_name"] = clinic_key.map(clinic_rollup_indexed["clinic_name"]).fillna(out["clinic_name"])
        out["clinic_display_name"] = clinic_key.map(clinic_rollup_indexed["clinic_display_name"]).fillna(out["clinic_display_name"])
        out["system_name"] = system_key.map(system_rollup_indexed["system_name"]).fillna(out["system_name"])
        out["system_display_name"] = system_key.map(system_rollup_indexed["system_display_name"]).fillna(out["system_display_name"])
        out["provider_count"] = clinic_key.map(clinic_rollup_indexed["provider_count"]).fillna(0).astype("int64")
        out["provider_count_entity"] = system_key.map(system_rollup_indexed["provider_count_entity"]).fillna(0).astype("int64")
        out["clinic_row_count"] = clinic_key.map(clinic_rollup_indexed["clinic_row_count"]).fillna(0).astype("int64")
        out["clinic_npi_count"] = clinic_key.map(clinic_rollup_indexed["clinic_npi_count"]).fillna(0).astype("int64")
        out["system_row_count"] = system_key.map(system_rollup_indexed["system_row_count"]).fillna(0).astype("int64")
        out["system_npi_count"] = system_key.map(system_rollup_indexed["system_npi_count"]).fillna(0).astype("int64")
        self.clinic_rollup_path(metadata.get("date") or datetime.now().strftime("%Y%m%d")).parent.mkdir(parents=True, exist_ok=True)
        clinic_rollup.to_parquet(self.clinic_rollup_path(metadata.get("date") or datetime.now().strftime("%Y%m%d")), index=False)
        system_rollup.to_parquet(self.system_rollup_path(metadata.get("date") or datetime.now().strftime("%Y%m%d")), index=False)

        metadata = add_mode_completion(
            metadata,
            "provider_count",
            time.time() - start,
            len(out),
            ["provider_count", "provider_count_entity", "clinic_row_count", "system_row_count", "clinic_npi_count", "system_npi_count"],
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

        need_lookup = out["website"].eq("") & out["is_hospital"].fillna(False).astype(bool)
        if offline_only:
            LOGGER.info("Skipping Overpass lookup in offline-only mode; using HGI-derived websites and local Nominatim cache only.")
        else:
            overpass = load_overpass(self.data_dir)
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
            cache_path = self.hash_dir / f"osm_website_cache_pecos_{date_str}.json"
            if offline_only and not cache_path.exists():
                LOGGER.info(
                    "Skipping offline Nominatim fallback because cache does not exist: %s",
                    cache_path,
                )
            else:
                client = NominatimClient(cache_path=cache_path, offline_only=offline_only)
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
                client.flush()

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
    parser.add_argument("--force", action="store_true", help="Re-run the requested mode even if metadata marks it completed.")
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
    if args.force:
        modes_to_clear = valid_modes - {"all"} if mode == "all" else {mode}
        metadata["processing_modes_completed"] = [
            item for item in metadata.get("processing_modes_completed", []) if item.get("mode") not in modes_to_clear
        ]
        LOGGER.info("Force re-run requested. Cleared completion metadata for modes: %s", ", ".join(sorted(modes_to_clear)))
    alias_signature = processor.system_alias_override_signature()
    previous_alias_signature = str(metadata.get("system_alias_override_signature") or "")
    if alias_signature != previous_alias_signature:
        LOGGER.info("Approved system alias overrides changed. Rebuilding org processing from base PECOS parquet.")
        dependent_modes = {"npi_enrichment", "hgi_enrichment", "provider_count", "website_mapping"}
        metadata["processing_modes_completed"] = [
            item for item in metadata.get("processing_modes_completed", []) if item.get("mode") not in dependent_modes
        ]
        LOGGER.info("Loading base org parquet: %s", base_path)
        df = pd.read_parquet(base_path)
    else:
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
    metadata["system_alias_override_signature"] = alias_signature
    output_path = processor.output_path(data_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    save_metadata(metadata, metadata_path)
    LOGGER.info("Saved processed orgs: %s (%s rows, %s columns)", output_path, f"{len(df):,}", len(df.columns))


if __name__ == "__main__":
    main()
