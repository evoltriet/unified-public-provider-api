"""Data loading and canonical schema adapters for the provider API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)


CMS_PROVIDER_COLUMNS = [
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
    "mapping_confidence",
    "mapping_confidence_tier",
    "mapping_method",
    "mapping_score",
    "mapping_margin_score",
    "mapping_signal_count",
    "primary_is_tied",
    "primary_is_ambiguous",
    "relationship_source",
    "relationship_start_date",
    "relationship_end_date",
    "is_active_relationship",
    "cms_current_active_flag",
    "cms_currentness_tier",
    "cms_currentness_reason",
    "cms_active_as_of_date",
    "ppef_snapshot_date",
    "pecos_snapshot_date",
    "npi_registry_snapshot_date",
    "provider_identity_source",
    "specialty_source",
    "phone_source",
    "address_source",
    "clinic_source",
    "system_source",
]

CLINIC_COLUMNS = [
    "clinic_id",
    "clinic_name",
    "clinic_display_name",
    "system_id",
    "system_name",
    "system_display_name",
    "org_npi",
    "org_entity_id",
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
    "provider_count",
    "provider_count_entity",
    "clinic_row_count",
    "clinic_npi_count",
    "duplicate_address_group_size",
    "rollup_quality_warning",
]

SYSTEM_COLUMNS = [
    "system_id",
    "system_name",
    "system_display_name",
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
    "provider_count_entity",
    "system_row_count",
    "system_npi_count",
    "duplicate_address_group_size",
    "rollup_quality_warning",
]


def text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[column].astype("string").fillna("").str.strip()


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y"})
    )


def first_nonempty(df: pd.DataFrame, column: str) -> str:
    values = text_series(df, column)
    values = values[values.ne("")]
    return str(values.iloc[0]) if not values.empty else ""


def latest_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern)) if directory.exists() else []
    return matches[-1] if matches else None


def resolve_path(explicit: str | None, directory: Path, pattern: str) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    return latest_file(directory, pattern)


def read_available_columns(
    path: Path,
    requested: Iterable[str],
    filters: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    columns = [column for column in requested if column in available]
    return pd.read_parquet(path, columns=columns, filters=filters)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = ""
    return out


def _zip5(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.extract(r"(\d{5})", expand=False)
    return values.fillna("")


def prepare_provider_search_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_columns = [
        "provider_id",
        "npi",
        "provider_full_name",
        "first_name",
        "last_name",
        "taxonomy_code_primary",
        "taxonomy_desc_primary",
        "taxonomy_codes_all",
        "taxonomy_descs_all",
        "specialty_normalized",
        "specialty_group",
        "provider_type_group",
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
    ]
    out = _ensure_columns(out, text_columns)
    for column in text_columns:
        out[column] = text_series(out, column)
    out["provider_practice_state"] = out["provider_practice_state"].str.upper()
    out["mapped_practice_state"] = out["mapped_practice_state"].str.upper()
    out["provider_practice_zip5"] = _zip5(out["provider_practice_zip5"])
    out["mapped_practice_zip5"] = _zip5(out["mapped_practice_zip5"])
    out["_search_name"] = out["provider_full_name"].str.upper()
    specialty_columns = [
        "taxonomy_code_primary",
        "taxonomy_desc_primary",
        "taxonomy_codes_all",
        "taxonomy_descs_all",
        "specialty_normalized",
        "specialty_group",
        "provider_type_group",
    ]
    specialty_search = out[specialty_columns[0]]
    for column in specialty_columns[1:]:
        specialty_search = specialty_search.str.cat(out[column], sep=" | ")
    out["_search_specialty"] = specialty_search.str.upper()
    out["_search_affiliation"] = out["mapped_clinic_name"].str.cat(
        out["mapped_system_name"], sep=" | "
    ).str.upper()
    out["_provider_city"] = out["provider_practice_city"].str.upper()
    out["_mapped_city"] = out["mapped_practice_city"].str.upper()
    return out


def prepare_organization_search_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = df.copy()
    if kind == "clinic":
        out = _ensure_columns(out, CLINIC_COLUMNS)
        out["_entity_id"] = text_series(out, "clinic_id")
        out["_display_name"] = text_series(out, "clinic_display_name").where(
            text_series(out, "clinic_display_name").ne(""), text_series(out, "clinic_name")
        )
    else:
        out = _ensure_columns(out, SYSTEM_COLUMNS)
        out["_entity_id"] = text_series(out, "system_id")
        out["_display_name"] = text_series(out, "system_display_name").where(
            text_series(out, "system_display_name").ne(""), text_series(out, "system_name")
        )
    for column in [
        "practice_address_1",
        "practice_address_2",
        "practice_city",
        "practice_state",
        "practice_zip5",
        "practice_phone",
        "website",
        "website_source",
        "website_confidence",
        "rollup_quality_warning",
    ]:
        out[column] = text_series(out, column)
    out["practice_state"] = out["practice_state"].str.upper()
    out["practice_zip5"] = _zip5(out["practice_zip5"])
    out["is_hospital"] = bool_series(out, "is_hospital")
    out["_search_name"] = out["_display_name"].str.upper()
    out["_search_city"] = out["practice_city"].str.upper()
    out["_search_address"] = out["practice_address_1"].str.upper()
    return out


def adapt_legacy_providers(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["baseline_view"] = "legacy_npi"
    out["provider_id"] = text_series(df, "NPI")
    out["npi"] = text_series(df, "NPI")
    out["enrollment_id"] = ""
    first = text_series(df, "Provider First Name")
    middle = text_series(df, "Provider Middle Name")
    last = text_series(df, "Provider Last Name (Legal Name)")
    out["first_name"] = first
    out["middle_name"] = middle
    out["last_name"] = last
    out["provider_full_name"] = (first + " " + middle + " " + last).str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    taxonomy_columns = [
        column for column in df.columns if column.startswith("Healthcare Provider Taxonomy Code_")
    ]
    out["taxonomy_code_primary"] = text_series(df, "Healthcare Provider Taxonomy Code_1")
    out["taxonomy_desc_primary"] = ""
    out["taxonomy_codes_all"] = (
        df[taxonomy_columns].astype("string").fillna("").agg(" | ".join, axis=1)
        if taxonomy_columns
        else ""
    )
    out["taxonomy_descs_all"] = ""
    out["specialty_normalized"] = ""
    out["specialty_group"] = ""
    out["provider_type_group"] = ""
    mapping = {
        "provider_practice_address_1": "Provider First Line Business Practice Location Address",
        "provider_practice_address_2": "Provider Second Line Business Practice Location Address",
        "provider_practice_city": "Provider Business Practice Location Address City Name",
        "provider_practice_state": "Provider Business Practice Location Address State Name",
        "provider_practice_zip5": "Provider Business Practice Location Address Postal Code",
        "provider_practice_phone": "Provider Business Practice Location Address Telephone Number",
    }
    for target, source in mapping.items():
        out[target] = text_series(df, source)
    out["mapped_clinic_id"] = ""
    out["mapped_clinic_name"] = text_series(df, "mapped_org_name")
    out["mapped_system_id"] = ""
    out["mapped_system_name"] = text_series(df, "mapped_org_name")
    for suffix in ["address_1", "address_2", "city", "state", "zip5", "phone"]:
        out[f"mapped_practice_{suffix}"] = ""
    out["cms_current_active_flag"] = False
    out["cms_currentness_tier"] = "legacy_unknown"
    out["provider_identity_source"] = "NPPES"
    out["specialty_source"] = "NPPES"
    out["phone_source"] = "NPPES"
    out["address_source"] = "NPPES"
    return prepare_provider_search_columns(out)


def adapt_legacy_systems(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["system_id"] = text_series(df, "org_entity_id").where(
        text_series(df, "org_entity_id").ne(""), text_series(df, "NPI")
    )
    out["system_name"] = text_series(df, "Provider Organization Name (Legal Business Name)")
    out["system_display_name"] = out["system_name"]
    out["org_entity_id"] = text_series(df, "org_entity_id")
    out["org_npi"] = text_series(df, "NPI")
    mapping = {
        "practice_address_1": "Provider First Line Business Practice Location Address",
        "practice_address_2": "Provider Second Line Business Practice Location Address",
        "practice_city": "Provider Business Practice Location Address City Name",
        "practice_state": "Provider Business Practice Location Address State Name",
        "practice_zip5": "Provider Business Practice Location Address Postal Code",
        "practice_phone": "Provider Business Practice Location Address Telephone Number",
    }
    for target, source in mapping.items():
        out[target] = text_series(df, source)
    out["is_hospital"] = bool_series(df, "is_hospital", default=True)
    out["website"] = text_series(df, "system_homepage_url")
    out["website_source"] = "legacy_npi"
    out["website_confidence"] = ""
    out["provider_count_entity"] = pd.to_numeric(
        df["provider_count_entity"] if "provider_count_entity" in df.columns else 0,
        errors="coerce",
    ).fillna(0).astype("int64")
    return prepare_organization_search_columns(out, "system")


@dataclass
class DataStore:
    providers: pd.DataFrame | None = None
    clinics: pd.DataFrame | None = None
    systems: pd.DataFrame | None = None
    source: str = ""
    baseline_view: str = ""
    provider_path: Path | None = None
    clinic_path: Path | None = None
    system_path: Path | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def providers_loaded(self) -> bool:
        return self.providers is not None and not self.providers.empty

    @property
    def clinics_loaded(self) -> bool:
        return self.clinics is not None and not self.clinics.empty

    @property
    def systems_loaded(self) -> bool:
        return self.systems is not None and not self.systems.empty

    def clear(self) -> None:
        self.providers = None
        self.clinics = None
        self.systems = None
        self.source = ""
        self.baseline_view = ""
        self.provider_path = None
        self.clinic_path = None
        self.system_path = None
        self.metadata = {}

    def load(self, data_dir: Path) -> bool:
        requested = os.getenv("API_DATA_SOURCE", "cms").strip().lower() or "cms"
        allow_fallback = os.getenv("ALLOW_LEGACY_NPI_FALLBACK", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.clear()
        if requested not in {"cms", "legacy_npi"}:
            raise ValueError("API_DATA_SOURCE must be 'cms' or 'legacy_npi'")
        if requested == "cms":
            try:
                self._load_cms(data_dir)
                return True
            except Exception:
                LOGGER.exception("Failed to load CMS API artifacts")
                if not allow_fallback:
                    return False
                LOGGER.warning("Falling back to legacy NPI API artifacts")
        return self._load_legacy(data_dir)

    def _load_cms(self, data_dir: Path) -> None:
        processed = data_dir / "processed_data"
        provider_path = resolve_path(
            os.getenv("CMS_PROVIDER_PATH"),
            processed,
            "ppef_individuals_comparison_baseline_*.parquet",
        )
        clinic_path = resolve_path(
            os.getenv("CMS_CLINIC_PATH"), processed, "pecos_clinic_rollup_*.parquet"
        )
        system_path = resolve_path(
            os.getenv("CMS_SYSTEM_PATH"), processed, "pecos_system_rollup_*.parquet"
        )
        missing = [
            name
            for name, path in [
                ("CMS provider baseline", provider_path),
                ("PECOS clinic rollup", clinic_path),
                ("PECOS system rollup", system_path),
            ]
            if path is None
        ]
        if missing:
            raise FileNotFoundError("Missing " + ", ".join(missing))

        view = os.getenv("CMS_BASELINE_VIEW", "current_active_primary").strip()
        providers = read_available_columns(
            provider_path,
            CMS_PROVIDER_COLUMNS,
            filters=[("baseline_view", "==", view)],
        )
        if providers.empty:
            raise RuntimeError(f"CMS baseline view {view!r} contains no rows")
        providers = prepare_provider_search_columns(providers)
        provider_ids = text_series(providers, "provider_id")
        npis = text_series(providers, "npi")
        if provider_ids.eq("").any() or npis.eq("").any():
            raise RuntimeError(f"CMS baseline view {view!r} contains blank provider IDs or NPIs")
        if provider_ids.duplicated().any() or npis.duplicated().any():
            raise RuntimeError(
                f"CMS baseline view {view!r} is not one row per provider/NPI and is not API-safe"
            )

        clinics = prepare_organization_search_columns(
            read_available_columns(clinic_path, CLINIC_COLUMNS), "clinic"
        )
        systems = prepare_organization_search_columns(
            read_available_columns(system_path, SYSTEM_COLUMNS), "system"
        )
        self.providers = providers.reset_index(drop=True)
        self.clinics = clinics.reset_index(drop=True)
        self.systems = systems.reset_index(drop=True)
        self.source = "cms"
        self.baseline_view = view
        self.provider_path = provider_path
        self.clinic_path = clinic_path
        self.system_path = system_path
        self.metadata = {
            "ppef_snapshot_date": first_nonempty(providers, "ppef_snapshot_date"),
            "pecos_snapshot_date": first_nonempty(providers, "pecos_snapshot_date"),
            "npi_registry_snapshot_date": first_nonempty(providers, "npi_registry_snapshot_date"),
        }
        LOGGER.info(
            "Loaded CMS API data: providers=%s clinics=%s systems=%s view=%s",
            f"{len(providers):,}",
            f"{len(clinics):,}",
            f"{len(systems):,}",
            view,
        )

    def _load_legacy(self, data_dir: Path) -> bool:
        processed = data_dir / "processed_data"
        parquet = data_dir / "parquet"
        provider_path = latest_file(processed, "npi_individuals_processed_*.parquet")
        provider_path = provider_path or latest_file(parquet, "npi_individuals_*.parquet")
        system_path = latest_file(processed, "npi_organizations_processed_*.parquet")
        system_path = system_path or latest_file(processed, "npi_origanizations_processed_*.parquet")
        system_path = system_path or latest_file(parquet, "npi_organizations_*.parquet")
        if provider_path is None:
            LOGGER.error("No legacy NPI individual parquet found")
            return False
        provider_columns = [
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
            "mapped_org_name",
        ] + [f"Healthcare Provider Taxonomy Code_{index}" for index in range(1, 16)]
        providers = adapt_legacy_providers(read_available_columns(provider_path, provider_columns))
        systems = pd.DataFrame()
        if system_path is not None:
            system_columns = [
                "NPI",
                "Provider Organization Name (Legal Business Name)",
                "Provider First Line Business Practice Location Address",
                "Provider Second Line Business Practice Location Address",
                "Provider Business Practice Location Address City Name",
                "Provider Business Practice Location Address State Name",
                "Provider Business Practice Location Address Postal Code",
                "Provider Business Practice Location Address Telephone Number",
                "is_hospital",
                "system_homepage_url",
                "org_entity_id",
                "provider_count_entity",
            ]
            systems = adapt_legacy_systems(read_available_columns(system_path, system_columns))
        self.providers = providers.reset_index(drop=True)
        self.clinics = pd.DataFrame()
        self.systems = systems.reset_index(drop=True)
        self.source = "legacy_npi"
        self.baseline_view = "legacy_npi"
        self.provider_path = provider_path
        self.system_path = system_path
        self.metadata = {}
        LOGGER.info(
            "Loaded legacy NPI API data: providers=%s systems=%s",
            f"{len(providers):,}",
            f"{len(systems):,}",
        )
        return True


STORE = DataStore()
