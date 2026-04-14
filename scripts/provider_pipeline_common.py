#!/usr/bin/env python3
"""
Shared helpers for the PECOS/PPEF processing scripts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq

HOSPITAL_TAXONOMY_CODES = {"282N00000X", "283Q00000X", "284300000X", "283X00000X"}


def ensure_dirs(data_dir: Path):
    for directory in [
        data_dir,
        data_dir / "parquet",
        data_dir / "processed_data",
        data_dir / "hashes",
        data_dir / "meta",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def extract_date_from_filename(path: Path) -> str:
    match = re.search(r"(\d{8})", path.name)
    return match.group(1) if match else datetime.now().strftime("%Y%m%d")


def find_latest_parquet(directory: Path, glob_pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(glob_pattern))
    return files[-1] if files else None


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def read_projected_parquet(path: Path, candidate_columns: Iterable[str] | None = None) -> pd.DataFrame:
    if candidate_columns is None:
        return pd.read_parquet(path)
    available = set(parquet_columns(path))
    selected = [column for column in candidate_columns if column in available]
    return pd.read_parquet(path, columns=selected if selected else None)


def load_metadata(metadata_path: Path, data_date: str) -> Dict:
    if metadata_path.exists():
        try:
            return json.loads(metadata_path.read_text())
        except Exception:
            pass
    return {
        "date": data_date,
        "created_at": datetime.now().isoformat(),
        "total_records": 0,
        "processing_modes_completed": [],
    }


def save_metadata(metadata: Dict, metadata_path: Path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))


def get_completed_modes(metadata: Dict) -> list[str]:
    return [item["mode"] for item in metadata.get("processing_modes_completed", [])]


def is_mode_completed(mode: str, metadata: Dict) -> bool:
    return mode in get_completed_modes(metadata)


def add_mode_completion(
    metadata: Dict,
    mode: str,
    duration_seconds: float,
    changed_records: int,
    columns_added: list[str],
) -> Dict:
    completed = get_completed_modes(metadata)
    if mode in completed:
        metadata["processing_modes_completed"] = [
            item for item in metadata.get("processing_modes_completed", []) if item.get("mode") != mode
        ]
    metadata.setdefault("processing_modes_completed", []).append(
        {
            "mode": mode,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 1),
            "changed_records": int(changed_records),
            "columns_added": columns_added,
        }
    )
    return metadata


def safe_str(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def safe_phone(value) -> Optional[str]:
    text = safe_str(value)
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    return digits if digits else None


def zip5(value) -> str:
    text = safe_str(value) or ""
    digits = re.sub(r"\D", "", text)
    return digits[:5]


def normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(value) -> set[str]:
    return {token for token in normalize_text(value).split(" ") if token}


def name_similarity(left, right) -> float:
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def address_similarity(left, right) -> float:
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def normalize_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse, urlunparse

    value = safe_str(url)
    if not value:
        return None
    if not re.match(r"^https?://", value, flags=re.I):
        value = "https://" + value
    try:
        parsed = urlparse(value)
        return urlunparse(("https", (parsed.netloc or "").lower(), parsed.path or "/", "", "", ""))
    except Exception:
        return value


def root_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse, urlunparse

    value = normalize_url(url)
    if not value:
        return None
    parsed = urlparse(value)
    if not parsed.netloc:
        return None
    return urlunparse(("https", parsed.netloc.lower(), "/", "", "", ""))


def location_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse

    value = normalize_url(url)
    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.path and parsed.path != "/" else None


def combine_address(*parts) -> str:
    return ", ".join([safe_str(part) for part in parts if safe_str(part)])


def make_hash_id(prefix: str, *parts) -> str:
    normalized = [normalize_text(part) for part in parts if normalize_text(part)]
    if not normalized:
        normalized = ["missing"]
    payload = json.dumps(normalized, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def coalesce_columns(df: pd.DataFrame, columns: Iterable[str], default: str = "") -> pd.Series:
    series = pd.Series(default, index=df.index, dtype="string")
    for column in columns:
        if column not in df.columns:
            continue
        current = df[column].astype("string").fillna("").str.strip()
        series = series.where(series != "", current)
    return series.fillna(default)


def parse_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("string").replace({"": pd.NA, "nan": pd.NA}), errors="coerce")


def get_taxonomy_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if "Healthcare Provider Taxonomy Code" in column]


def resolve_primary_taxonomy_code(df: pd.DataFrame) -> pd.Series:
    taxonomy_cols = get_taxonomy_columns(df)
    if not taxonomy_cols:
        return pd.Series("", index=df.index, dtype="string")

    primary_switch_cols = [c for c in df.columns if "Healthcare Provider Primary Taxonomy Switch" in c]
    code = pd.Series("", index=df.index, dtype="string")
    if primary_switch_cols:
        for idx, column in enumerate(primary_switch_cols, start=1):
            code_col = f"Healthcare Provider Taxonomy Code_{idx}"
            if code_col not in df.columns:
                continue
            switch = df[column].astype("string").fillna("").str.upper().str.strip()
            values = df[code_col].astype("string").fillna("").str.strip()
            code = code.where(code != "", values.where(switch == "Y", ""))

    if (code == "").all():
        code = coalesce_columns(df, taxonomy_cols)
    else:
        code = code.where(code != "", coalesce_columns(df, taxonomy_cols))
    return code.fillna("")


def load_taxonomy_lookup(data_dir: Path) -> dict[str, str]:
    path = data_dir / "taxonomy_lookup.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_primary_taxonomy_description(df: pd.DataFrame, taxonomy_lookup: dict[str, str]) -> pd.Series:
    direct_candidates = [
        "Healthcare Provider Primary Taxonomy Description",
        "taxonomy_desc_primary",
        "taxonomy_desc",
    ]
    direct = coalesce_columns(df, [c for c in direct_candidates if c in df.columns])
    code = resolve_primary_taxonomy_code(df)
    mapped = code.map(lambda item: taxonomy_lookup.get(item, item) if item else "")
    return direct.where(direct != "", mapped).fillna("")


def is_hospital_from_row(row: pd.Series) -> bool:
    code = safe_str(row.get("taxonomy_code_primary"))
    if code and code in HOSPITAL_TAXONOMY_CODES:
        return True
    name = safe_str(row.get("system_name")) or safe_str(row.get("clinic_name")) or safe_str(row.get("org_name"))
    if name:
        lowered = name.lower()
        if any(token in lowered for token in ["hospital", "medical center", "health system", "healthcare system", "memorial"]):
            return True
    return False


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def find_column_by_tokens(
    df: pd.DataFrame,
    *,
    required_tokens: Iterable[str],
    any_tokens: Iterable[str] | None = None,
    exclude_tokens: Iterable[str] | None = None,
) -> Optional[str]:
    required = {normalize_text(token) for token in required_tokens if normalize_text(token)}
    allowed_any = {normalize_text(token) for token in (any_tokens or []) if normalize_text(token)}
    excluded = {normalize_text(token) for token in (exclude_tokens or []) if normalize_text(token)}

    best_match: Optional[str] = None
    best_score = -1
    for column in df.columns:
        column_text = normalize_text(column)
        column_tokens = set(column_text.split())
        if required and not required.issubset(column_tokens):
            continue
        if excluded and column_tokens & excluded:
            continue
        if allowed_any and not (column_tokens & allowed_any):
            continue
        score = len(column_tokens & required) + len(column_tokens & allowed_any)
        if score > best_score:
            best_match = column
            best_score = score
    return best_match
