#!/usr/bin/env python3
"""
Shared helpers for the public CMS enrollment dumps used by PPEF/PECOS scripts.
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tqdm import tqdm

CMS_DATA_JSON_URL = "https://data.cms.gov/data.json"
PPEF_DATA_PAGE = (
    "https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/"
    "medicare-fee-for-service-public-provider-enrollment/data"
)
PPEF_LANDING_PAGE = PPEF_DATA_PAGE.removesuffix("/data")
PPEF_DATASET_TITLE = "Medicare Fee-For-Service Public Provider Enrollment"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
CSV_READ_ENCODINGS = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PARQUET_DIR = BASE_DIR / "parquet"
META_DIR = BASE_DIR / "meta"
LOGS_DIR = Path("logs")


@dataclass(frozen=True)
class DownloadAsset:
    name: str
    url: str

    @property
    def filename(self) -> str:
        return self.url.rstrip("/").split("/")[-1]


def ensure_directories():
    for directory in [RAW_DIR, PARQUET_DIR, META_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def build_logger(name: str, log_filename: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOGS_DIR / log_filename, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def http_get(
    session: requests.Session,
    url: str,
    desc: str = "Downloading",
    chunk_size: int = 1024 * 1024,
) -> bytes:
    with session.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    buf.write(chunk)
                    pbar.update(len(chunk))
        return buf.getvalue()


def get_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def schema_meta_path(stem: str) -> Path:
    return META_DIR / f"{stem}_schema.json"


def warn_if_schema_drift(df: pd.DataFrame, stem: str, logger: logging.Logger):
    meta_path = schema_meta_path(stem)
    cols = list(df.columns)
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            prev_cols = prev.get("columns", [])
            if set(prev_cols) != set(cols):
                logger.warning(
                    "SCHEMA DRIFT for %s. Previous columns=%s Current columns=%s",
                    stem,
                    prev_cols,
                    cols,
                )
        except Exception as exc:
            logger.warning("Could not read previous schema for %s: %s", stem, exc)

    meta = {
        "dataset": stem,
        "columns": cols,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def save_parquet_from_csv(
    csv_path: Path,
    parquet_path: Path,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False, encoding=encoding)
            if logger is not None and encoding != "utf-8":
                logger.warning(
                    "Decoded %s using fallback encoding %s",
                    csv_path.name,
                    encoding,
                )
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        if last_error is not None:
            raise last_error
        raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    return df


def _normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _normalize_landing_page(url: str | None) -> str:
    normalized = (url or "").strip().split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if normalized.endswith("/data"):
        normalized = normalized[: -len("/data")]
    return normalized.casefold()


def _sort_records_by_modified(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: (
            str(record.get("modified") or ""),
            str(record.get("title") or ""),
        ),
        reverse=True,
    )


def resolve_catalog_dataset(
    catalog_payload: dict[str, Any],
    *,
    landing_page: str = PPEF_LANDING_PAGE,
    dataset_title: str = PPEF_DATASET_TITLE,
) -> dict[str, Any] | None:
    datasets = catalog_payload.get("dataset")
    if not isinstance(datasets, list):
        return None

    dataset_records = [dataset for dataset in datasets if isinstance(dataset, dict)]
    target_page = _normalize_landing_page(landing_page)
    target_title = _normalize_space(dataset_title)

    page_matches = [
        dataset
        for dataset in dataset_records
        if _normalize_landing_page(dataset.get("landingPage")) == target_page
    ]
    if not page_matches and target_page:
        slug = target_page.rsplit("/", 1)[-1]
        page_matches = [
            dataset
            for dataset in dataset_records
            if slug and slug in _normalize_landing_page(dataset.get("landingPage"))
        ]
    if page_matches:
        title_matches = [
            dataset
            for dataset in page_matches
            if _normalize_space(dataset.get("title")) == target_title
        ]
        return _sort_records_by_modified(title_matches or page_matches)[0]

    title_matches = [
        dataset
        for dataset in dataset_records
        if _normalize_space(dataset.get("title")) == target_title
    ]
    return _sort_records_by_modified(title_matches)[0] if title_matches else None


def select_latest_distribution(dataset_payload: dict[str, Any]) -> dict[str, Any] | None:
    distributions = dataset_payload.get("distribution")
    if not isinstance(distributions, list):
        return None

    distribution_records = [record for record in distributions if isinstance(record, dict)]
    latest = [
        record
        for record in distribution_records
        if _normalize_space(record.get("description")) == "latest" and record.get("resourcesAPI")
    ]
    if latest:
        return _sort_records_by_modified(latest)[0]

    with_resources_api = [record for record in distribution_records if record.get("resourcesAPI")]
    return _sort_records_by_modified(with_resources_api)[0] if with_resources_api else None


def extract_resource_downloads(resource_payload: dict[str, Any]) -> list[DownloadAsset]:
    resources = resource_payload.get("data")
    if not isinstance(resources, list):
        return []

    assets: list[DownloadAsset] = []
    seen_urls: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        url = str(resource.get("downloadURL") or "").strip()
        if not url or url in seen_urls:
            continue
        name = str(resource.get("name") or url.rsplit("/", 1)[-1]).strip()
        seen_urls.add(url)
        assets.append(DownloadAsset(name=name, url=url))
    return assets


def filter_resource_downloads(
    assets: Iterable[DownloadAsset], file_patterns: Iterable[str]
) -> list[DownloadAsset]:
    patterns = list(file_patterns)
    filtered: list[DownloadAsset] = []
    seen_urls: set[str] = set()
    for asset in assets:
        if asset.url in seen_urls:
            continue
        asset_kind = classify_download_asset(asset)
        file_ext = Path(asset.filename).suffix.lower()
        filename_match = file_ext in {".csv", ".zip"} and any(
            re.search(pattern, asset.filename, re.I) for pattern in patterns
        )
        semantic_match = asset_kind in {"enrollment", "reassignment", "practice_location"}
        if filename_match or semantic_match:
            seen_urls.add(asset.url)
            filtered.append(asset)
    return filtered


def classify_download_asset(asset: DownloadAsset) -> str:
    label = f"{asset.name} {asset.filename}".casefold()
    file_ext = Path(asset.filename).suffix.lower()
    if file_ext not in {".csv", ".zip"}:
        return "other"
    if re.search(r"data dictionary|methodology|fact sheet|historical", label, re.I):
        return "other"
    if re.search(r"additional\s+npis|secondary\s+specialty", label, re.I):
        return "other"
    if re.search(r"reassign", label, re.I):
        return "reassignment"
    if re.search(r"practice|location|address\s+sub[- ]file", label, re.I):
        return "practice_location"
    if re.search(r"enroll", label, re.I):
        return "enrollment"
    return "other"


def extract_and_convert_csvs(
    asset_path: Path,
    out_dir: Path,
    file_patterns: Iterable[str],
    logger: logging.Logger,
) -> list[tuple[Path, Path]]:
    patterns = list(file_patterns)
    converted: list[tuple[Path, Path]] = []

    def _accept(name: str) -> bool:
        return name.lower().endswith(".csv") and any(re.search(pattern, name, re.I) for pattern in patterns)

    if asset_path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(asset_path, "r") as archive:
                for member in archive.namelist():
                    if not _accept(member):
                        continue
                    csv_path = out_dir / Path(member).name
                    if not csv_path.exists():
                        logger.info("Extracting %s -> %s", member, csv_path)
                        with archive.open(member) as src, open(csv_path, "wb") as dst:
                            dst.write(src.read())
                    stem = csv_path.stem
                    parquet_path = PARQUET_DIR / f"{stem}.parquet"
                    df = save_parquet_from_csv(csv_path, parquet_path, logger=logger)
                    warn_if_schema_drift(df, stem, logger)
                    converted.append((csv_path, parquet_path))
        except zipfile.BadZipFile:
            logger.error("Bad ZIP file for CMS enrollment asset: %s", asset_path)
    elif asset_path.suffix.lower() == ".csv" and _accept(asset_path.name):
        stem = asset_path.stem
        parquet_path = PARQUET_DIR / f"{stem}.parquet"
        df = save_parquet_from_csv(asset_path, parquet_path, logger=logger)
        warn_if_schema_drift(df, stem, logger)
        converted.append((asset_path, parquet_path))

    return converted


def download_assets(
    *,
    session: requests.Session,
    file_patterns: Iterable[str],
    source_name: str,
    logger: logging.Logger,
    force_redownload: bool = False,
    page_url: str = PPEF_DATA_PAGE,
) -> list[tuple[Path, Path]]:
    ensure_directories()
    target_landing_page = _normalize_landing_page(page_url)
    try:
        catalog_payload = get_json(session, CMS_DATA_JSON_URL)
        dataset_payload = resolve_catalog_dataset(
            catalog_payload,
            landing_page=target_landing_page or PPEF_LANDING_PAGE,
        )
    except Exception as exc:
        logger.error("Could not load CMS data catalog from %s: %s", CMS_DATA_JSON_URL, exc)
        return []

    if dataset_payload is None:
        logger.error(
            "Could not resolve the CMS enrollment dataset from %s. "
            "Expected landing_page=%s title=%s",
            CMS_DATA_JSON_URL,
            target_landing_page or PPEF_LANDING_PAGE,
            PPEF_DATASET_TITLE,
        )
        return []

    latest_distribution = select_latest_distribution(dataset_payload)
    if latest_distribution is None:
        logger.error(
            "Resolved CMS dataset '%s' but it did not expose a latest resources API entry.",
            dataset_payload.get("title", PPEF_DATASET_TITLE),
        )
        return []

    resources_api = str(latest_distribution.get("resourcesAPI") or "").strip()
    if not resources_api:
        logger.error(
            "Resolved CMS dataset '%s' but latest distribution had no resources API URL.",
            dataset_payload.get("title", PPEF_DATASET_TITLE),
        )
        return []

    try:
        resource_payload = get_json(session, resources_api)
    except Exception as exc:
        logger.error("Could not load CMS resource manifest from %s: %s", resources_api, exc)
        return []

    available_assets = extract_resource_downloads(resource_payload)
    candidate_assets = filter_resource_downloads(available_assets, file_patterns)
    if not candidate_assets:
        logger.error(
            "No matching CMS enrollment assets were resolved from %s. "
            "Patterns=%s Available resources=%s",
            resources_api,
            list(file_patterns),
            [asset.name for asset in available_assets],
        )
        return []

    candidate_assets = sorted(
        candidate_assets,
        key=lambda asset: (
            {"enrollment": 0, "reassignment": 1, "practice_location": 2}.get(
                classify_download_asset(asset), 99
            ),
            asset.filename,
        ),
    )
    logger.info(
        "Resolved CMS dataset '%s' via catalog. latest_modified=%s temporal=%s assets=%s",
        dataset_payload.get("title", PPEF_DATASET_TITLE),
        latest_distribution.get("modified", ""),
        latest_distribution.get("temporal", ""),
        [asset.filename for asset in candidate_assets],
    )

    today = datetime.now().strftime("%Y%m%d")
    raw_dir = RAW_DIR / source_name / today
    raw_dir.mkdir(parents=True, exist_ok=True)

    converted: list[tuple[Path, Path]] = []
    for asset in candidate_assets:
        url = asset.url
        filename = asset.filename
        destination = raw_dir / filename

        if destination.exists() and not force_redownload:
            logger.info("Asset already exists: %s", destination)
        else:
            logger.info("Downloading asset: %s", url)
            data = http_get(session, url, desc=filename)
            write_bytes(destination, data)
            logger.info("Saved: %s (%.2f MB)", destination, destination.stat().st_size / 1e6)

        converted.extend(
            extract_and_convert_csvs(destination, raw_dir, file_patterns=file_patterns, logger=logger)
        )

    return converted


def find_latest_parquet(
    *,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str] | None = None,
) -> Path | None:
    include = list(include_patterns)
    exclude = list(exclude_patterns or [])
    candidates = sorted(PARQUET_DIR.glob("*.parquet"))
    matches: list[Path] = []
    for path in candidates:
        name = path.name
        if not all(re.search(pattern, name, re.I) for pattern in include):
            continue
        if any(re.search(pattern, name, re.I) for pattern in exclude):
            continue
        matches.append(path)
    return matches[-1] if matches else None


def find_latest_raw_csv(
    *,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str] | None = None,
) -> Path | None:
    include = list(include_patterns)
    exclude = list(exclude_patterns or [])
    candidates = sorted(RAW_DIR.rglob("*.csv"))
    matches: list[Path] = []
    for path in candidates:
        name = path.name
        if not all(re.search(pattern, name, re.I) for pattern in include):
            continue
        if any(re.search(pattern, name, re.I) for pattern in exclude):
            continue
        matches.append(path)
    return matches[-1] if matches else None


def column_as_string(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[column].astype("string").fillna("").str.strip()
