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
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm

PPEF_DATA_PAGE = (
    "https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/"
    "medicare-fee-for-service-public-provider-enrollment/data"
)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PARQUET_DIR = BASE_DIR / "parquet"
META_DIR = BASE_DIR / "meta"
LOGS_DIR = Path("logs")


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


def get_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


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


def save_parquet_from_csv(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    return df


def discover_asset_links(
    session: requests.Session, page_url: str = PPEF_DATA_PAGE
) -> list[str]:
    html = get_html(session, page_url)
    links = re.findall(r'href="(https?://[^"]+\.(?:zip|csv))"', html, flags=re.I)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in links:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def filter_asset_links(links: Iterable[str], file_patterns: Iterable[str]) -> list[str]:
    patterns = list(file_patterns)
    filtered = []
    for url in links:
        name = url.split("/")[-1]
        if any(re.search(pattern, name, re.I) for pattern in patterns):
            filtered.append(url)
    return filtered


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
                    df = save_parquet_from_csv(csv_path, parquet_path)
                    warn_if_schema_drift(df, stem, logger)
                    converted.append((csv_path, parquet_path))
        except zipfile.BadZipFile:
            logger.error("Bad ZIP file for CMS enrollment asset: %s", asset_path)
    elif asset_path.suffix.lower() == ".csv" and _accept(asset_path.name):
        stem = asset_path.stem
        parquet_path = PARQUET_DIR / f"{stem}.parquet"
        df = save_parquet_from_csv(asset_path, parquet_path)
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
    all_links = discover_asset_links(session, page_url=page_url)
    candidate_links = filter_asset_links(all_links, file_patterns)
    if not candidate_links:
        logger.warning("No matching CMS enrollment files discovered on %s.", page_url)
        return []

    today = datetime.now().strftime("%Y%m%d")
    raw_dir = RAW_DIR / source_name / today
    raw_dir.mkdir(parents=True, exist_ok=True)

    converted: list[tuple[Path, Path]] = []
    for url in candidate_links:
        filename = url.split("/")[-1]
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


def column_as_string(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[column].astype("string").fillna("").str.strip()
