"""
CMS Hospital General Information (HGI) Downloader + Optional Enrichments
-----------------------------------------------------------------------------
Features:
1) Download latest **Hospital General Information (HGI)** CSV from CMS Provider Data Catalog (PDC)
   and convert to Parquet. (No API usage; CSV-only per requirements.)
2) Optional enrichments (CSV-only):
   - **Birthing-Friendly Hospitals with Geocoded Addresses** (geocoded subset)
   - **Hospital Quality** example: "Timely and Effective Care - Hospital" table
   - **Medicare Fee-For-Service Public Provider Enrollment (PPEF)** public files for
     partial affiliation signals (e.g., Reassignments, Practice Locations, Enrollments)
3) Outputs both **CSV and Parquet**; warns on **schema drift** (warn-only).

Notes:
- All downloads scrape dataset pages for the **"Download full dataset (CSV)"** link
  (no API calls). If CMS changes page structure, regex may need updating.
- PPEF links are discovered from the public CMS page that lists data files. This script
  downloads any found ZIP/CSV assets and converts extracted CSVs to Parquet. The PPEF
  data are relational and can be large; only download the parts you need.

Data Sources (reference):
- Hospital General Information dataset page (PDC):
  https://data.cms.gov/provider-data/dataset/xubh-q36u
- Birthing Friendly Hospitals with Geocoded Addresses dataset page (PDC):
  https://data.cms.gov/provider-data/dataset/hbf-map
- Hospitals Topic landing page (to locate quality datasets like Timely & Effective Care):
  https://data.cms.gov/provider-data/topics/hospitals
- Medicare Fee-For-Service Public Provider Enrollment (PPEF) landing/data pages:
  https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-fee-for-service-public-provider-enrollment
  https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/medicare-fee-for-service-public-provider-enrollment/data

Author: Triet Pham
"""

import csv
import io
import json
import logging
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------------- Configuration -----------------------------
HGI_DATASET_PAGE = "https://data.cms.gov/provider-data/dataset/xubh-q36u"
BF_DATASET_PAGE = "https://data.cms.gov/provider-data/dataset/hbf-map"
HOSPITALS_TOPIC_PAGE = "https://data.cms.gov/provider-data/topics/hospitals"
PPEF_DATA_PAGE = (
    "https://data.cms.gov/provider-characteristics/medicare-provider-supplier-enrollment/"
    "medicare-fee-for-service-public-provider-enrollment/data"
)

# Dataset IDs we know and can use for robust resolution
HGI_DATASET_ID = "xubh-q36u"
BF_DATASET_ID = "hbf-map"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Output dirs
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PARQUET_DIR = BASE_DIR / "parquet"
META_DIR = BASE_DIR / "meta"
LOGS_DIR = Path("logs")

for d in [RAW_DIR, PARQUET_DIR, META_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "hgi_downloader.log", mode="a"),
    ],
)
logger = logging.getLogger("hgi_downloader")

# ------------------------------ Utilities --------------------------------
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def _http_get(url: str, desc: str = "Downloading", chunk_size: int = 1024 * 1024) -> bytes:
    """Stream a file from URL and return bytes. Shows a tqdm progress bar."""
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        buf = io.BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    buf.write(chunk)
                    pbar.update(len(chunk))
        return buf.getvalue()


# --- CSV URL Resolvers (fallbacks) -----------------------------------------
def _resolve_csv_from_metastore(dataset_id: str) -> str | None:
    """
    Use the PDC metastore to get the dataset distribution downloadURL (CSV).
    Returns a direct CSV URL or None if not found.
    """
    meta_url = f"https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{dataset_id}"
    try:
        r = session.get(meta_url, timeout=30)
        r.raise_for_status()
        meta = r.json()
        # distributions: list of objects that may include downloadURL for CSV
        for d in meta.get("distribution", []):
            url = d.get("downloadURL") or d.get("accessURL")
            if url and url.lower().endswith(".csv"):
                return url
    except Exception as e:
        logger.debug("Metastore lookup failed for %s: %s", dataset_id, e)
    return None


def _legacy_rows_csv(dataset_id: str) -> str:
    """
    Return the legacy rows.csv URL for a given dataset id (works for many PDC datasets).
    """
    return f"https://data.medicare.gov/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"


def _find_csv_download_link(html: str, dataset_id: str | None = None) -> str:
    """
    Resolve CSV link for a PDC dataset page.

    Tries, in order:
      1) Scrape 'Download full dataset (CSV)' anchor from the HTML (if present)
      2) PDC Metastore 'distribution.downloadURL' for dataset_id
      3) Legacy Medicare rows.csv fallback for dataset_id

    Raises if nothing found.
    """
    # 1) Try to scrape an anchor (works if the link is rendered server-side)
    m = re.search(r'<a[^>]+href="([^"]+)"[^>]*>\s*Download full dataset \(CSV\)', html, re.I)
    if m:
        return m.group(1)

    # 2) Metastore fallback (requires dataset_id)
    if dataset_id:
        csv_url = _resolve_csv_from_metastore(dataset_id)
        if csv_url:
            logger.info("Resolved CSV via metastore: %s", csv_url)
            return csv_url

        # 3) Legacy 'rows.csv' fallback
        legacy = _legacy_rows_csv(dataset_id)
        logger.info("Falling back to legacy rows.csv URL: %s", legacy)
        return legacy

    # If we didn't get dataset_id, try any .csv present as last resort
    m = re.search(r'href="([^"]+\.csv)"', html, re.I)
    if m:
        return m.group(1)

    raise RuntimeError("Could not locate CSV download link on dataset page.")


def _get_html(url: str) -> str:
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def _write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True
    )
    with open(path, "wb") as f:
        f.write(data)


def _save_parquet_from_csv(csv_path: Path, parquet_path: Path):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    return df


def _schema_meta_path(stem: str) -> Path:
    return META_DIR / f"{stem}_schema.json"


def _warn_if_schema_drift(df: pd.DataFrame, stem: str):
    meta_path = _schema_meta_path(stem)
    cols = list(df.columns)
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            prev_cols = prev.get("columns", [])
            if set(prev_cols) != set(cols):
                logger.warning(
                    "SCHEMA DRIFT for %s: columns changed.\nPrev(%d): %s\nCurr(%d): %s",
                    stem, len(prev_cols), prev_cols, len(cols), cols,
                )
        except Exception as e:
            logger.warning("Could not read previous schema for %s: %s", stem, e)
    # Always write current schema (warn-only policy)
    meta = {
        "dataset": stem,
        "columns": cols,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

# ------------------------------ Downloaders -------------------------------

def download_hgi(force_redownload: bool = False):
    """
    Download Hospital General Information CSV and convert to Parquet.
    Outputs: data/raw/hgi_YYYYMMDD.csv, data/parquet/hgi_YYYYMMDD.parquet
    """
    html = _get_html(HGI_DATASET_PAGE)
    csv_url = _find_csv_download_link(html, dataset_id=HGI_DATASET_ID)
    today = datetime.now().strftime("%Y%m%d")
    csv_path = RAW_DIR / f"hgi_{today}.csv"
    parquet_path = PARQUET_DIR / f"hgi_{today}.parquet"

    if csv_path.exists() and not force_redownload:
        logger.info("HGI CSV already exists: %s", csv_path)
    else:
        logger.info("Downloading HGI CSV from: %s", csv_url)
        data = _http_get(csv_url, desc="HGI CSV")
        _write_bytes(csv_path, data)
        logger.info("Saved: %s (%.2f MB)", csv_path, csv_path.stat().st_size / 1e6)

    logger.info("Converting HGI CSV -> Parquet ...")
    df = _save_parquet_from_csv(csv_path, parquet_path)
    logger.info("Saved: %s (%.2f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    _warn_if_schema_drift(df, stem="hgi")
    return csv_path, parquet_path


def download_birthing_friendly(force_redownload: bool = False):
    """
    Download Birthing Friendly Hospitals with Geocoded Addresses CSV.
    Outputs: data/raw/hgi_bf_geo_YYYYMMDD.csv, data/parquet/hgi_bf_geo_YYYYMMDD.parquet
    """
    html = _get_html(BF_DATASET_PAGE)
    csv_url = _find_csv_download_link(html, dataset_id=BF_DATASET_ID)
    today = datetime.now().strftime("%Y%m%d")
    csv_path = RAW_DIR / f"hgi_bf_geo_{today}.csv"
    parquet_path = PARQUET_DIR / f"hgi_bf_geo_{today}.parquet"

    if csv_path.exists() and not force_redownload:
        logger.info("Birthing Friendly CSV already exists: %s", csv_path)
    else:
        logger.info("Downloading Birthing Friendly CSV from: %s", csv_url)
        data = _http_get(csv_url, desc="Birthing Friendly CSV")
        _write_bytes(csv_path, data)
        logger.info("Saved: %s (%.2f MB)", csv_path, csv_path.stat().st_size / 1e6)

    logger.info("Converting Birthing Friendly CSV -> Parquet ...")
    df = _save_parquet_from_csv(csv_path, parquet_path)
    logger.info("Saved: %s (%.2f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    _warn_if_schema_drift(df, stem="hgi_bf_geo")
    return csv_path, parquet_path


def _find_dataset_page_by_title(topic_html: str, title: str) -> str:
    """
    From a topic page HTML, find the dataset page URL by visible title text.
    Returns dataset page URL, or raises if not found.
    """
    pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>\s*' + re.escape(title) + r'\s*<', re.I)
    m = pattern.search(topic_html)
    if m:
        href = m.group(1)
        if href.startswith("/"):
            href = "https://data.cms.gov" + href
        return href

    m = re.search(r'href="([^"]+)"[^>]*>\s*[^<]*' + re.escape(title), topic_html, re.I)
    if m:
        href = m.group(1)
        if href.startswith("/"):
            href = "https://data.cms.gov" + href
        return href

    raise RuntimeError(f"Dataset titled '{title}' not found on topic page.")


def download_quality_te(force_redownload: bool = False, title: str = "Timely and Effective Care - Hospital"):
    """
    Download a hospital quality dataset by title from the Hospitals topic page.
    Defaults to the provider-level Timely & Effective Care table.
    Outputs use stem based on a short code: hgi_te_hospital_YYYYMMDD.*
    """
    topic_html = _get_html(HOSPITALS_TOPIC_PAGE)
    ds_url = _find_dataset_page_by_title(topic_html, title)
    html = _get_html(ds_url)
    # For quality dataset we still rely on page scrape. If this starts failing,
    # we can add a dataset_id param here once you pick the exact dataset ID to pin.
    csv_url = _find_csv_download_link(html, dataset_id=None)

    today = datetime.now().strftime("%Y%m%d")
    stem = "hgi_te_hospital"
    csv_path = RAW_DIR / f"{stem}_{today}.csv"
    parquet_path = PARQUET_DIR / f"{stem}_{today}.parquet"

    if csv_path.exists() and not force_redownload:
        logger.info("Quality (TE Hospital) CSV already exists: %s", csv_path)
    else:
        logger.info("Downloading Quality (TE Hospital) CSV from: %s", csv_url)
        data = _http_get(csv_url, desc="TE Hospital CSV")
        _write_bytes(csv_path, data)
        logger.info("Saved: %s (%.2f MB)", csv_path, csv_path.stat().st_size / 1e6)

    logger.info("Converting TE Hospital CSV -> Parquet ...")
    df = _save_parquet_from_csv(csv_path, parquet_path)
    logger.info("Saved: %s (%.2f MB)", parquet_path, parquet_path.stat().st_size / 1e6)
    _warn_if_schema_drift(df, stem=stem)
    return csv_path, parquet_path


# ------------------------------ PPEF (Public Provider Enrollment Files) ---
PPEF_FILE_PATTERNS = [
    r"ENROLL",     # enrollments
    r"REASSIGN",   # reassignments of benefits (individual -> organization)
    r"PRACTICE",   # practice locations
    r"LOCATION",   # practice location variants
]


def _discover_ppef_asset_links() -> list:
    """Scrape the PPEF data page and return a list of asset URLs (ZIP/CSV)."""
    html = _get_html(PPEF_DATA_PAGE)
    links = re.findall(r'href="(https?://[^"]+\.(?:zip|csv))"', html, flags=re.I)
    seen, deduped = set(), []
    for u in links:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _filter_ppef_links(links: list) -> list:
    filtered = []
    for url in links:
        name = url.split("/")[-1]
        if any(re.search(p, name, re.I) for p in PPEF_FILE_PATTERNS):
            filtered.append(url)
    return filtered


def download_ppef(force_redownload: bool = False):
    """
    Download PPEF assets relevant to partial affiliations and convert.

    - Discovers data file links on the PPEF Data page
    - Filters to files matching patterns for Enrollments, Reassignments, Practice Locations
    - Saves under data/raw/ppef/YYYYMMDD/ and converts any CSVs to Parquet
    - If a ZIP is found, extracts and converts all CSVs inside, applying the same filter

    Returns list of (csv_path, parquet_path) for converted files.
    """
    all_links = _discover_ppef_asset_links()
    cand_links = _filter_ppef_links(all_links)
    if not cand_links:
        logger.warning("No PPEF asset links discovered. The page structure may have changed or the list is empty.")
        return []

    today_dir = RAW_DIR / "ppef" / datetime.now().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for url in cand_links:
        fname = url.split("/")[-1]
        dest = today_dir / fname
        if dest.exists() and not force_redownload:
            logger.info("PPEF asset already exists: %s", dest)
        else:
            logger.info("Downloading PPEF asset: %s", url)
            data = _http_get(url, desc=f"PPEF {fname}")
            _write_bytes(dest, data)
            logger.info("Saved: %s (%.2f MB)", dest, dest.stat().st_size / 1e6)

        if dest.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(dest, "r") as zf:
                    for member in zf.namelist():
                        if not member.lower().endswith(".csv"):
                            continue
                        if not any(re.search(p, member, re.I) for p in PPEF_FILE_PATTERNS):
                            continue
                        out_csv = today_dir / Path(member).name
                        if not out_csv.exists() or force_redownload:
                            logger.info("Extracting %s -> %s", member, out_csv)
                            with zf.open(member) as src, open(out_csv, "wb") as dst:
                                dst.write(src.read())
                        stem = Path(out_csv.stem).stem if out_csv.suffixes and len(out_csv.suffixes) > 1 else out_csv.stem
                        parquet_path = PARQUET_DIR / f"{stem}.parquet"
                        logger.info("Converting %s -> %s", out_csv.name, parquet_path.name)
                        df = _save_parquet_from_csv(out_csv, parquet_path)
                        _warn_if_schema_drift(df, stem)
                        converted.append((out_csv, parquet_path))
            except zipfile.BadZipFile:
                logger.error("Bad ZIP file for PPEF asset: %s", dest)
        elif dest.suffix.lower() == ".csv":
            stem = Path(dest.stem).stem if dest.suffixes and len(dest.suffixes) > 1 else dest.stem
            parquet_path = PARQUET_DIR / f"{stem}.parquet"
            logger.info("Converting %s -> %s", dest.name, parquet_path.name)
            df = _save_parquet_from_csv(dest, parquet_path)
            _warn_if_schema_drift(df, stem)
            converted.append((dest, parquet_path))
        else:
            logger.info("Skipping non-CSV asset: %s", dest)

    if not converted:
        logger.warning("No PPEF CSVs converted. Check filters or CMS page.")
    return converted

# ------------------------------ CLI --------------------------------------

def main():
    print("\n" + "=" * 80)
    print("CMS HGI DOWNLOADER + OPTIONAL ENRICHMENTS (CSV only)")
    print("=" * 80)
    print("\nThis tool will:")
    print(" - Download Hospital General Information (HGI) and convert to Parquet")
    print(" - Optionally download:")
    print("   * Birthing-Friendly (geocoded)")
    print("   * Timely & Effective Care - Hospital (quality example)")
    print("   * PPEF public provider enrollment files (partial affiliations)")
    print(" - Warn on schema drift; outputs CSV + Parquet")
    print("\nNOTE: No API usage. All downloads parse the dataset pages for CSV links.")
    print("=" * 80 + "\n")

    choice = input(
        "Choose:\n"
        "1. HGI only: Download + Convert (default)\n"
        "2. HGI + select enrichments (BF / Quality / PPEF)\n"
        "3. Download CSV(s) only\n"
        "4. Convert existing CSV(s) to Parquet\n\n"
        "Enter 1, 2, 3, or 4: "
    ).strip() or "1"

    force = input("Force re-download if files exist? (y/N): ").strip().lower() == "y"

    if choice == "1":
        download_hgi(force_redownload=force)

    elif choice == "2":
        download_hgi(force_redownload=force)
        if input("Add Birthing-Friendly (geocoded)? (y/N): ").strip().lower() == "y":
            download_birthing_friendly(force_redownload=force)
        if input("Add Quality (Timely & Effective Care - Hospital)? (y/N): ").strip().lower() == "y":
            download_quality_te(force_redownload=force)
        if input("Add PPEF public enrollment files (partial affiliations)? (y/N): ").strip().lower() == "y":
            download_ppef(force_redownload=force)

    elif choice == "3":
        # CSVs only
        download_hgi(force_redownload=force)
        if input("Also download Birthing-Friendly (geocoded)? (y/N): ").strip().lower() == "y":
            download_birthing_friendly(force_redownload=force)
        if input("Also download Quality (TE - Hospital)? (y/N): ").strip().lower() == "y":
            download_quality_te(force_redownload=force)
        if input("Also download PPEF files? (y/N): ").strip().lower() == "y":
            download_ppef(force_redownload=force)
        print("\nCSV(s) downloaded. Parquet conversion skipped by user choice.")

    elif choice == "4":
        # Convert any existing CSVs matching our patterns in data/raw/
        today = datetime.now().strftime("%Y%m%d")
        patterns = [
            f"hgi_{today}.csv",
            f"hgi_bf_geo_{today}.csv",
            f"hgi_te_hospital_{today}.csv",
        ]
        # Convert HGI & enrichment CSVs if present
        for pat in patterns:
            for csv_path in RAW_DIR.glob(pat):
                stem = csv_path.stem
                parquet_path = PARQUET_DIR / f"{stem}.parquet"
                logger.info("Converting %s -> %s", csv_path.name, parquet_path.name)
                df = _save_parquet_from_csv(csv_path, parquet_path)
                _warn_if_schema_drift(df, stem)
        # Convert any PPEF CSVs under data/raw/ppef/DATE/
        for csv_path in (RAW_DIR / "ppef").rglob("*.csv"):
            stem = csv_path.stem
            parquet_path = PARQUET_DIR / f"{stem}.parquet"
            logger.info("Converting %s -> %s", csv_path.name, parquet_path.name)
            df = _save_parquet_from_csv(csv_path, parquet_path)
            _warn_if_schema_drift(df, stem)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()