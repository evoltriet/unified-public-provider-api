"""

NPI Registry - Unified Processing with Metadata Tracking

===========================================================================

Features:

1. GEOCODING: Full geocoding with multiple providers

2. GEOHASHING: Fast spatial indexing without API calls

3. DATA QUALITY CHECK: Identify problematic records before geocoding

4. ENTITY MAPPING: Map individuals to organizations

Architecture:

- Single 'processed_data' folder

- One parquet per week with ALL processing columns combined

- Metadata.json for audit trail and processing status tracking

- Smart resumption: Continue from where you left off

"""

import os
import requests

import pandas as pd

import zipfile

import time

import hashlib

from pathlib import Path

from datetime import datetime, timedelta

import logging

from tqdm import tqdm

import re

import warnings

import json
import random
from urllib.parse import urlparse, urlunparse

from typing import Optional, Dict, List, Tuple, Set

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

BASE_URL = "https://npiregistry.cms.hhs.gov/api/"

API_VERSION = "2.1"

DOWNLOAD_PAGE = "https://download.cms.gov/nppes/NPI_Files.html"


class GeocodingService:
    """Handles geocoding operations with rate limiting and caching."""

    def __init__(self, provider="nominatim", api_key=None, rate_limit=1.0):

        self.provider = provider

        self.api_key = api_key

        self.rate_limit = rate_limit

        self.last_request_time = 0

        self.cache = {}

        self.config = {
            "nominatim": {
                "url": "https://nominatim.openstreetmap.org/search",
                "rate_limit": 1.0,
                "requires_key": False,
            },
            "google": {
                "url": "https://maps.googleapis.com/maps/api/geocode/json",
                "rate_limit": 50.0,
                "requires_key": True,
            },
            "geoapify": {
                "url": "https://api.geoapify.com/v1/geocode/search",
                "rate_limit": 5.0,
                "requires_key": True,
            },
            "azure": {
                "url": "https://atlas.microsoft.com/search/address/json",
                "rate_limit": 50.0,
                "requires_key": True,
            },
        }

        if self.config[provider]["requires_key"] and not api_key:

            logger.warning(
                f"{provider} requires an API key. Falling back to Nominatim."
            )

            self.provider = "nominatim"

            self.rate_limit = 1.0

    def _rate_limit_wait(self):

        elapsed = time.time() - self.last_request_time

        wait_time = (1.0 / self.rate_limit) - elapsed

        if wait_time > 0:

            time.sleep(wait_time)

        self.last_request_time = time.time()

    def geocode_address(
        self, address: str, city: str, state: str, postal_code: str, country: str = "US"
    ) -> Optional[Dict]:

        address_parts = [
            p for p in [address, city, state, postal_code, country] if pd.notna(p) and p
        ]

        full_address = ", ".join(address_parts)

        if full_address in self.cache:

            return self.cache[full_address]

        self._rate_limit_wait()

        try:

            if self.provider == "nominatim":

                result = self._geocode_nominatim(full_address)

            elif self.provider == "google":

                result = self._geocode_google(full_address)

            elif self.provider == "geoapify":

                result = self._geocode_geoapify(full_address, country)

            elif self.provider == "azure":

                result = self._geocode_azure(full_address, country)

            else:

                result = None

            self.cache[full_address] = result

            return result

        except Exception as e:

            logger.warning(f"Geocoding failed for '{full_address}': {e}")

            return None

    def _geocode_nominatim(self, address: str) -> Optional[Dict]:

        params = {"q": address, "format": "json", "limit": 1}

        headers = {"User-Agent": "NPI-Registry-Geocoder/1.0"}

        response = requests.get(
            self.config["nominatim"]["url"], params=params, headers=headers, timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            if data:

                return {
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"]),
                    "formatted_address": data[0].get("display_name", ""),
                    "confidence": float(data[0].get("importance", 0.5)),
                }

        return None

    def _geocode_google(self, address: str) -> Optional[Dict]:

        params = {"address": address, "key": self.api_key}

        response = requests.get(self.config["google"]["url"], params=params, timeout=10)

        if response.status_code == 200:

            data = response.json()

            if data["status"] == "OK" and data["results"]:

                location = data["results"][0]["geometry"]["location"]

                return {
                    "latitude": location["lat"],
                    "longitude": location["lng"],
                    "formatted_address": data["results"][0]["formatted_address"],
                    "confidence": 0.9,
                }

        return None

    def _geocode_geoapify(self, address: str, country: str) -> Optional[Dict]:

        params = {"text": address, "format": "json", "limit": 1, "apiKey": self.api_key}

        if country:

            params["filter"] = f"countrycode:{country.lower()}"

        response = requests.get(
            self.config["geoapify"]["url"], params=params, timeout=10
        )

        if response.status_code == 200:

            data = response.json()

            if data.get("results"):

                result = data["results"][0]

                return {
                    "latitude": result["lat"],
                    "longitude": result["lon"],
                    "formatted_address": result.get("formatted", ""),
                    "confidence": result.get("rank", {}).get("confidence", 0.5),
                }

        return None

    def _geocode_azure(self, address: str, country: str) -> Optional[Dict]:

        params = {
            "query": address,
            "api-version": "1.0",
            "subscription-key": self.api_key,
            "countrySet": country,
        }

        response = requests.get(self.config["azure"]["url"], params=params, timeout=10)

        if response.status_code == 200:

            data = response.json()

            if data.get("results"):

                result = data["results"][0]

                position = result["position"]

                return {
                    "latitude": position["lat"],
                    "longitude": position["lon"],
                    "formatted_address": result.get("address", {}).get(
                        "freeformAddress", ""
                    ),
                    "confidence": result.get("score", 0.5),
                }

        return None


class NPIRegistryProcessed:
    """Unified NPI Registry processor with metadata tracking."""

    def __init__(
        self,
        data_dir="../data",
        geocoding_provider="nominatim",
        api_key=None,
        output_prefix="npi_processed",
    ):

        self.data_dir = Path(data_dir)

        self.data_dir.mkdir(exist_ok=True)

        self.raw_dir = self.data_dir / "raw"

        self.raw_dir.mkdir(exist_ok=True)

        self.parquet_dir = self.data_dir / "parquet"

        self.parquet_dir.mkdir(exist_ok=True)

        self.processed_dir = self.data_dir / "processed_data"

        self.processed_dir.mkdir(exist_ok=True)

        self.hash_dir = self.data_dir / "hashes"

        self.hash_dir.mkdir(exist_ok=True)

        self.output_prefix = output_prefix

        self.geocoder = GeocodingService(provider=geocoding_provider, api_key=api_key)

    def get_data_date(self, file_path: Path) -> Optional[str]:
        """Extract date from filename (format: *_YYYYMMDD.parquet or *_YYYYMMDD.csv)."""

        match = re.search(r"(\d{8})", str(file_path))

        return match.group(1) if match else None

    def find_data_file(self) -> Optional[Path]:
        """Find latest NPI INDIVIDUALS data file (skip organizations)."""

        # Priority 1: Look for npi_individuals_*.parquet files

        individuals_files = list(self.parquet_dir.glob("npi_individuals_*.parquet"))

        if individuals_files:

            latest_individuals = sorted(individuals_files)[-1]

            logger.info(f"Found individuals parquet file: {latest_individuals}")

            return latest_individuals

        # Priority 2: Look for any parquet file EXCEPT organizations

        parquet_files = [
            f
            for f in self.parquet_dir.glob("*.parquet")
            if "organizations" not in f.name
        ]

        if parquet_files:

            latest_parquet = sorted(parquet_files)[-1]

            logger.info(f"Found parquet file: {latest_parquet}")

            return latest_parquet

        # Priority 3: Look for CSV files EXCEPT organizations

        csv_files = [
            f for f in self.raw_dir.glob("*.csv") if "organizations" not in f.name
        ]

        if csv_files:

            latest_csv = sorted(csv_files)[-1]

            logger.info(f"Found CSV file: {latest_csv}")

            return latest_csv

        return None


    def find_split_parquet(self, split: str) -> Optional[Path]:
        """Find latest split parquet file for a given split ('individuals' or 'organizations')."""
        if split not in {"individuals", "organizations"}:
            raise ValueError("split must be 'individuals' or 'organizations'")
        files = list(self.parquet_dir.glob(f"npi_{split}_*.parquet"))
        if not files:
            return None
        latest = sorted(files)[-1]
        logger.info(f"Found {split} parquet file: {latest}")
        return latest


    def find_latest_processed_parquet(self, prefix: str) -> Optional[Path]:
        """Find latest processed parquet in processed_data/ with a given prefix."""
        files = list(self.processed_dir.glob(f"{prefix}_*.parquet"))
        if not files:
            return None
        return sorted(files)[-1]


    def load_data(self) -> Optional[pd.DataFrame]:
        """Load NPI data from available sources."""

        data_file = self.find_data_file()

        if data_file is None:

            logger.error("No NPI data file found")
            logger.error(data_file)

            return None

        logger.info(f"Loading data from: {data_file}")

        try:

            start_time = time.time()

            if str(data_file).endswith(".parquet"):

                df = pd.read_parquet(data_file)

            else:

                df = pd.read_csv(data_file)

            elapsed_time = time.time() - start_time

            logger.info(f"Loaded {len(df):,} records in {elapsed_time:.1f} seconds")

            return df

        except Exception as e:

            logger.error(f"Failed to load data: {e}")

            return None

    # ==================== METADATA MANAGEMENT ====================

    def get_metadata_path(self, data_date: str) -> Path:
        """Get metadata file path for given date."""

        return self.processed_dir / f"{self.output_prefix}_{data_date}_metadata.json"

    def get_processed_data_path(self, data_date: str) -> Path:
        """Get processed data parquet path for given date."""

        return self.processed_dir / f"{self.output_prefix}_{data_date}.parquet"

    def load_metadata(self, data_date: str) -> Dict:
        """Load existing metadata or create new."""

        metadata_path = self.get_metadata_path(data_date)

        if metadata_path.exists():

            logger.info(f"Loading existing metadata: {metadata_path}")

            try:

                with open(metadata_path, "r") as f:

                    return json.load(f)

            except Exception as e:

                logger.warning(f"Failed to load metadata: {e}. Creating new metadata.")

        # Create new metadata

        return {
            "date": data_date,
            "created_at": datetime.now().isoformat(),
            "total_records": 0,
            "processing_modes_completed": [],
        }

    def save_metadata(self, metadata: Dict, data_date: str):
        """Save metadata to JSON file."""

        metadata_path = self.get_metadata_path(data_date)

        try:

            with open(metadata_path, "w") as f:

                json.dump(metadata, f, indent=2)

            logger.info(f"Saved metadata: {metadata_path}")

        except Exception as e:

            logger.error(f"Failed to save metadata: {e}")

    def get_completed_modes(self, metadata: Dict) -> List[str]:
        """Get list of completed processing modes from metadata."""

        return [m["mode"] for m in metadata.get("processing_modes_completed", [])]

    def is_mode_completed(self, mode: str, metadata: Dict) -> bool:
        """Check if a processing mode is already completed."""

        return mode in self.get_completed_modes(metadata)

    def add_mode_completion(
        self,
        metadata: Dict,
        mode: str,
        duration_seconds: float,
        changed_records: int,
        columns_added: List[str],
    ) -> Dict:
        """Add mode completion info to metadata."""

        metadata["processing_modes_completed"].append(
            {
                "mode": mode,
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": round(duration_seconds, 1),
                "changed_records": changed_records,
                "columns_added": columns_added,
            }
        )

        return metadata

    # ==================== HELPER METHODS - ENTITY MAPPING ====================

    def _safe_str(self, value) -> Optional[str]:
        """Safely convert value to string, handling NaN and None."""

        if pd.isna(value) or value is None:

            return None

        s = str(value).strip()

        return s if s and s.lower() != "nan" else None

    def _safe_phone(self, value) -> Optional[str]:
        """Safely convert phone to string, handling floats and NaN."""

        if pd.isna(value) or value is None:

            return None

        if isinstance(value, (int, float)):

            phone_str = str(int(value)).strip()

        else:

            phone_str = str(value).strip()

        return (
            phone_str
            if phone_str and phone_str != "nan" and len(phone_str) > 0
            else None
        )

    

    # ==================== HELPER METHODS - WEBSITE MAPPING ====================
    def _normalize_url(self, url: Optional[str]) -> Optional[str]:
        """Normalize URL by forcing https, removing query/fragment, and lowercasing host."""
        url = self._safe_str(url)
        if not url:
            return None
        if not re.match(r"^https?://", url, flags=re.I):
            url = "https://" + url
        try:
            p = urlparse(url)
            scheme = "https"
            netloc = (p.netloc or "").lower()
            path = p.path or "/"
            return urlunparse((scheme, netloc, path, "", "", ""))
        except Exception:
            return url

    def _root_url(self, url: Optional[str]) -> Optional[str]:
        """Return the scheme+host root homepage (https://host/)."""
        url = self._normalize_url(url)
        if not url:
            return None
        try:
            p = urlparse(url)
            if not p.netloc:
                return None
            return urlunparse(("https", p.netloc.lower(), "/", "", "", ""))
        except Exception:
            return None

    def _location_url(self, url: Optional[str]) -> Optional[str]:
        """Return the URL if it appears to be a non-root location/campus page (path != '/')."""
        url = self._normalize_url(url)
        if not url:
            return None
        try:
            p = urlparse(url)
            if p.path and p.path != "/":
                return url
        except Exception:
            pass
        return None

    def _get_taxonomy_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if "Healthcare Provider Taxonomy Code" in c]

    def _is_hospital_row(self, row: pd.Series, taxonomy_cols: List[str]) -> bool:
        """Hospital heuristic: taxonomy codes or org name keywords."""
        hospital_taxonomy: Set[str] = {"282N00000X", "283Q00000X", "284300000X", "283X00000X"}
        for c in taxonomy_cols:
            v = self._safe_str(row.get(c))
            if v and v in hospital_taxonomy:
                return True
        name = self._safe_str(row.get("Provider Organization Name (Legal Business Name)"))
        if name:
            n = name.lower()
            if any(k in n for k in ["hospital", "medical center", "med ctr", "regional medical", "memorial"]):
                return True
        return False

    # -------------------- OSM / NOMINATIM WEBSITE LOOKUP (OPTIMIZED) --------------------

    def _osm_cache_path(self, data_date: str) -> Path:
        """Path for persistent OSM website cache for a given run date (stored under data/hashes/)."""
        return self.hash_dir / f"osm_website_cache_{data_date}.json"

    def _load_osm_cache(self, data_date: str) -> Dict[str, Optional[Dict]]:
        """Load persistent OSM cache (key -> result dict or None)."""
        cache_path = self._osm_cache_path(data_date)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load OSM cache {cache_path}: {e}. Starting empty.")
        return {}

    def _save_osm_cache(self, cache: Dict[str, Optional[Dict]], data_date: str):
        """Save OSM cache to disk."""
        cache_path = self._osm_cache_path(data_date)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save OSM cache {cache_path}: {e}")

    def _nominatim_request_with_retry(
        self,
        params: Dict,
        headers: Dict,
        max_retries: int = 4,
        base_sleep_s: float = 0.6,
        jitter_s: float = 0.3,
        timeout_s: int = 20,
    ) -> Optional[List[Dict]]:
        """Nominatim request with retry + jitter. Returns JSON list or None."""
        url = "https://nominatim.openstreetmap.org/search"
        retriable = {429, 500, 502, 503, 504}

        for attempt in range(max_retries + 1):
            # Gentle pacing to respect shared service + avoid burst patterns
            time.sleep(base_sleep_s + random.random() * jitter_s)
            try:
                r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
                if r.status_code == 200:
                    data = r.json() or []
                    return data if isinstance(data, list) else None

                if r.status_code in retriable:
                    backoff = (2 ** attempt) * (base_sleep_s + random.random() * jitter_s)
                    logger.info(
                        f"Nominatim status {r.status_code} (attempt {attempt+1}/{max_retries+1}); backoff {backoff:.2f}s"
                    )
                    time.sleep(backoff)
                    continue

                # non-retriable
                return None

            except Exception as e:
                backoff = (2 ** attempt) * (base_sleep_s + random.random() * jitter_s)
                logger.info(
                    f"Nominatim error {e} (attempt {attempt+1}/{max_retries+1}); backoff {backoff:.2f}s"
                )
                time.sleep(backoff)

        return None

    def _nominatim_lookup_website(
        self,
        name: str,
        address: str,
        city: str,
        state: str,
        postal_code: str,
        country_code: str = "us",
        min_importance: float = 0.25,
    ) -> Optional[Dict]:
        """    OSM/Nominatim lookup using STRUCTURED parameters when possible.

        - Uses street/city/state/postalcode/countrycodes for better precision.
        - Accepts multiple website-related tag keys.
        - Applies an importance threshold to reduce false matches.
        """
        headers = {"User-Agent": "NPI-Registry-WebsiteMapper/1.0"}
        postal5 = (postal_code or "")[:5]

        params = {
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
            "extratags": 1,
            "namedetails": 1,
            "countrycodes": (country_code or "us").lower(),
        }

        # Prefer structured params when we have enough address signal
        if address and city and state:
            params.update({"street": address, "city": city, "state": state})
            if postal5:
                params["postalcode"] = postal5
        else:
            # Fallback to text query
            q_parts = [name, address, city, state, postal5, (country_code or "us").upper()]
            params["q"] = ", ".join([p for p in q_parts if p])

        data = self._nominatim_request_with_retry(params=params, headers=headers)
        if not data:
            return None

        item = data[0]
        importance = float(item.get("importance", 0.0) or 0.0)
        if importance and importance < min_importance:
            return None

        tags = item.get("extratags") or {}
        website = (
            tags.get("website")
            or tags.get("contact:website")
            or tags.get("url")
            or tags.get("operator:website")
            or tags.get("brand:website")
        )
        if not website:
            return None

        matched_name = (item.get("namedetails") or {}).get("name") or item.get("display_name", "")

        return {
            "url": self._normalize_url(website),
            "confidence": importance if importance else 0.6,
            "matched_name": matched_name,
            "raw": {
                "importance": importance,
                "place_rank": item.get("place_rank"),
                "type": item.get("type"),
                "class": item.get("class"),
            },
        }
# ==================== HASHING ====================

    def compute_address_hash(self, row: pd.Series) -> str:
        """Compute SHA-256 hash of address fields to detect changes."""

        address_fields = [
            str(row.get("Provider First Line Business Practice Location Address", "")),
            str(row.get("Provider Second Line Business Practice Location Address", "")),
            str(row.get("Provider Business Practice Location Address City Name", "")),
            str(row.get("Provider Business Practice Location Address State Name", "")),
            str(row.get("Provider Business Practice Location Address Postal Code", "")),
            str(
                row.get(
                    "Provider Business Practice Location Address Country Code (If outside U.S.)",
                    "",
                )
            ),
        ]

        address_string = "|".join(address_fields)

        return hashlib.sha256(address_string.encode()).hexdigest()

    def load_current_week_hashes(self, data_date: str) -> Optional[Dict[str, str]]:
        """Load hashes for current week if they already exist."""

        hash_file = self.hash_dir / f"address_hashes_{data_date}.parquet"

        if not hash_file.exists():

            return None

        logger.info(f"Found existing hashes for current data: {hash_file}")

        try:

            start_time = time.time()

            df_hashes = pd.read_parquet(hash_file)

            hashes_dict = {}

            for idx, row in df_hashes.iterrows():

                npi = str(int(row["NPI"])) if pd.notna(row["NPI"]) else None

                hash_val = row["address_hash"]

                if npi and hash_val:

                    hashes_dict[npi] = hash_val

            elapsed_time = time.time() - start_time

            logger.info(
                f"Loaded {len(hashes_dict):,} current week hashes in {elapsed_time:.1f} seconds"
            )

            return hashes_dict

        except Exception as e:

            logger.warning(f"Failed to load current week hashes: {e}")

            return None

    def load_previous_week_hashes(self, current_data_date: str) -> Dict[str, str]:
        """Load previous week's address hashes (excluding current week)."""

        hash_files = sorted(self.hash_dir.glob("address_hashes_*.parquet"))

        if not hash_files:

            logger.info(f"No hash files found in: {self.hash_dir}")

            return {}

        hash_files = [
            f for f in hash_files if self.get_data_date(f) != current_data_date
        ]

        if not hash_files:

            logger.info(f"No previous week hash files found (only current week exists)")

            return {}

        latest_hash_file = hash_files[-1]

        logger.info(
            f"Found {len(hash_files)} previous hash files. Loading latest: {latest_hash_file}"
        )

        try:

            start_time = time.time()

            df_hashes = pd.read_parquet(latest_hash_file)

            hashes_dict = {}

            for idx, row in df_hashes.iterrows():

                npi = str(int(row["NPI"])) if pd.notna(row["NPI"]) else None

                hash_val = row["address_hash"]

                if npi and hash_val:

                    hashes_dict[npi] = hash_val

            elapsed_time = time.time() - start_time

            logger.info(
                f"Loaded {len(hashes_dict):,} previous week hashes in {elapsed_time:.1f} seconds"
            )

            return hashes_dict

        except Exception as e:

            logger.warning(f"Failed to load hashes: {e}")

            return {}

    def save_current_hashes(self, df: pd.DataFrame, data_date: str):
        """Save current address hashes for next week's comparison."""

        hash_file = self.hash_dir / f"address_hashes_{data_date}.parquet"

        if hash_file.exists():

            logger.info(f"Hash file already exists: {hash_file} (skipping save)")

            return

        start_time = time.time()

        hash_df = df[["NPI", "address_hash"]].copy()

        hash_df["NPI"] = hash_df["NPI"].astype("int64")

        hash_df.to_parquet(hash_file, compression="snappy", index=False)

        elapsed_time = time.time() - start_time

        logger.info(
            f"Saved {len(hash_df):,} address hashes to: {hash_file} ({elapsed_time:.1f} seconds)"
        )

    def identify_changed_addresses(
        self, df: pd.DataFrame, data_date: str
    ) -> pd.DataFrame:
        """Identify records with changed addresses since last run."""

        current_hashes_dict = self.load_current_week_hashes(data_date)

        if current_hashes_dict is not None:

            logger.info(
                "Using existing hashes for current data (no recomputation needed)"
            )

            df["address_hash"] = (
                df["NPI"]
                .astype(str)
                .str.replace(".0", "")
                .apply(lambda npi: current_hashes_dict.get(str(int(float(npi))), ""))
            )

        else:

            logger.info("Computing address hashes for current data...")

            start_time = time.time()

            df["address_hash"] = df.apply(self.compute_address_hash, axis=1)

            elapsed_time = time.time() - start_time

            logger.info(
                f"Computed {len(df):,} address hashes in {elapsed_time:.1f} seconds"
            )

        previous_hashes = self.load_previous_week_hashes(data_date)

        if not previous_hashes:

            logger.info("No previous hashes found. Will process all records.")

            df["needs_processing"] = True

        else:

            logger.info(
                f"Comparing current data ({len(df):,} records) with {len(previous_hashes):,} previous hashes..."
            )

            start_time = time.time()

            df["npi_str"] = (
                df["NPI"]
                .astype(str)
                .str.replace(".0", "")
                .apply(lambda x: str(int(float(x))))
            )

            def address_changed(row):

                npi_key = row["npi_str"]

                current_hash = row["address_hash"]

                previous_hash = previous_hashes.get(npi_key, None)

                return previous_hash is None or current_hash != previous_hash

            df["needs_processing"] = df.apply(address_changed, axis=1)

            df = df.drop("npi_str", axis=1)

            elapsed_time = time.time() - start_time

            logger.info(f"Comparison completed in {elapsed_time:.1f} seconds")

        changed_count = df["needs_processing"].sum()

        total_count = len(df)

        unchanged_count = total_count - changed_count

        logger.info(f"\nChange detection summary:")

        logger.info(f" - Total records: {total_count:,}")

        logger.info(
            f" - Changed/New: {changed_count:,} ({changed_count/total_count*100:.1f}%)"
        )

        logger.info(
            f" - Unchanged: {unchanged_count:,} ({unchanged_count/total_count*100:.1f}%)"
        )

        return df

    def load_or_create_processed_data(
        self, df: pd.DataFrame, data_date: str
    ) -> Tuple[pd.DataFrame, Dict]:
        """

        Load existing processed data or create new one.

        Returns (dataframe, metadata)

        """

        processed_path = self.get_processed_data_path(data_date)

        metadata = self.load_metadata(data_date)

        if processed_path.exists():

            logger.info(f"\n" + "=" * 80)

            logger.info(f"LOADING EXISTING PROCESSED DATA FOR {data_date}")

            logger.info(f"Path: {processed_path}")

            logger.info("=" * 80)

            try:

                df_processed = pd.read_parquet(processed_path)

                logger.info(
                    f"Loaded {len(df_processed):,} records with {len(df_processed.columns)} columns"
                )

                # Show completed modes

                completed_modes = self.get_completed_modes(metadata)

                logger.info(
                    f"\nProcessing modes already completed: {', '.join(completed_modes) if completed_modes else 'None'}"
                )

                return df_processed, metadata

            except Exception as e:

                logger.warning(f"Failed to load existing processed data: {e}")

                logger.info("Creating new processed data...")

        else:

            logger.info(f"\n" + "=" * 80)

            logger.info(f"CREATING NEW PROCESSED DATA FOR {data_date}")

            logger.info(f"Path: {processed_path}")

            logger.info("=" * 80)

            # Update metadata with total records

            metadata["total_records"] = len(df)

        return df, metadata

    def save_processed_data(self, df: pd.DataFrame, metadata: Dict, data_date: str):
        """Save processed data and metadata."""

        processed_path = self.get_processed_data_path(data_date)

        start_time = time.time()

                # Drop legacy/back-compat columns if they still exist
        for _c in ["front_page_url", "front_page_url_source", "front_page_url_confidence", "front_page_url_matched_name"]:
            if _c in df.columns:
                df = df.drop(columns=[_c])
        df.to_parquet(processed_path, compression="snappy", index=False)

        elapsed_time = time.time() - start_time

        logger.info(f"Saved processed data: {processed_path}")

        logger.info(f" - Rows: {len(df):,}")

        logger.info(f" - Columns: {len(df.columns)}")

        logger.info(f" - Size: {processed_path.stat().st_size / 1024 / 1024:.1f} MB")

        logger.info(f" - Time: {elapsed_time:.1f} seconds")

        self.save_metadata(metadata, data_date)

    # ==================== GEOHASHING ====================

    def geohash_address(
        self, address: str, city: str, state: str, postal_code: str, precision: int = 5
    ) -> str:
        """Create a geohash from address components."""

        location_key = f"{postal_code}|{state}|{city}".lower()

        hash_obj = hashlib.md5(location_key.encode())

        hash_int = int(hash_obj.hexdigest(), 16)

        base32 = "0123456789bcdefghjkmnpqrstuvwxyz"

        geohash = ""

        for _ in range(precision):

            geohash += base32[hash_int % 32]

            hash_int //= 32

        return geohash[:precision]

    def apply_geohashing(
        self, df: pd.DataFrame, metadata: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Apply geohashing to changed records only."""

        logger.info("\n" + "=" * 80)

        logger.info("APPLYING GEOHASHING")

        logger.info("=" * 80)

        start_time = time.time()

        # Check if already completed

        if self.is_mode_completed("geohash", metadata):

            logger.info("Geohashing already completed for this data. Skipping.")

            logger.info("=" * 80)

            return df, metadata

        # Only process changed records

        df_to_process = df[df["needs_processing"]].copy()

        total_to_process = len(df_to_process)

        if total_to_process == 0:

            logger.info("No records need geohashing (all unchanged)")

            logger.info("=" * 80)

            return df, metadata

        logger.info(f"Computing geohashes for {total_to_process:,} changed records...")

        df_to_process["geohash"] = df_to_process.apply(
            lambda row: self.geohash_address(
                address=row.get(
                    "Provider First Line Business Practice Location Address", ""
                ),
                city=row.get(
                    "Provider Business Practice Location Address City Name", ""
                ),
                state=row.get(
                    "Provider Business Practice Location Address State Name", ""
                ),
                postal_code=row.get(
                    "Provider Business Practice Location Address Postal Code", ""
                ),
            ),
            axis=1,
        )

        # Merge back to full dataframe

        df.loc[df_to_process.index, "geohash"] = df_to_process["geohash"]

        elapsed_time = time.time() - start_time

        # Statistics

        unique_geohashes = df_to_process["geohash"].nunique()

        logger.info(f"\nGeohashing complete:")

        logger.info(f" - Records processed: {total_to_process:,} (changed/new only)")

        logger.info(f" - Unique geohashes: {unique_geohashes:,}")

        logger.info(f" - Time: {elapsed_time:.1f} seconds")

        logger.info(f" - Speed: {total_to_process/elapsed_time:,.0f} records/second")

        logger.info("=" * 80)

        # Update metadata

        metadata = self.add_mode_completion(
            metadata, "geohash", elapsed_time, total_to_process, ["geohash"]
        )

        return df, metadata

    # ==================== DATA QUALITY CHECK ====================

    def data_quality_check(
        self, df: pd.DataFrame, metadata: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Analyze data quality for changed records only."""

        logger.info("\n" + "=" * 80)

        logger.info("DATA QUALITY CHECK")

        logger.info("=" * 80)

        start_time = time.time()

        # Check if already completed

        if self.is_mode_completed("quality_check", metadata):

            logger.info("Quality check already completed for this data. Skipping.")

            logger.info("=" * 80)

            return df, metadata

        # Only process changed records

        df_to_process = df[df["needs_processing"]].copy()

        total_to_process = len(df_to_process)

        if total_to_process == 0:

            logger.info("No records need quality check (all unchanged)")

            logger.info("=" * 80)

            return df, metadata

        logger.info(
            f"Running quality checks on {total_to_process:,} changed records..."
        )

        quality_scores = []

        for idx, row in df_to_process.iterrows():

            issues = []

            score = 100

            address = str(
                row.get("Provider First Line Business Practice Location Address", "")
            ).strip()

            if not address or len(address) < 5:

                issues.append("Missing/Invalid address")

                score -= 30

            city = str(
                row.get("Provider Business Practice Location Address City Name", "")
            ).strip()

            if not city or len(city) < 2:

                issues.append("Missing city")

                score -= 25

            state = str(
                row.get("Provider Business Practice Location Address State Name", "")
            ).strip()

            if not state or len(state) != 2:

                issues.append("Invalid state")

                score -= 25

            postal = str(
                row.get("Provider Business Practice Location Address Postal Code", "")
            ).strip()

            if not postal or len(postal) < 5:

                issues.append("Invalid postal code")

                score -= 20

            country = str(
                row.get(
                    "Provider Business Practice Location Address Country Code (If outside U.S.)",
                    "US",
                )
            ).strip()

            if country not in ["US", "USA", ""]:

                issues.append("Non-US address")

                score -= 20

            quality_scores.append(
                {
                    "NPI": row.get("NPI"),
                    "quality_score": max(0, score),
                    "geocodable": score >= 50,
                    "issues": "; ".join(issues) if issues else "None",
                }
            )

        quality_df = pd.DataFrame(quality_scores)

        elapsed_time = time.time() - start_time

        # Merge back to original dataframe

        df.loc[df_to_process.index, "quality_score"] = df_to_process.merge(
            quality_df[["NPI", "quality_score"]], on="NPI", how="left"
        )["quality_score"].values

        df.loc[df_to_process.index, "geocodable"] = df_to_process.merge(
            quality_df[["NPI", "geocodable"]], on="NPI", how="left"
        )["geocodable"].values

        df.loc[df_to_process.index, "issues"] = df_to_process.merge(
            quality_df[["NPI", "issues"]], on="NPI", how="left"
        )["issues"].values

        # Fill NaN values for unchanged records with defaults

        df["quality_score"].fillna(100, inplace=True)

        df["geocodable"].fillna(True, inplace=True)

        df["issues"].fillna("None", inplace=True)

        # Statistics (for processed records only)

        geocodable = quality_df["geocodable"].sum()

        pct_geocodable = (
            (geocodable / len(quality_df)) * 100 if len(quality_df) > 0 else 100
        )

        non_geocodable = len(quality_df) - geocodable

        logger.info(f"\nQuality Summary (for changed records):")

        logger.info(f" - Records processed: {total_to_process:,} (changed/new only)")

        logger.info(f" - Geocodable: {geocodable:,} ({pct_geocodable:.1f}%)")

        logger.info(f" - Problematic: {non_geocodable:,} ({100-pct_geocodable:.1f}%)")

        logger.info(
            f" - Avg quality score: {quality_df['quality_score'].mean():.1f}/100"
        )

        logger.info(f" - Time: {elapsed_time:.1f} seconds")

        logger.info(f" - Speed: {total_to_process/elapsed_time:,.0f} records/second")

        logger.info(f"\nCost Savings by skipping problematic records:")

        logger.info(f" - Skip {non_geocodable:,} records")

        logger.info(f" - Save ${non_geocodable * 0.005:,.0f} on Google ($5 per 1k)")

        logger.info(
            f" - Save ${non_geocodable * 0.0015:,.0f} on Geoapify ($1.50 per 1k)"
        )

        logger.info("=" * 80)

        # Update metadata

        metadata = self.add_mode_completion(
            metadata,
            "quality_check",
            elapsed_time,
            total_to_process,
            ["quality_score", "geocodable", "issues"],
        )

        return df, metadata

    # ==================== GEOCODING ====================

    def geocode_batch(
        self, df: pd.DataFrame, metadata: Dict, batch_size: int = 1000
    ) -> Tuple[pd.DataFrame, Dict]:
        """Geocode addresses in batch with rate limiting."""

        logger.info("\n" + "=" * 80)

        logger.info("GEOCODING ADDRESSES")

        logger.info("=" * 80)

        start_time = time.time()

        # Check if already completed

        if self.is_mode_completed("geocode", metadata):

            logger.info("Geocoding already completed for this data. Skipping.")

            logger.info("=" * 80)

            return df, metadata

        df_to_geocode = df[df["needs_processing"]].copy()

        total_to_geocode = len(df_to_geocode)

        if total_to_geocode == 0:

            logger.info("No addresses need geocoding (all unchanged)")

            logger.info("=" * 80)

            return df, metadata

        logger.info(
            f"Geocoding {total_to_geocode:,} changed addresses (out of {len(df):,} total)..."
        )

        logger.info(f"Provider: {self.geocoder.provider}")

        logger.info(f"Rate limit: {self.geocoder.rate_limit} requests/second")

        logger.info(
            f"Estimated time: {total_to_geocode / self.geocoder.rate_limit / 60:.1f} minutes"
        )

        logger.info("=" * 80)

        df_to_geocode["latitude"] = None

        df_to_geocode["longitude"] = None

        df_to_geocode["formatted_address"] = None

        df_to_geocode["geocode_confidence"] = None

        with tqdm(total=total_to_geocode, desc="Geocoding") as pbar:

            for idx, row in df_to_geocode.iterrows():

                result = self.geocoder.geocode_address(
                    address=row.get(
                        "Provider First Line Business Practice Location Address", ""
                    ),
                    city=row.get(
                        "Provider Business Practice Location Address City Name", ""
                    ),
                    state=row.get(
                        "Provider Business Practice Location Address State Name", ""
                    ),
                    postal_code=row.get(
                        "Provider Business Practice Location Address Postal Code", ""
                    ),
                    country=row.get(
                        "Provider Business Practice Location Address Country Code (If outside U.S.)",
                        "US",
                    ),
                )

                if result:

                    df_to_geocode.at[idx, "latitude"] = result["latitude"]

                    df_to_geocode.at[idx, "longitude"] = result["longitude"]

                    df_to_geocode.at[idx, "formatted_address"] = result[
                        "formatted_address"
                    ]

                    df_to_geocode.at[idx, "geocode_confidence"] = result["confidence"]

                pbar.update(1)

        elapsed_time = time.time() - start_time

        # Merge back to full dataframe

        df.loc[df_to_geocode.index, "latitude"] = df_to_geocode["latitude"]

        df.loc[df_to_geocode.index, "longitude"] = df_to_geocode["longitude"]

        df.loc[df_to_geocode.index, "formatted_address"] = df_to_geocode[
            "formatted_address"
        ]

        df.loc[df_to_geocode.index, "geocode_confidence"] = df_to_geocode[
            "geocode_confidence"
        ]

        success_count = df_to_geocode["latitude"].notna().sum()

        logger.info("\n" + "=" * 80)

        logger.info(f"Geocoding complete:")

        logger.info(f" - Records processed: {total_to_geocode:,} (changed/new only)")

        logger.info(
            f" - Successful: {success_count:,} ({success_count/total_to_geocode*100:.1f}%)"
        )

        logger.info(f" - Failed: {total_to_geocode - success_count:,}")

        logger.info(
            f" - Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)"
        )

        logger.info(f" - Speed: {total_to_geocode/elapsed_time:.1f} records/second")

        logger.info("=" * 80)

        # Update metadata

        metadata = self.add_mode_completion(
            metadata,
            "geocode",
            elapsed_time,
            total_to_geocode,
            ["latitude", "longitude", "formatted_address", "geocode_confidence"],
        )

        return df, metadata

    # ==================== WEBSITE MAPPING ====================
    def _overpass_data_path(self) -> Path:
        """Return path to the latest cached Overpass US hospitals file under data/."""
        # Preferred: data/overpass_hospitals_us_latest.json
        latest = self.data_dir / "overpass_hospitals_us_latest.json"
        if latest.exists():
            return latest
        # Fallback: any dated file
        candidates = sorted(self.data_dir.glob("overpass_hospitals_us_*.json"))
        return candidates[-1] if candidates else latest

    def _load_overpass_hospitals(self) -> Optional[Dict]:
        """Load cached Overpass hospitals dataset (JSON)."""
        path = self._overpass_data_path()
        if not path.exists():
            logger.warning(f"Overpass hospital cache not found at {path}. Run scripts/fetch_overpass_us_hospitals.py")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load Overpass hospital cache {path}: {e}")
            return None

    def _norm_text(self, s: Optional[str]) -> str:
        """Normalize text for deterministic matching."""
        if s is None:
            return ""
        s = str(s).lower().strip()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _token_set(self, s: str) -> set:
        return set([t for t in self._norm_text(s).split(" ") if t])

    def _name_similarity(self, a: str, b: str) -> float:
        """Deterministic token Jaccard similarity."""
        ta, tb = self._token_set(a), self._token_set(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    def _address_similarity(self, a: str, b: str) -> float:
        """Deterministic token overlap for street addresses."""
        ta, tb = self._token_set(a), self._token_set(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, min(len(ta), len(tb)))

    def _extract_overpass_address(self, tags: Dict) -> str:
        hn = tags.get("addr:housenumber") or ""
        street = tags.get("addr:street") or ""
        return (str(hn).strip() + " " + str(street).strip()).strip()

    def _overpass_match_website(
        self,
        overpass: Dict,
        name: str,
        address: str,
        city: str,
        state: str,
        postal: str,
        min_score: float = 0.55,
    ) -> Optional[Dict]:
        """Match one NPI record to an Overpass hospital feature using city/state/postal filtering + name/address similarity."""
        postal5 = (postal or "")[:5]
        name_n = self._norm_text(name)
        city_n = self._norm_text(city)
        state_n = self._norm_text(state)
        addr_n = self._norm_text(address)

        best = None
        best_score = 0.0

        elements = (overpass or {}).get("elements") or []
        for el in elements:
            tags = el.get("tags") or {}
            osm_name = tags.get("name") or tags.get("official_name") or ""
            if not osm_name:
                continue

            # City/State/Postal filtering when tags exist
            osm_city = tags.get("addr:city") or ""
            osm_state = tags.get("addr:state") or ""
            osm_post = tags.get("addr:postcode") or ""

            if osm_state and state_n and self._norm_text(osm_state) != state_n:
                continue
            if osm_city and city_n and self._norm_text(osm_city) != city_n:
                continue
            if osm_post and postal5 and (str(osm_post).strip()[:5] != postal5):
                continue

            # Scoring
            ns = self._name_similarity(name_n, osm_name)
            # address tags may be missing
            osm_addr = self._extract_overpass_address(tags)
            ads = self._address_similarity(addr_n, osm_addr) if osm_addr else 0.0

            # structured bonuses if addr tags match exactly
            bonus = 0.0
            if osm_post and postal5 and str(osm_post).strip()[:5] == postal5:
                bonus += 0.10
            if osm_city and city_n and self._norm_text(osm_city) == city_n:
                bonus += 0.05

            score = 0.70 * ns + 0.25 * ads + bonus

            if score > best_score:
                # Extract website tags
                website = (
                    tags.get("website")
                    or tags.get("contact:website")
                    or tags.get("url")
                    or tags.get("operator:website")
                    or tags.get("brand:website")
                )
                operator_website = tags.get("operator:website") or tags.get("brand:website")
                best_score = score
                best = {
                    "website": self._normalize_url(website) if website else None,
                    "operator_website": self._normalize_url(operator_website) if operator_website else None,
                    "matched_name": osm_name,
                    "score": score,
                }

        if not best or best_score < min_score or (not best.get("website") and not best.get("operator_website")):
            return None

        # Derive system + location URLs
        loc_full = best.get("website")
        sys_full = best.get("operator_website") or best.get("website")
        system_home = self._root_url(sys_full)
        location_page = self._location_url(loc_full) if loc_full else None

        return {
            "system_homepage_url": system_home,
            "location_page_url": location_page,
            "source": "overpass",
            "confidence": round(best_score, 3),
            "matched_name": best.get("matched_name", ""),
        }

    def _ensure_website_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all website mapping output columns exist."""
        for col in [
            "system_homepage_url",
            "location_page_url",
            "front_page_url",  # back-compat
            "front_page_url_source",
            "front_page_url_confidence",
            "front_page_url_matched_name",
            "system_url_source",
            "location_url_source",
        ]:
            if col not in df.columns:
                df[col] = None
        return df

    def _add_is_hospital_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add/refresh boolean is_hospital column on organizations dataset."""
        if "is_hospital" not in df.columns:
            df["is_hospital"] = False
        taxonomy_cols = self._get_taxonomy_columns(df)
        # Deterministic, safe per-row heuristic
        df["is_hospital"] = df.apply(lambda r: self._is_hospital_row(r, taxonomy_cols), axis=1)
        return df

    def website_mapping_osm(
        self,
        df: Optional[pd.DataFrame],
        metadata: Dict,
        max_records: Optional[int] = None,
        nominatim_min_importance: float = 0.25,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Website mapping for hospital organizations using OSM/Nominatim only.

        - Adds boolean column `is_hospital` to the organizations dataset.
        - Uses deduped, cached structured Nominatim lookups for website tags.
        - Does NOT fallback to Overpass (use `website_mapping_overpass` mode for that).
        """
        logger.info("=" * 80)
        logger.info("WEBSITE MAPPING (OSM/Nominatim) - Hospitals")
        logger.info("=" * 80)
        start_time = time.time()

        if self.is_mode_completed("website_mapping_osm", metadata):
            logger.info("website_mapping_osm already completed. Skipping.")
            logger.info("=" * 80)
            return df if df is not None else pd.DataFrame(), metadata

        # Load orgs if not provided
        if df is None:
            org_file = self.find_split_parquet("organizations")
            if org_file is None:
                logger.error("No organizations parquet file found (npi_organizations_*.parquet).")
                return pd.DataFrame(), metadata
            logger.info(f"Loading organizations: {org_file.name}")
            df = pd.read_parquet(org_file)

        data_date = metadata.get("date") or datetime.now().strftime("%Y%m%d")
        osm_cache = self._load_osm_cache(data_date)

        # Output columns + is_hospital
        df = self._ensure_website_output_columns(df)
        df = self._add_is_hospital_column(df)

        df_hosp = df[df["is_hospital"] == True].copy()
        logger.info(f"Hospital candidates: {len(df_hosp):,} out of {len(df):,} organizations")

        # Only process changed rows (if the flag exists) AND missing system_homepage_url
        if "needs_processing" in df_hosp.columns:
            to_process = (
                (df_hosp["needs_processing"] == True)
                & (df_hosp["system_homepage_url"].isna() | (df_hosp["system_homepage_url"] == ""))
            )
        else:
            to_process = (df_hosp["system_homepage_url"].isna() | (df_hosp["system_homepage_url"] == ""))

        df_to_process = df_hosp[to_process].copy()
        if max_records:
            df_to_process = df_to_process.head(max_records)

        total = len(df_to_process)
        if total == 0:
            logger.info("No hospital records need OSM/Nominatim website mapping.")
            return df, metadata

        # Build parts & stable key
        def get_parts(row: pd.Series):
            name = self._safe_str(row.get("Provider Organization Name (Legal Business Name)")) or ""
            addr1 = self._safe_str(row.get("Provider First Line Business Practice Location Address")) or ""
            addr2 = self._safe_str(row.get("Provider Second Line Business Practice Location Address")) or ""
            address = ", ".join([p for p in [addr1, addr2] if p])
            city = self._safe_str(row.get("Provider Business Practice Location Address City Name")) or ""
            state = self._safe_str(row.get("Provider Business Practice Location Address State Name")) or ""
            postal = self._safe_str(row.get("Provider Business Practice Location Address Postal Code")) or ""
            return name, address, city, state, postal

        def make_key(parts) -> str:
            name, address, city, state, postal = parts
            obj = {
                "name": name,
                "address": address,
                "city": city,
                "state": state,
                "postal5": (postal or "")[:5],
                "country": "us",
            }
            return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

        df_to_process["_nm_parts"] = df_to_process.apply(get_parts, axis=1)
        df_to_process["_nm_key"] = df_to_process["_nm_parts"].apply(make_key)

        unique = df_to_process.drop_duplicates(subset=["_nm_key"])[["_nm_key", "_nm_parts"]]
        logger.info(f"Deduped Nominatim lookups: {len(unique):,} unique queries from {total:,} rows")

        results_by_key: Dict[str, Optional[Dict]] = {}
        new_queries = 0
        nom_hits = 0

        for _, u in tqdm(unique.iterrows(), total=len(unique), desc="Nominatim (deduped)"):
            k = u["_nm_key"]
            name, address, city, state, postal = u["_nm_parts"]

            if k in osm_cache:
                results_by_key[k] = osm_cache[k]
                continue

            new_queries += 1
            res = self._nominatim_lookup_website(
                name=name,
                address=address,
                city=city,
                state=state,
                postal_code=postal,
                country_code="us",
                min_importance=nominatim_min_importance,
            )
            osm_cache[k] = res  # cache (including None)
            results_by_key[k] = res
            if res and res.get("url"):
                nom_hits += 1

            if new_queries % 200 == 0:
                self._save_osm_cache(osm_cache, data_date)

        self._save_osm_cache(osm_cache, data_date)

        # Apply Nominatim hits
        updated_rows = 0
        for idx, r in df_to_process.iterrows():
            k = r["_nm_key"]
            res = results_by_key.get(k)
            if res is None:
                res = osm_cache.get(k)
            if not res or not res.get("url"):
                continue

            full_url = res["url"]
            root_url = self._root_url(full_url)
            loc_url = self._location_url(full_url)

            df.at[idx, "system_homepage_url"] = root_url
            df.at[idx, "location_page_url"] = loc_url
            df.at[idx, "system_url_source"] = "nominatim"
            df.at[idx, "location_url_source"] = "nominatim" if loc_url else None

            df.at[idx, "front_page_url"] = root_url
            df.at[idx, "front_page_url_source"] = "nominatim"
            df.at[idx, "front_page_url_confidence"] = res.get("confidence", 0.6)
            df.at[idx, "front_page_url_matched_name"] = res.get("matched_name", "")
            updated_rows += 1

        elapsed = time.time() - start_time
        logger.info(
            f"OSM/Nominatim website mapping summary: hospitals={len(df_hosp):,}, to_process={total:,}, "
            f"unique={len(unique):,}, nominatim_hits={nom_hits:,}, rows_updated={updated_rows:,}, time={elapsed:.1f}s"
        )

        metadata = self.add_mode_completion(
            metadata,
            "website_mapping_osm",
            elapsed,
            total,
            [
                "is_hospital",
                "system_homepage_url",
                "location_page_url",
                "system_url_source",
                "location_url_source",
                "front_page_url",
                "front_page_url_source",
                "front_page_url_confidence",
                "front_page_url_matched_name",
            ],
        )
        return df, metadata

    def website_mapping_overpass(
        self,
        df: Optional[pd.DataFrame],
        metadata: Dict,
        max_records: Optional[int] = None,
        min_score: float = 0.55,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Website mapping for hospital organizations using cached Overpass dataset only.

        - Adds boolean column `is_hospital` to the organizations dataset.
        - Matches missing hospital website URLs via city/state/postal filtering + name/address similarity.
        - Requires cached Overpass US hospitals JSON (see warning if missing).
        """
        logger.info("=" * 80)
        logger.info("WEBSITE MAPPING (Overpass) - Hospitals")
        logger.info("=" * 80)
        start_time = time.time()

        if self.is_mode_completed("website_mapping_overpass", metadata):
            logger.info("website_mapping_overpass already completed. Skipping.")
            logger.info("=" * 80)
            return df if df is not None else pd.DataFrame(), metadata

        # Load orgs if not provided
        if df is None:
            org_file = self.find_split_parquet("organizations")
            if org_file is None:
                logger.error("No organizations parquet file found (npi_organizations_*.parquet).")
                return pd.DataFrame(), metadata
            logger.info(f"Loading organizations: {org_file.name}")
            df = pd.read_parquet(org_file)

        overpass = self._load_overpass_hospitals()
        if overpass is None:
            logger.warning("Overpass hospital dataset not available. Skipping Overpass website mapping.")
            return df, metadata

        # Output columns + is_hospital
        df = self._ensure_website_output_columns(df)
        df = self._add_is_hospital_column(df)

        df_hosp = df[df["is_hospital"] == True].copy()
        logger.info(f"Hospital candidates: {len(df_hosp):,} out of {len(df):,} organizations")

        # Only process changed rows (if flag exists) AND missing system_homepage_url
        if "needs_processing" in df_hosp.columns:
            to_process = (
                (df_hosp["needs_processing"] == True)
                & (df_hosp["system_homepage_url"].isna() | (df_hosp["system_homepage_url"] == ""))
            )
        else:
            to_process = (df_hosp["system_homepage_url"].isna() | (df_hosp["system_homepage_url"] == ""))

        df_to_process = df_hosp[to_process].copy()
        if max_records:
            df_to_process = df_to_process.head(max_records)

        total = len(df_to_process)
        if total == 0:
            logger.info("No hospital records need Overpass website mapping.")
            return df, metadata

        # Build parts & stable key
        def get_parts(row: pd.Series):
            name = self._safe_str(row.get("Provider Organization Name (Legal Business Name)")) or ""
            addr1 = self._safe_str(row.get("Provider First Line Business Practice Location Address")) or ""
            addr2 = self._safe_str(row.get("Provider Second Line Business Practice Location Address")) or ""
            address = ", ".join([p for p in [addr1, addr2] if p])
            city = self._safe_str(row.get("Provider Business Practice Location Address City Name")) or ""
            state = self._safe_str(row.get("Provider Business Practice Location Address State Name")) or ""
            postal = self._safe_str(row.get("Provider Business Practice Location Address Postal Code")) or ""
            return name, address, city, state, postal

        def make_key(parts) -> str:
            name, address, city, state, postal = parts
            obj = {
                "name": name,
                "address": address,
                "city": city,
                "state": state,
                "postal5": (postal or "")[:5],
                "country": "us",
            }
            return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

        df_to_process["_op_parts"] = df_to_process.apply(get_parts, axis=1)
        df_to_process["_op_key"] = df_to_process["_op_parts"].apply(make_key)

        unique = df_to_process.drop_duplicates(subset=["_op_key"])[["_op_key", "_op_parts"]]
        logger.info(f"Deduped Overpass matches: {len(unique):,} unique match attempts from {total:,} rows")

        overpass_by_key: Dict[str, Optional[Dict]] = {}
        for _, u in tqdm(unique.iterrows(), total=len(unique), desc="Overpass (deduped)"):
            k = u["_op_key"]
            name, address, city, state, postal = u["_op_parts"]
            overpass_by_key[k] = self._overpass_match_website(
                overpass=overpass,
                name=name,
                address=address,
                city=city,
                state=state,
                postal=postal,
                min_score=min_score,
            )

        # Apply Overpass matches
        updated_rows = 0
        hits = 0
        for idx, r in df_to_process.iterrows():
            k = r["_op_key"]
            mres = overpass_by_key.get(k)
            if not mres:
                continue

            df.at[idx, "system_homepage_url"] = mres.get("system_homepage_url")
            df.at[idx, "location_page_url"] = mres.get("location_page_url")
            df.at[idx, "system_url_source"] = "overpass"
            df.at[idx, "location_url_source"] = "overpass" if mres.get("location_page_url") else None

            df.at[idx, "front_page_url"] = mres.get("system_homepage_url")
            df.at[idx, "front_page_url_source"] = "overpass"
            df.at[idx, "front_page_url_confidence"] = mres.get("confidence")
            df.at[idx, "front_page_url_matched_name"] = mres.get("matched_name")
            updated_rows += 1
            hits += 1

        elapsed = time.time() - start_time
        logger.info(
            f"Overpass website mapping summary: hospitals={len(df_hosp):,}, to_process={total:,}, unique={len(unique):,}, "
            f"overpass_hits={hits:,}, rows_updated={updated_rows:,}, time={elapsed:.1f}s"
        )

        metadata = self.add_mode_completion(
            metadata,
            "website_mapping_overpass",
            elapsed,
            total,
            [
                "is_hospital",
                "system_homepage_url",
                "location_page_url",
                "system_url_source",
                "location_url_source",
                "front_page_url",
                "front_page_url_source",
                "front_page_url_confidence",
                "front_page_url_matched_name",
            ],
        )
        return df, metadata

    # Backward-compatible wrapper (kept so existing callers don't break)
    def website_mapping(
        self,
        df: Optional[pd.DataFrame],
        metadata: Dict,
        max_records: Optional[int] = None,
        nominatim_min_importance: float = 0.25,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Deprecated combined mode. Runs OSM/Nominatim then Overpass fallback."""
        df, metadata = self.website_mapping_osm(
            df=df,
            metadata=metadata,
            max_records=max_records,
            nominatim_min_importance=nominatim_min_importance,
        )
        df, metadata = self.website_mapping_overpass(df=df, metadata=metadata, max_records=max_records)
        return df, metadata

# ==================== ENTITY MAPPING ====================
    def entity_mapping(
        self, df: Optional[pd.DataFrame], metadata: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Map individuals to their organizations (keyed by Organization NPI).

        Adds/updates on INDIVIDUALS:
          - mapped_org_npi: Organization NPI inferred from address + phone
          - mapped_org_name: Organization legal name (debug/convenience)

        Phone is OPTIONAL.
        Only processes rows flagged with needs_processing == True (when present).
        """
        logger.info("\n" + "=" * 80)
        logger.info("ENTITY MAPPING - Map individuals to organizations (maps to ORG NPI)")
        logger.info("=" * 80)
        start_time = time.time()

        if df is None:
            individuals_file = self.find_split_parquet("individuals")
            if individuals_file is None:
                logger.error("Cannot find individuals parquet file (npi_individuals_*.parquet)")
                logger.info("=" * 80)
                return pd.DataFrame(), metadata
            logger.info(f"Loading individuals: {individuals_file.name}")
            df = pd.read_parquet(individuals_file)

        if self.is_mode_completed("entity_mapping", metadata):
            logger.info("Entity mapping already completed for this data. Skipping.")
            logger.info("=" * 80)
            return df, metadata

        if "mapped_org_npi" not in df.columns:
            df["mapped_org_npi"] = None
        if "mapped_org_name" not in df.columns:
            df["mapped_org_name"] = None

        logger.info("\nLoading organizations data...")
        organizations_file = self.find_split_parquet("organizations")
        if organizations_file is None:
            logger.error("Cannot find organizations parquet file (npi_organizations_*.parquet)")
            logger.info("=" * 80)
            return df, metadata
        logger.info(f"Loading organizations: {organizations_file.name}")

        try:
            df_org = pd.read_parquet(organizations_file)
            logger.info(f"Loaded {len(df_org):,} organizations")
        except Exception as e:
            logger.error(f"Failed to load organizations: {e}")
            logger.info("=" * 80)
            return df, metadata

        logger.info("\nBuilding organization lookup by address + phone...")
        org_lookup: Dict[str, Dict[str, Dict[str, str]]] = {}

        for _, org_row in df_org.iterrows():
            org_npi = self._safe_str(org_row.get("NPI"))
            if org_npi is not None:
                try:
                    org_npi = str(int(float(org_npi)))
                except Exception:
                    org_npi = self._safe_str(org_npi)

            addr1 = self._safe_str(org_row.get("Provider First Line Business Practice Location Address"))
            addr2 = self._safe_str(org_row.get("Provider Second Line Business Practice Location Address"))
            address = ", ".join([p for p in [addr1, addr2] if p])
            city = self._safe_str(org_row.get("Provider Business Practice Location Address City Name"))
            state = self._safe_str(org_row.get("Provider Business Practice Location Address State Name"))
            postal = self._safe_str(org_row.get("Provider Business Practice Location Address Postal Code"))
            phone = self._safe_phone(org_row.get("Provider Business Practice Location Address Telephone Number"))
            org_name = self._safe_str(org_row.get("Provider Organization Name (Legal Business Name)"))

            if org_npi and address and city and state and postal and org_name:
                address_key = f"{address}|{city}|{state}|{postal[:5]}"
                if address_key not in org_lookup:
                    org_lookup[address_key] = {}
                phone_key = phone if phone else "__no_phone__"
                org_lookup[address_key][phone_key] = {"org_npi": org_npi, "org_name": org_name}

        logger.info(f"Built lookup with {len(org_lookup):,} unique addresses")

        if "needs_processing" in df.columns:
            df_to_process = df[df["needs_processing"] == True].copy()
        else:
            df_to_process = df.copy()

        total_to_process = len(df_to_process)
        if total_to_process == 0:
            logger.info("No records need entity mapping (all unchanged)")
            logger.info("=" * 80)
            return df, metadata

        logger.info(f"\nMapping {total_to_process:,} records to organizations...")

        def map_to_org(row: pd.Series):
            addr1 = self._safe_str(row.get("Provider First Line Business Practice Location Address"))
            addr2 = self._safe_str(row.get("Provider Second Line Business Practice Location Address"))
            address = ", ".join([p for p in [addr1, addr2] if p])
            city = self._safe_str(row.get("Provider Business Practice Location Address City Name"))
            state = self._safe_str(row.get("Provider Business Practice Location Address State Name"))
            postal = self._safe_str(row.get("Provider Business Practice Location Address Postal Code"))
            phone = self._safe_phone(row.get("Provider Business Practice Location Address Telephone Number"))

            if address and city and state and postal:
                address_key = f"{address}|{city}|{state}|{postal[:5]}"
                bucket = org_lookup.get(address_key)
                if not bucket:
                    return (None, None)

                if phone and phone in bucket:
                    rec = bucket[phone]
                    return (rec.get("org_npi"), rec.get("org_name"))

                if "__no_phone__" in bucket:
                    rec = bucket["__no_phone__"]
                    return (rec.get("org_npi"), rec.get("org_name"))

                rec = next(iter(bucket.values()))
                return (rec.get("org_npi"), rec.get("org_name"))

            return (None, None)

        mapped = df_to_process.apply(map_to_org, axis=1)
        df.loc[df_to_process.index, "mapped_org_npi"] = mapped.apply(lambda x: x[0])
        df.loc[df_to_process.index, "mapped_org_name"] = mapped.apply(lambda x: x[1])

        elapsed_time = time.time() - start_time
        matched_count = int(df.loc[df_to_process.index, "mapped_org_npi"].notna().sum())
        unmatched_count = total_to_process - matched_count

        logger.info("\nEntity Mapping Results (processed rows only):")
        logger.info(f" - Records processed: {total_to_process:,}")
        logger.info(f" - Successfully mapped: {matched_count:,} ({matched_count/total_to_process*100:.1f}%)")
        logger.info(f" - Not mapped: {unmatched_count:,} ({unmatched_count/total_to_process*100:.1f}%)")
        logger.info(f" - Time: {elapsed_time:.1f} seconds")
        logger.info(f" - Speed: {total_to_process/elapsed_time:,.0f} records/second")
        logger.info("=" * 80)

        metadata = self.add_mode_completion(
            metadata,
            "entity_mapping",
            elapsed_time,
            total_to_process,
            ["mapped_org_npi", "mapped_org_name"],
        )
        return df, metadata

# ==================== ORG ENTITY (Heuristic A) ====================
    def _compute_org_entity_id(self, org_row: pd.Series) -> Optional[str]:
        """Compute org_entity_id using Heuristic A: normalized name + address (+ optional phone)."""
        name = self._safe_str(org_row.get("Provider Organization Name (Legal Business Name)"))
        addr1 = self._safe_str(org_row.get("Provider First Line Business Practice Location Address"))
        addr2 = self._safe_str(org_row.get("Provider Second Line Business Practice Location Address"))
        city = self._safe_str(org_row.get("Provider Business Practice Location Address City Name"))
        state = self._safe_str(org_row.get("Provider Business Practice Location Address State Name"))
        postal = self._safe_str(org_row.get("Provider Business Practice Location Address Postal Code"))
        phone = self._safe_phone(org_row.get("Provider Business Practice Location Address Telephone Number"))

        if not (name and addr1 and city and state and postal):
            return None

        address = ", ".join([p for p in [addr1, addr2] if p])
        postal5 = (postal or "")[:5]

        obj = {
            "name": self._norm_text(name),
            "address": self._norm_text(address),
            "city": self._norm_text(city),
            "state": self._norm_text(state),
            "postal5": postal5,
            "phone": phone or "",
        }
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()

# ==================== PROVIDER COUNT ====================
    def compute_provider_count(
        self, df: Optional[pd.DataFrame], metadata: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Compute provider_count + provider_count_entity for ORGANIZATIONS."""
        logger.info("\n" + "=" * 80)
        logger.info("PROVIDER COUNT - org NPI + org entity rollup (Heuristic A)")
        logger.info("=" * 80)
        start_time = time.time()

        if self.is_mode_completed("provider_count", metadata):
            logger.info("provider_count already completed for this data. Skipping.")
            logger.info("=" * 80)
            return df if df is not None else pd.DataFrame(), metadata

        if df is None:
            org_file = self.find_split_parquet("organizations")
            if org_file is None:
                logger.error("No organizations parquet file found (npi_organizations_*.parquet).")
                return pd.DataFrame(), metadata
            logger.info(f"Loading organizations: {org_file.name}")
            df = pd.read_parquet(org_file)

        for col, default in [("org_entity_id", None), ("provider_count", 0), ("provider_count_entity", 0)]:
            if col not in df.columns:
                df[col] = default

        df["org_entity_id"] = df.apply(self._compute_org_entity_id, axis=1)

        data_date = metadata.get("date")
        ind_file = None
        if data_date:
            same_date = self.processed_dir / f"npi_individuals_processed_{data_date}.parquet"
            if same_date.exists():
                ind_file = same_date
        if ind_file is None:
            ind_file = self.find_latest_processed_parquet("npi_individuals_processed")

        if ind_file is None or (not ind_file.exists()):
            logger.error("No individuals processed parquet found. Run entity_mapping first.")
            return df, metadata

        logger.info(f"Using individuals processed file: {ind_file.name}")
        df_ind = pd.read_parquet(ind_file)
        if "mapped_org_npi" not in df_ind.columns:
            logger.error("Individuals processed file missing 'mapped_org_npi'. Re-run entity_mapping.")
            return df, metadata

        df_ind = df_ind[["NPI", "mapped_org_npi"]].copy()
        df_ind["NPI"] = df_ind["NPI"].astype("string")
        df_ind["mapped_org_npi"] = df_ind["mapped_org_npi"].astype("string")
        valid_map = df_ind["mapped_org_npi"].notna() & (df_ind["mapped_org_npi"] != "")

        counts_npi = df_ind.loc[valid_map].groupby("mapped_org_npi")["NPI"].nunique()
        df["provider_count"] = df["NPI"].astype("string").map(counts_npi).fillna(0).astype("int64")

        org_npi_to_entity = df.set_index(df["NPI"].astype("string"))["org_entity_id"]
        df_ind["org_entity_id"] = df_ind["mapped_org_npi"].map(org_npi_to_entity)
        valid_entity = df_ind["org_entity_id"].notna() & (df_ind["org_entity_id"] != "")
        counts_entity = df_ind.loc[valid_entity].groupby("org_entity_id")["NPI"].nunique()
        df["provider_count_entity"] = df["org_entity_id"].map(counts_entity).fillna(0).astype("int64")

        elapsed_time = time.time() - start_time
        nonzero_npi = int((df["provider_count"] > 0).sum())
        nonzero_ent = int((df["provider_count_entity"] > 0).sum())
        logger.info(
            f"provider_count computed for {len(df):,} orgs; nonzero(provider_count)={nonzero_npi:,}; "
            f"nonzero(provider_count_entity)={nonzero_ent:,}; time={elapsed_time:.1f}s"
        )
        logger.info("=" * 80)

        metadata = self.add_mode_completion(
            metadata,
            "provider_count",
            elapsed_time,
            len(df),
            ["org_entity_id", "provider_count", "provider_count_entity"],
        )
        return df, metadata




def get_mode_input() -> str:
    """Get mode from user input."""
    print("\nSelect processing mode:")
    print(" 1 or 'geocode' - Full geocoding (expensive)")
    print(" 2 or 'geohash' - Free spatial indexing")
    print(" 3 or 'quality_check' - Free quality analysis")
    print(" 4 or 'entity_mapping' - Map individuals to organizations (ORG NPI)")
    print(" 5 or 'website_mapping_osm' - Map hospital orgs to websites using OSM/Nominatim (cached)")
    print(" 6 or 'website_mapping_overpass' - Map hospital orgs to websites using Overpass cache (no API calls)")
    print(" 7 or 'provider_count' - Add provider_count + provider_count_entity to organizations")

    mode_input = (
        input(
            "\nMode [1/2/3/4/5/6/7 or geocode/geohash/quality_check/entity_mapping/website_mapping_osm/website_mapping_overpass/provider_count]: "
        )
        .strip()
        .lower()
    )

    mode_map = {
        "1": "geocode",
        "2": "geohash",
        "3": "quality_check",
        "4": "entity_mapping",
        "5": "website_mapping_osm",
        "6": "website_mapping_overpass",
        "7": "provider_count",
    }

    mode = mode_map.get(mode_input, mode_input)

    valid_modes = [
        "geocode",
        "geohash",
        "quality_check",
        "entity_mapping",
        "website_mapping_osm",
        "website_mapping_overpass",
        "website_mapping",
        "provider_count",
    ]

    if mode not in valid_modes:
        logger.error(f"Invalid mode: {mode_input}")
        logger.error(f"Valid modes: {', '.join(valid_modes)} or 1-7")
        return None

    return mode


def main():

    """Main execution function."""
    print("\n" + "=" * 80)
    print("NPI REGISTRY - UNIFIED PROCESSING WITH METADATA TRACKING")
    print("=" * 80)
    print("\nProcessing Modes:")
    print(" 1. GEOCODING - Full address geocoding (expensive API calls)")
    print(" 2. GEOHASHING - Free spatial indexing (no API calls)")
    print(" 3. QUALITY_CHECK - Free data quality analysis (no API calls)")
    print(" 4. ENTITY_MAPPING - Map individuals to organizations (free, no API calls)")
    print(" 5. WEBSITE_MAPPING_OSM - Map hospital orgs to websites using OSM/Nominatim (cached)")
    print(" 6. WEBSITE_MAPPING_OVERPASS - Map hospital orgs to websites using Overpass cache (no API calls)")
    print(" 7. PROVIDER_COUNT - Add provider_count + provider_count_entity to organizations")
    print("\nArchitecture:")
    print(" ✓ Single 'processed_data' folder")
    print(" ✓ One parquet per run with processing columns combined")
    print(" ✓ Metadata.json for audit trail and processing status")
    print(" ✓ Smart resumption: Continue from where you left off")
    print("=" * 80)

    mode = get_mode_input()
    if mode is None:
        return

    # Configure geocoding provider only when needed
    provider = "nominatim"
    api_key = None
    if mode == "geocode":
        provider = (
            input("\nGeocoding provider (nominatim/google/geoapify/azure) [nominatim]: ")
            .strip()
            .lower()
            or "nominatim"
        )
        if provider != "nominatim":
            api_key = input(f"Enter {provider} API key: ").strip()

    # Output naming: entity mapping writes a dedicated individuals-processed parquet
    output_prefix = ("npi_individuals_processed" if mode == "entity_mapping" else "npi_organizations_processed" if mode in ("website_mapping_osm", "website_mapping_overpass", "website_mapping", "provider_count") else "npi_processed")
    processor = NPIRegistryProcessed(
        data_dir="../data",
        geocoding_provider=provider,
        api_key=api_key,
        output_prefix=output_prefix,
    )

    print("\nLoading data...")

    # For special modes we explicitly choose the split parquet
    if mode == "entity_mapping":
        data_file = processor.find_split_parquet("individuals")
        if data_file is None:
            logger.error("No individuals parquet file found (npi_individuals_*.parquet). Exiting.")
            return
        df = pd.read_parquet(data_file)
    elif mode in ("website_mapping_osm", "website_mapping_overpass", "website_mapping", "provider_count"):
        data_file = processor.find_split_parquet("organizations")
        if data_file is None:
            logger.error("No organizations parquet file found (npi_organizations_*.parquet). Exiting.")
            return
        df = pd.read_parquet(data_file)
    else:
        df = processor.load_data()
        data_file = processor.find_data_file()

    if df is None or data_file is None:
        logger.error("\nFailed to load data. Exiting.")
        return

    data_date = processor.get_data_date(data_file)
    if not data_date:
        data_date = datetime.now().strftime("%Y%m%d")
        logger.warning(f"Could not extract date from filename. Using today: {data_date}")

    logger.info(f"Data date: {data_date}")

    # Load or create processed data
    df, metadata = processor.load_or_create_processed_data(df, data_date)

        # Identify changed addresses / hashes (skip for provider_count)
    if mode != "provider_count":
        # Identify changed addresses
        df = processor.identify_changed_addresses(df, data_date)

        # Save hashes
        logger.info("\n" + "=" * 80)
        logger.info("SAVING HASHES FOR NEXT WEEK")
        logger.info("=" * 80)
        processor.save_current_hashes(df, data_date)
        logger.info("=" * 80)

# Execute based on mode
    if mode == "geohash":
        df, metadata = processor.apply_geohashing(df, metadata)
    elif mode == "quality_check":
        df, metadata = processor.data_quality_check(df, metadata)
    elif mode == "geocode":
        df, metadata = processor.geocode_batch(df, metadata, batch_size=100)
    elif mode == "entity_mapping":
        df, metadata = processor.entity_mapping(df, metadata)
    elif mode == "website_mapping_osm":
        df, metadata = processor.website_mapping_osm(df, metadata)
    elif mode == "website_mapping_overpass":
        df, metadata = processor.website_mapping_overpass(df, metadata)
    elif mode == "website_mapping":
        df, metadata = processor.website_mapping(df, metadata)
    elif mode == "provider_count":
        df, metadata = processor.compute_provider_count(df, metadata)
    # Save processed data and metadata
    processor.save_processed_data(df, metadata, data_date)

    print(f"\n✓ Processing complete!")
    print(f"Processed data saved to: {processor.processed_dir}")


if __name__ == "__main__":
    main()
