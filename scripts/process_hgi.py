#!/usr/bin/env python3
"""
Modes:
 1. website_mapping_osm       - Map hospitals to websites using OSM/Nominatim (cached)
 2. website_mapping_overpass  - Map hospitals to websites using Overpass cache (no API calls)
 3. both                      - Run Overpass first, then OSM/Nominatim for remaining gaps
 4. provider_count            - Add provider_count to hospitals using processed PPEF individuals

Behavior:
 • Assumes base data dir = ../data (no prompt)
 • Auto-discovers latest ../data/parquet/hgi_*.parquet (no prompt)
 • Writes to ../data/processed_data/hgi_processed_YYYYMMDD.parquet
 • Only prompts for the mode (to match hash_based_processing.py)
"""

import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

# ---------------------------- logging ---------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("process_hospitals")

# ---------------------------- helpers ---------------------------------
def _safe_str(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None

def _normalize_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse, urlunparse
    url = _safe_str(url)
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

def _root_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse, urlunparse
    url = _normalize_url(url)
    if not url:
        return None
    try:
        p = urlparse(url)
        if not p.netloc:
            return None
        return urlunparse(("https", p.netloc.lower(), "/", "", "", ""))
    except Exception:
        return None

def _location_url(url: Optional[str]) -> Optional[str]:
    from urllib.parse import urlparse
    url = _normalize_url(url)
    if not url:
        return None
    try:
        p = urlparse(url)
        if p.path and p.path != "/":
            return url
    except Exception:
        pass
    return None

# ------------------------- OSM/Nominatim -------------------------------
class NominatimClient:
    BASE_URL = "https://nominatim.openstreetmap.org/search"

    def __init__(self, cache_path: Path, offline_only: bool = False):
        self.cache_path = cache_path
        self.offline_only = offline_only
        self.cache: Dict[str, Optional[dict]] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception as e:
                logger.warning("Failed to read cache %s: %s", self.cache_path, e)
                self.cache = {}

    def _save(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2))
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", self.cache_path, e)

    @staticmethod
    def _make_key(name: str, address: str, city: str, state: str, postal: str) -> str:
        obj = {"name": name, "address": address, "city": city, "state": state, "postal5": (postal or "")[:5], "country": "us"}
        return str(abs(hash(json.dumps(obj, sort_keys=True))))

    def lookup_website(self, name: str, address: str, city: str, state: str, postal: str, min_importance: float = 0.25) -> Optional[dict]:
        key = self._make_key(name, address, city, state, postal)
        if key in self.cache:
            return self.cache[key]
        if self.offline_only:
            self.cache[key] = None
            self._save()
            return None
        headers = {"User-Agent": "HGI-WebsiteMapper/1.0"}
        params = {"format": "jsonv2", "limit": 1, "addressdetails": 1, "extratags": 1, "namedetails": 1, "countrycodes": "us"}
        addr = _safe_str(address)
        if addr and city and state:
            params.update({"street": addr, "city": city, "state": state})
            if postal:
                params["postalcode"] = (postal or "")[:5]
        else:
            q = ", ".join([p for p in [name, address, city, state, (postal or "")[:5], "US"] if p])
            params["q"] = q
        for attempt in range(4):
            try:
                time.sleep(0.5 + random.random() * 0.3)
                r = requests.get(self.BASE_URL, params=params, headers=headers, timeout=20)
                if r.status_code != 200:
                    if r.status_code in (429, 500, 502, 503, 504):
                        time.sleep((2 ** attempt) * (0.5 + random.random() * 0.3))
                        continue
                    self.cache[key] = None
                    self._save()
                    return None
                data = r.json() or []
                if not data:
                    self.cache[key] = None
                    self._save()
                    return None
                item = data[0]
                importance = float(item.get("importance") or 0.0)
                if importance < min_importance:
                    self.cache[key] = None
                    self._save()
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
                    self.cache[key] = None
                    self._save()
                    return None
                matched_name = (item.get("namedetails") or {}).get("name") or item.get("display_name", "")
                res = {
                    "url": _normalize_url(website),
                    "confidence": importance if importance else 0.6,
                    "matched_name": matched_name,
                    "raw": {"importance": importance, "place_rank": item.get("place_rank"), "type": item.get("type"), "class": item.get("class")},
                }
                self.cache[key] = res
                self._save()
                return res
            except Exception as e:
                logger.info("Nominatim error %s (attempt %s)", e, attempt + 1)
                time.sleep((2 ** attempt) * (0.5 + random.random() * 0.3))
        self.cache[key] = None
        self._save()
        return None

# ------------------------- Overpass cache -------------------------------
def overpass_cache_path(data_dir: Path) -> Path:
    pref = data_dir / "overpass_hospitals_us_latest.json"
    if pref.exists():
        return pref
    candidates = sorted(data_dir.glob("overpass_hospitals_us_*.json"))
    return candidates[-1] if candidates else pref

def load_overpass(data_dir: Path) -> Optional[dict]:
    p = overpass_cache_path(data_dir)
    if not p.exists():
        logger.warning("Overpass cache not found at %s", p)
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("Failed to read Overpass cache %s: %s", p, e)
        return None

# ------------------------- Matching utils -------------------------------
def _norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str) -> set:
    return set([t for t in _norm_text(s).split(" ") if t])

def _name_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))

def _address_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))

def _extract_overpass_address(tags: Dict) -> str:
    hn = tags.get("addr:housenumber") or ""
    street = tags.get("addr:street") or ""
    return (str(hn).strip() + " " + str(street).strip()).strip()

# ------------------------- Processor -----------------------------------
class HospitalWebsiteProcessor:
    def __init__(self, data_dir: Path = Path("../data")):
        self.data_dir = data_dir
        self.parquet_dir = self.data_dir / "parquet"
        self.processed_dir = self.data_dir / "processed_data"
        self.hash_dir = self.data_dir / "hashes"
        for d in [self.processed_dir, self.hash_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_date_from_filename(path: Path) -> str:
        m = re.search(r"(\d{8})", str(path))
        return m.group(1) if m else datetime.now().strftime("%Y%m%d")

    def default_output_path(self, date_str: str) -> Path:
        return self.processed_dir / f"hgi_processed_{date_str}.parquet"

    def find_latest_hgi(self) -> Optional[Path]:
        """Return the latest hgi_YYYYMMDD.parquet only (exclude hgi_bf_geo_*)."""
        import re
        candidates = []
        for f in sorted(self.parquet_dir.glob('hgi_*.parquet')):
            if re.match(r'^hgi_\d{8}\.parquet$', f.name):
                candidates.append(f)
        return candidates[-1] if candidates else None

    def find_latest_processed_individuals(self) -> Optional[Path]:
        candidates = sorted(self.processed_dir.glob("ppef_individuals_processed_*.parquet"))
        return candidates[-1] if candidates else None

    # overpass matching
    def overpass_match(self, op: dict, name: str, addr: str, city: str, state: str, postal: str, min_score: float = 0.55) -> Optional[dict]:
        postal5 = (postal or "")[:5]
        name_n = _norm_text(name)
        city_n = _norm_text(city)
        state_n = _norm_text(state)
        addr_n = _norm_text(addr)
        best = None
        best_score = 0.0
        elements = (op or {}).get("elements") or []
        for el in elements:
            tags = el.get("tags") or {}
            osm_name = tags.get("name") or tags.get("official_name") or ""
            if not osm_name:
                continue
            osm_city = tags.get("addr:city") or ""
            osm_state = tags.get("addr:state") or ""
            osm_post = tags.get("addr:postcode") or ""
            if osm_state and state_n and _norm_text(osm_state) != state_n:
                continue
            if osm_city and city_n and _norm_text(osm_city) != city_n:
                continue
            if osm_post and postal5 and str(osm_post).strip()[:5] != postal5:
                continue
            ns = _name_similarity(name_n, osm_name)
            osm_addr = _extract_overpass_address(tags)
            ads = _address_similarity(addr_n, osm_addr) if osm_addr else 0.0
            bonus = 0.0
            if osm_post and postal5 and str(osm_post).strip()[:5] == postal5:
                bonus += 0.10
            if osm_city and city_n and _norm_text(osm_city) == city_n:
                bonus += 0.05
            score = 0.70 * ns + 0.25 * ads + bonus
            if score > best_score:
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
                    "website": _normalize_url(website) if website else None,
                    "operator_website": _normalize_url(operator_website) if operator_website else None,
                    "matched_name": osm_name,
                    "score": score,
                }
        if not best or best_score < min_score or (not best.get("website") and not best.get("operator_website")):
            return None
        loc_full = best.get("website")
        sys_full = best.get("operator_website") or best.get("website")
        return {
            "system_url": _root_url(sys_full),
            "location_url": _location_url(loc_full) if loc_full else None,
            "matched_name": best.get("matched_name", ""),
            "match_confidence": round(best_score, 3),
            "source": "overpass",
        }

    # runners
    def run_overpass(self, df: pd.DataFrame, data_dir: Path, date_str: str, max_records: Optional[int]):
        print("\n" + "=" * 80)
        print("WEBSITE MAPPING (Overpass) - Hospitals")
        print("=" * 80)

        op = load_overpass(data_dir)
        if op is None:
            logger.warning("Overpass cache missing. Skipping overpass mapping.")
            return df

        # Ensure output columns
        for c in ["system_url", "location_url", "system_url_source", "location_url_source", "matched_name", "match_confidence"]:
            if c not in df.columns:
                df[c] = None

        to_proc = df[df["system_url"].isna() | (df["system_url"] == "")].copy()
        total = len(to_proc)
        print(f"Candidates without system_url: {total:,}")

        if max_records:
            to_proc = to_proc.head(max_records)

        hits = 0
        for idx, r in to_proc.iterrows():
            name = _safe_str(r.get("Facility Name")) or ""
            addr = _safe_str(r.get("Address")) or ""
            city = _safe_str(r.get("City/Town")) or ""
            state = _safe_str(r.get("State")) or ""
            postal = _safe_str(r.get("ZIP Code")) or ""
            m = self.overpass_match(op, name, addr, city, state, postal)
            if not m:
                continue
            df.at[idx, "system_url"] = m["system_url"]
            df.at[idx, "location_url"] = m["location_url"]
            df.at[idx, "system_url_source"] = "overpass"
            df.at[idx, "location_url_source"] = "overpass" if m["location_url"] else None
            df.at[idx, "matched_name"] = m["matched_name"]
            df.at[idx, "match_confidence"] = m["match_confidence"]
            hits += 1

        print(f"\nOverpass mapping summary:")
        print(f" - Candidates processed: {total:,}")
        print(f" - Rows updated:        {hits:,}")
        print("=" * 80)
        return df

    def run_osm(self, df: pd.DataFrame, date_str: str, max_records: Optional[int], offline_only: bool, min_importance: float):
        print("\n" + "=" * 80)
        print("WEBSITE MAPPING (OSM/Nominatim) - Hospitals")
        print("=" * 80)

        cache_path = self.hash_dir / f"osm_website_cache_hgi_{date_str}.json"
        client = NominatimClient(cache_path=cache_path, offline_only=offline_only)

        # Ensure output columns
        for c in ["system_url", "location_url", "system_url_source", "location_url_source", "matched_name", "match_confidence"]:
            if c not in df.columns:
                df[c] = None

        to_proc = df[df["system_url"].isna() | (df["system_url"] == "")].copy()
        total = len(to_proc)
        print(f"Candidates without system_url: {total:,}")
        print(f"Offline-only: {offline_only} | Min importance: {min_importance}")

        if max_records:
            to_proc = to_proc.head(max_records)

        hits = 0
        for idx, r in to_proc.iterrows():
            name = _safe_str(r.get("Facility Name")) or ""
            addr = _safe_str(r.get("Address")) or ""
            city = _safe_str(r.get("City/Town")) or ""
            state = _safe_str(r.get("State")) or ""
            postal = _safe_str(r.get("ZIP Code")) or ""
            res = client.lookup_website(name=name, address=addr, city=city, state=state, postal=postal, min_importance=min_importance)
            if not res or not res.get("url"):
                continue
            full_url = res["url"]
            root = _root_url(full_url)
            loc = _location_url(full_url)
            df.at[idx, "system_url"] = root
            df.at[idx, "location_url"] = loc
            df.at[idx, "system_url_source"] = "nominatim"
            df.at[idx, "location_url_source"] = "nominatim" if loc else None
            df.at[idx, "matched_name"] = res.get("matched_name", "")
            df.at[idx, "match_confidence"] = res.get("confidence", 0.6)
            hits += 1

        print(f"\nOSM/Nominatim mapping summary:")
        print(f" - Candidates processed: {total:,}")
        print(f" - Rows updated:        {hits:,}")
        print("=" * 80)
        return df

    def run_provider_count(self, df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("PROVIDER COUNT - Hospitals")
        print("=" * 80)

        individuals_path = self.find_latest_processed_individuals()
        if individuals_path is None or not individuals_path.exists():
            logger.warning(
                "No processed PPEF individuals parquet found at %s. Run process_individuals.py first.",
                self.processed_dir,
            )
            if "provider_count" not in df.columns:
                df["provider_count"] = 0
            return df

        print(f"Using processed individuals file: {individuals_path}")
        df_individuals = pd.read_parquet(individuals_path)
        if "provider_count" not in df.columns:
            df["provider_count"] = 0

        if "mapped_facility_id" not in df_individuals.columns:
            logger.warning("Processed individuals file is missing 'mapped_facility_id'.")
            df["provider_count"] = df["provider_count"].fillna(0).astype("int64")
            return df

        id_col = "NPI" if "NPI" in df_individuals.columns else "ENRLMT_ID" if "ENRLMT_ID" in df_individuals.columns else None
        if id_col is None:
            logger.warning("Processed individuals file is missing both 'NPI' and 'ENRLMT_ID'.")
            df["provider_count"] = df["provider_count"].fillna(0).astype("int64")
            return df

        mapped = df_individuals[["mapped_facility_id", id_col]].copy()
        mapped["mapped_facility_id"] = mapped["mapped_facility_id"].astype("string").fillna("").str.strip()
        mapped[id_col] = mapped[id_col].astype("string").fillna("").str.strip()
        mapped = mapped[(mapped["mapped_facility_id"] != "") & (mapped[id_col] != "")]
        counts = mapped.groupby("mapped_facility_id")[id_col].nunique()

        df["provider_count"] = (
            df["Facility ID"].astype("string").map(counts).fillna(0).astype("int64")
        )

        nonzero = int((df["provider_count"] > 0).sum())
        print(f"Rows with provider_count > 0: {nonzero:,}")
        print("=" * 80)
        return df

# ------------------------- Minimal UX (mode-only) -----------------------
def get_mode_input() -> Optional[str]:
    print("\n" + "=" * 80)
    print("HGI WEBSITE MAPPING (Overpass / OSM)")
    print("=" * 80)
    print("\nProcessing Modes:")
    print(" 1. WEBSITE_MAPPING_OSM       - Map hospitals to websites using OSM/Nominatim (cached)")
    print(" 2. WEBSITE_MAPPING_OVERPASS  - Map hospitals to websites using Overpass cache (no API calls)")
    print(" 3. BOTH                      - Run Overpass then OSM")
    print(" 4. PROVIDER_COUNT            - Add provider_count using processed PPEF individuals")
    mode_in = input(
        "\nMode [1/2/3/4 or website_mapping_osm/website_mapping_overpass/both/provider_count]: "
    ).strip().lower()
    mode_map = {
        "1": "website_mapping_osm",
        "2": "website_mapping_overpass",
        "3": "both",
        "4": "provider_count",
    }
    mode = mode_map.get(mode_in, mode_in)
    valid = {"website_mapping_osm", "website_mapping_overpass", "both", "provider_count"}
    if mode not in valid:
        logger.error("Invalid mode: %s", mode_in)
        return None
    return mode

def main():
    print("\n" + "=" * 80)
    print("PROCESS HOSPITALS - UNIFIED WEBSITE MAPPING")
    print("=" * 80)
    print("\nArchitecture:")
    print(" ✓ Uses '../data' as base dir")
    print(" ✓ Auto-discovers latest '../data/parquet/hgi_*.parquet'")
    print(" ✓ Writes processed parquet + prints summaries")
    print("=" * 80)

    # Mode prompt (only prompt we keep)
    mode = get_mode_input()
    if mode is None:
        return

    # Fixed base dir (like hash_based_processing.py expectations)
    data_dir = Path("../data")
    processor = HospitalWebsiteProcessor(data_dir=data_dir)

    # Auto-discover latest HGI parquet
    data_path = processor.find_latest_hgi()
    print("\n" + "=" * 80)
    print("LOADING INPUT DATA")
    print("=" * 80)
    if not data_path or (not data_path.exists()):
        print(f"\nNo HGI parquet found at {data_dir}/parquet (expected hgi_*.parquet). Exiting.")
        return
    print(f"Using HGI file: {data_path}")

    df = pd.read_parquet(data_path)
    print(f"Loaded rows: {len(df):,}")

    expected = ["Facility ID", "Facility Name", "Address", "City/Town", "State", "ZIP Code"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        logger.warning("Missing expected HGI columns: %s", missing)

    date_str = processor.get_date_from_filename(data_path)
    out_path = processor.default_output_path(date_str)

    # Ensure output columns
    if mode in ("website_mapping_osm", "website_mapping_overpass", "both"):
        for c in ["system_url", "location_url", "system_url_source", "location_url_source", "matched_name", "match_confidence"]:
            if c not in df.columns:
                df[c] = None
    if mode == "provider_count" and "provider_count" not in df.columns:
        df["provider_count"] = 0

    # Defaults (no extra prompts)
    max_records = None
    offline_only = False
    min_importance = 0.25

    # Execute selected mode(s)
    if mode in ("website_mapping_overpass", "both"):
        df = processor.run_overpass(df, data_dir, date_str, max_records)
    if mode in ("website_mapping_osm", "both"):
        df = processor.run_osm(df, date_str, max_records, offline_only, min_importance)
    if mode == "provider_count":
        df = processor.run_provider_count(df)

    # Save
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved processed data: {out_path}")
    print(f" - Rows:    {len(df):,}")
    print(f" - Columns: {len(df.columns)}")
    print("\n✓ Processing complete!")
    print(f"Processed data saved to: {processor.processed_dir}")

if __name__ == "__main__":
    main()
