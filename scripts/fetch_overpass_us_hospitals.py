"""Fetch all US hospitals from OpenStreetMap via Overpass API.

Usage:
    cd scripts
    python fetch_overpass_us_hospitals.py

Output:
    ../data/overpass_hospitals_us_<YYYYMMDD>.json
    ../data/overpass_hospitals_us_latest.json

Notes:
    - This script fetches OSM hospital POIs (nodes/ways/relations) tagged as amenity=hospital
      or healthcare=hospital, and requests their tags plus center coordinates.
    - Designed to run periodically (e.g., weekly) and feed the website_mapping Overpass fallback.
"""

import json
import time
import random
from pathlib import Path

import requests


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def build_query() -> str:
    # Fetch all hospitals in the US (nodes/ways/relations) tagged as amenity=hospital OR healthcare=hospital.
    # Use out center tags so that ways/relations include a representative center point.
    return """
[out:json][timeout:1800];
area["ISO3166-1"="US"][admin_level=2]->.usa;
(
  nwr["amenity"="hospital"](area.usa);
  nwr["healthcare"="hospital"](area.usa);
);
out center tags;
""".strip()


def post_with_retry(url: str, data: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries + 1):
        # jitter to avoid synchronized load
        time.sleep(0.5 + random.random() * 0.5)
        try:
            r = requests.post(url, data={"data": data}, timeout=300)
            if r.status_code == 200:
                return r.json()

            # Overpass sometimes returns 429/504 on load
            if r.status_code in {429, 500, 502, 503, 504}:
                backoff = (2 ** attempt) * (1.0 + random.random())
                print(f"Overpass status {r.status_code} from {url}. Retry in {backoff:.1f}s")
                time.sleep(backoff)
                continue

            raise RuntimeError(f"Overpass request failed: HTTP {r.status_code}: {r.text[:200]}")

        except Exception as e:
            backoff = (2 ** attempt) * (1.0 + random.random())
            print(f"Overpass error from {url}: {e}. Retry in {backoff:.1f}s")
            time.sleep(backoff)

    raise RuntimeError("Exceeded Overpass retries")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    query = build_query()
    last_err = None

    payload = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"Fetching Overpass hospitals from: {endpoint}")
            payload = post_with_retry(endpoint, query)
            break
        except Exception as e:
            last_err = e
            print(f"Failed endpoint {endpoint}: {e}")

    if payload is None:
        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}")

    # Add metadata
    run_date = time.strftime("%Y%m%d")
    out = {
        "metadata": {
            "run_date": run_date,
            "source": "overpass",
            "query": query,
            "element_count": len(payload.get("elements") or []),
        },
        **payload,
    }

    dated_path = data_dir / f"overpass_hospitals_us_{run_date}.json"
    latest_path = data_dir / "overpass_hospitals_us_latest.json"

    # Write dated
    with open(dated_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    # Write latest
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote: {dated_path}")
    print(f"Wrote: {latest_path}")
    print(f"Elements: {out['metadata']['element_count']}")


if __name__ == "__main__":
    main()
