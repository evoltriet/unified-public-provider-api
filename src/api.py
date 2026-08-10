"""Flask API for the CMS provider baseline and PECOS organization rollups."""

from __future__ import annotations

import json
import logging
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from .data_store import STORE, bool_series, prepare_organization_search_columns, prepare_provider_search_columns
except ImportError:  # Supports `python src/api.py`.
    from data_store import STORE, bool_series, prepare_organization_search_columns, prepare_provider_search_columns


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MAX_RESULTS_DEFAULT = int(os.getenv("MAX_RESULTS_DEFAULT", "50"))
MAX_RESULTS_LIMIT = int(os.getenv("MAX_RESULTS_LIMIT", "500"))

TAXONOMY_CODES = {
    "cardiology": ["207RC0000X", "207RA0401X", "207RC0200X"],
    "family_medicine": ["207Q00000X", "207QA0401X", "207QA0000X"],
    "internal_medicine": ["207R00000X", "207RA0000X", "207RI0200X"],
    "pediatrics": ["208000000X", "2080P0216X", "2080A0000X"],
    "otolaryngology": ["207Y00000X", "207YX0901X", "207YP0228X", "207YS0012X"],
    "dermatology": ["207N00000X", "207ND0900X", "207NP0225X"],
    "emergency_medicine": ["207P00000X", "207PE0004X", "207PT0002X"],
    "orthopedic_surgery": ["207X00000X", "207XS0114X", "207XX0004X"],
    "psychiatry": ["208100000X", "2084P0800X", "2084N0400X"],
    "radiology": ["2085R0202X", "2085D0003X", "2085R0001X"],
    "anesthesiology": ["207L00000X", "207LA0401X", "207LC0200X"],
    "nurse_practitioner": ["363L00000X", "363LA2100X", "363LF0000X"],
    "physician_assistant": ["363A00000X"],
    "pharmacist": ["183500000X", "1835P1200X"],
    "physical_therapist": ["225100000X", "2251E1300X"],
    "psychologist": ["103T00000X", "103TC0700X"],
}


def _load_taxonomy_lookup() -> dict[str, str]:
    path = DATA_DIR / "taxonomy_lookup.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        LOGGER.warning("Could not load taxonomy lookup %s: %s", path, exc)
        return {}


TAXONOMY_LOOKUP = _load_taxonomy_lookup()


class RequestValidationError(ValueError):
    """Raised for invalid API query parameters."""


def _text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _number(value: Any, integer: bool = False) -> int | float | None:
    try:
        if value is None or pd.isna(value) or _text(value) == "":
            return None
        number = float(value)
        return int(number) if integer else round(number, 3)
    except (TypeError, ValueError):
        return None


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "t", "yes", "y"}


def _values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        text = _text(value)
        if not text:
            return []
        raw = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*(?=[A-Z0-9]{10}(?:\W|$))", text)
    result: list[str] = []
    for item in raw:
        cleaned = _text(item)
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _address(row: pd.Series, prefix: str) -> dict[str, str]:
    return {
        "address_1": _text(row.get(f"{prefix}_address_1")),
        "address_2": _text(row.get(f"{prefix}_address_2")),
        "city": _text(row.get(f"{prefix}_city")),
        "state": _text(row.get(f"{prefix}_state")),
        "postal_code": _text(row.get(f"{prefix}_zip5"))[:5],
        "phone": _text(row.get(f"{prefix}_phone")),
    }


def _taxonomy_codes(row: pd.Series) -> list[str]:
    codes = _values(row.get("taxonomy_codes_all"))
    primary = _text(row.get("taxonomy_code_primary"))
    if primary and primary not in codes:
        codes.insert(0, primary)
    return codes


def format_provider(row: pd.Series) -> dict[str, Any]:
    codes = _taxonomy_codes(row)
    descriptions = _values(row.get("taxonomy_descs_all"))
    primary_description = _text(row.get("taxonomy_desc_primary"))
    if primary_description and primary_description not in descriptions:
        descriptions.insert(0, primary_description)
    if not descriptions:
        descriptions = [TAXONOMY_LOOKUP.get(code, code) for code in codes]

    provider_address = _address(row, "provider_practice")
    mapped_address = _address(row, "mapped_practice")
    clinic_name = _text(row.get("mapped_clinic_name"))
    system_name = _text(row.get("mapped_system_name"))
    primary_specialty = (
        _text(row.get("specialty_normalized"))
        or primary_description
        or (descriptions[0] if descriptions else "")
    )

    return {
        "npi": _text(row.get("npi")),
        "provider_id": _text(row.get("provider_id")),
        "entity_type": "Individual",
        "provider_name": _text(row.get("provider_full_name")),
        "organization_name": clinic_name or system_name or None,
        "taxonomy_codes": codes,
        "primary_taxonomy": codes[0] if codes else None,
        "specialty_descriptions": descriptions,
        "primary_specialty": primary_specialty or None,
        "address": provider_address,
        "specialty": {
            "taxonomy_code": _optional_text(row.get("taxonomy_code_primary")),
            "taxonomy_description": primary_description or None,
            "normalized": _optional_text(row.get("specialty_normalized")),
            "group": _optional_text(row.get("specialty_group")),
            "provider_type": _optional_text(row.get("provider_type_group")),
            "all_taxonomy_codes": codes,
            "all_taxonomy_descriptions": descriptions,
        },
        "practice_location": provider_address,
        "affiliation": {
            "clinic": {
                "id": _optional_text(row.get("mapped_clinic_id")),
                "name": clinic_name or None,
            },
            "system": {
                "id": _optional_text(row.get("mapped_system_id")),
                "name": system_name or None,
            },
            "practice_location": mapped_address,
            "relationship": {
                "source": _optional_text(row.get("relationship_source")),
                "start_date": _optional_text(row.get("relationship_start_date")),
                "end_date": _optional_text(row.get("relationship_end_date")),
                "active": _boolean(row.get("is_active_relationship")),
            },
        },
        "currentness": {
            "active": _boolean(row.get("cms_current_active_flag")),
            "tier": _optional_text(row.get("cms_currentness_tier")),
            "reason": _optional_text(row.get("cms_currentness_reason")),
            "as_of_date": _optional_text(row.get("cms_active_as_of_date")),
        },
        "mapping": {
            "confidence": _optional_text(row.get("mapping_confidence")),
            "confidence_tier": _optional_text(row.get("mapping_confidence_tier")),
            "method": _optional_text(row.get("mapping_method")),
            "score": _number(row.get("mapping_score")),
            "margin_score": _number(row.get("mapping_margin_score")),
            "signal_count": _number(row.get("mapping_signal_count"), integer=True),
            "tied": _boolean(row.get("primary_is_tied")),
            "ambiguous": _boolean(row.get("primary_is_ambiguous")),
        },
        "provenance": {
            "ppef_snapshot_date": _optional_text(row.get("ppef_snapshot_date")),
            "pecos_snapshot_date": _optional_text(row.get("pecos_snapshot_date")),
            "npi_registry_snapshot_date": _optional_text(row.get("npi_registry_snapshot_date")),
            "identity_source": _optional_text(row.get("provider_identity_source")),
            "specialty_source": _optional_text(row.get("specialty_source")),
            "phone_source": _optional_text(row.get("phone_source")),
            "address_source": _optional_text(row.get("address_source")),
            "clinic_source": _optional_text(row.get("clinic_source")),
            "system_source": _optional_text(row.get("system_source")),
        },
    }


def format_organization(row: pd.Series, kind: str) -> dict[str, Any]:
    count_column = "provider_count" if kind == "clinic" else "provider_count_entity"
    record = {
        "entity_type": kind,
        "id": _text(row.get("_entity_id")),
        "name": _text(row.get("_display_name")),
        "organization_name": _text(row.get("_display_name")),
        "org_npi": _optional_text(row.get("org_npi")),
        "org_entity_id": _optional_text(row.get("org_entity_id")),
        "address": _address(row, "practice"),
        "is_hospital": _boolean(row.get("is_hospital")),
        "provider_count": _number(row.get(count_column), integer=True),
        "website": _optional_text(row.get("website")),
        "website_source": _optional_text(row.get("website_source")),
        "website_confidence": _optional_text(row.get("website_confidence")),
        "quality_warning": _optional_text(row.get("rollup_quality_warning")),
    }
    if kind == "clinic":
        record["clinic_id"] = _text(row.get("clinic_id"))
        record["system"] = {
            "id": _optional_text(row.get("system_id")),
            "name": _text(row.get("system_display_name"))
            or _optional_text(row.get("system_name")),
        }
    else:
        record["system_id"] = _text(row.get("system_id"))
    return record


def _parse_limit() -> int:
    raw = request.args.get("limit", str(MAX_RESULTS_DEFAULT)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RequestValidationError("limit must be an integer") from exc
    if value < 1:
        raise RequestValidationError("limit must be at least 1")
    return min(value, MAX_RESULTS_LIMIT)


def _literal_contains(series: pd.Series, query: str) -> pd.Series:
    return series.astype("string").fillna("").str.contains(query.upper(), regex=False, na=False)


def _response(results: pd.DataFrame, formatter, search_params: dict[str, Any]):
    records = [formatter(row) for _, row in results.iterrows()]
    return jsonify({"count": len(records), "search_params": search_params, "results": records})


def require_providers(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not STORE.providers_loaded:
            return jsonify({"error": "Data not loaded", "message": "Provider data is unavailable."}), 503
        return func(*args, **kwargs)

    return wrapped


def require_organizations(kind: str):
    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            loaded = STORE.clinics_loaded if kind == "clinic" else STORE.systems_loaded
            if not loaded:
                return (
                    jsonify({"error": "Data not loaded", "message": f"{kind.title()} data is unavailable."}),
                    503,
                )
            return func(*args, **kwargs)

        return wrapped

    return decorator


def initialize_data() -> bool:
    """Load the configured API source into the process-local store."""
    return STORE.load(DATA_DIR)


def load_npi_data() -> bool:
    """Backward-compatible loader name used by older deployments."""
    return initialize_data()


def load_hospitals_data() -> bool:
    """Backward-compatible organization loader status."""
    if not STORE.providers_loaded:
        initialize_data()
    return STORE.systems_loaded


@app.errorhandler(RequestValidationError)
def invalid_request(error):
    return jsonify({"error": "Invalid request", "message": str(error)}), 400


@app.route("/api/health", methods=["GET"])
def health_check():
    providers = len(STORE.providers) if STORE.providers_loaded else 0
    clinics = len(STORE.clinics) if STORE.clinics_loaded else 0
    systems = len(STORE.systems) if STORE.systems_loaded else 0
    return jsonify(
        {
            "status": "healthy" if STORE.providers_loaded else "degraded",
            "data_loaded": STORE.providers_loaded,
            "data_source": STORE.source or None,
            "baseline_view": STORE.baseline_view or None,
            "providers_data_loaded": STORE.providers_loaded,
            "providers_data_source": STORE.source or None,
            "providers_data_file": str(STORE.provider_path) if STORE.provider_path else None,
            "total_providers": providers,
            "clinics_data_loaded": STORE.clinics_loaded,
            "clinics_data_file": str(STORE.clinic_path) if STORE.clinic_path else None,
            "total_clinics": clinics,
            "systems_data_loaded": STORE.systems_loaded,
            "systems_data_file": str(STORE.system_path) if STORE.system_path else None,
            "total_systems": systems,
            "hospitals_data_loaded": STORE.systems_loaded,
            "hospitals_data_source": STORE.source or None,
            "total_hospitals": int(STORE.systems["is_hospital"].sum()) if STORE.systems_loaded else 0,
            "snapshots": STORE.metadata,
            "taxonomy_lookup_loaded": bool(TAXONOMY_LOOKUP),
            "taxonomy_descriptions_count": len(TAXONOMY_LOOKUP),
        }
    )


@app.route("/api/taxonomy/codes", methods=["GET"])
@require_providers
def get_taxonomy_codes():
    providers = STORE.providers
    specialties = sorted(
        value for value in providers["specialty_normalized"].dropna().astype(str).unique() if value
    )
    groups = sorted(value for value in providers["specialty_group"].dropna().astype(str).unique() if value)
    return jsonify(
        {
            "taxonomy_mappings": TAXONOMY_CODES,
            "normalized_specialties": specialties,
            "specialty_groups": groups,
            "taxonomy_descriptions_loaded": bool(TAXONOMY_LOOKUP),
        }
    )


def _location_mask(
    df: pd.DataFrame,
    city: str = "",
    state: str = "",
    postal_code: str = "",
    source: str = "any",
) -> pd.Series:
    if source not in {"any", "provider", "affiliation"}:
        raise RequestValidationError("location_source must be any, provider, or affiliation")
    masks: list[pd.Series] = []
    if source in {"any", "provider"}:
        provider_mask = pd.Series(True, index=df.index)
        if city:
            provider_mask &= df["_provider_city"].eq(city.upper())
        if state:
            provider_mask &= df["provider_practice_state"].eq(state.upper())
        if postal_code:
            provider_mask &= df["provider_practice_zip5"].eq(postal_code)
        masks.append(provider_mask)
    if source in {"any", "affiliation"}:
        affiliation_mask = pd.Series(True, index=df.index)
        if city:
            affiliation_mask &= df["_mapped_city"].eq(city.upper())
        if state:
            affiliation_mask &= df["mapped_practice_state"].eq(state.upper())
        if postal_code:
            affiliation_mask &= df["mapped_practice_zip5"].eq(postal_code)
        masks.append(affiliation_mask)
    result = masks[0]
    for mask in masks[1:]:
        result |= mask
    return result


@app.route("/api/providers/<npi>", methods=["GET"])
@require_providers
def get_provider_by_npi(npi: str):
    npi = npi.strip()
    if not re.fullmatch(r"\d{10}", npi):
        raise RequestValidationError("npi must contain exactly 10 digits")
    rows = STORE.providers[STORE.providers["npi"].eq(npi)]
    if rows.empty:
        return jsonify({"error": "Provider not found", "message": f"No provider found with NPI {npi}"}), 404
    return jsonify(format_provider(rows.iloc[0]))


@app.route("/api/providers/search/specialty", methods=["GET"])
@require_providers
def search_by_specialty():
    specialty = request.args.get("specialty", "").strip()
    if not specialty:
        raise RequestValidationError("specialty parameter is required")
    state = request.args.get("state", "").strip().upper()
    limit = _parse_limit()
    normalized_keyword = specialty.lower().replace(" ", "_")
    terms = TAXONOMY_CODES.get(normalized_keyword, [specialty])
    mask = pd.Series(False, index=STORE.providers.index)
    for term in terms:
        mask |= _literal_contains(STORE.providers["_search_specialty"], term)
    if state:
        mask &= _location_mask(STORE.providers, state=state, source="any")
    results = STORE.providers[mask].head(limit)
    return _response(
        results,
        format_provider,
        {"specialty": specialty, "state": state or None, "limit": limit},
    )


@app.route("/api/providers/search/location", methods=["GET"])
@require_providers
def search_by_location():
    city = request.args.get("city", "").strip()
    state = request.args.get("state", "").strip().upper()
    specialty = request.args.get("specialty", "").strip()
    location_source = request.args.get("location_source", "any").strip().lower() or "any"
    if not city or not state:
        raise RequestValidationError("both city and state are required")
    limit = _parse_limit()
    mask = _location_mask(STORE.providers, city=city, state=state, source=location_source)
    if specialty:
        mask &= _literal_contains(STORE.providers["_search_specialty"], specialty)
    results = STORE.providers[mask].head(limit)
    return _response(
        results,
        format_provider,
        {
            "city": city,
            "state": state,
            "specialty": specialty or None,
            "location_source": location_source,
            "limit": limit,
        },
    )


@app.route("/api/providers/search/state/<state_code>", methods=["GET"])
@require_providers
def search_by_state(state_code: str):
    state = state_code.strip().upper()
    if not re.fullmatch(r"[A-Z]{2}", state):
        raise RequestValidationError("state must be a two-letter code")
    limit = _parse_limit()
    mask = _location_mask(STORE.providers, state=state, source="any")
    return _response(STORE.providers[mask].head(limit), format_provider, {"state": state, "limit": limit})


@app.route("/api/providers/search/postal_code/<postal_code>", methods=["GET"])
@require_providers
def search_by_postal_code(postal_code: str):
    postal_code = postal_code.strip()[:5]
    if not re.fullmatch(r"\d{5}", postal_code):
        raise RequestValidationError("postal_code must contain five digits")
    limit = _parse_limit()
    mask = _location_mask(STORE.providers, postal_code=postal_code, source="any")
    return _response(
        STORE.providers[mask].head(limit),
        format_provider,
        {"postal_code": postal_code, "limit": limit},
    )


@app.route("/api/providers/search/hospital", methods=["GET"])
@require_providers
def search_providers_by_hospital():
    hospital = request.args.get("hospital", "").strip()
    if not hospital:
        raise RequestValidationError("hospital parameter is required")
    state = request.args.get("state", "").strip().upper()
    limit = _parse_limit()
    mask = _literal_contains(STORE.providers["_search_affiliation"], hospital)
    if state:
        mask &= _location_mask(STORE.providers, state=state, source="any")
    return _response(
        STORE.providers[mask].head(limit),
        format_provider,
        {"hospital": hospital, "state": state or None, "limit": limit},
    )


@app.route("/api/providers/search/name", methods=["GET"])
@require_providers
def search_providers_by_name():
    name = request.args.get("name", "").strip()
    if not name:
        raise RequestValidationError("name parameter is required")
    city = request.args.get("city", "").strip()
    state = request.args.get("state", "").strip().upper()
    postal_code = request.args.get("postal_code", "").strip()[:5]
    hospital = request.args.get("hospital", "").strip()
    location_source = request.args.get("location_source", "any").strip().lower() or "any"
    limit = _parse_limit()
    mask = _literal_contains(STORE.providers["_search_name"], name)
    if city or state or postal_code:
        mask &= _location_mask(
            STORE.providers,
            city=city,
            state=state,
            postal_code=postal_code,
            source=location_source,
        )
    if hospital:
        mask &= _literal_contains(STORE.providers["_search_affiliation"], hospital)
    return _response(
        STORE.providers[mask].head(limit),
        format_provider,
        {
            "name": name,
            "city": city or None,
            "state": state or None,
            "postal_code": postal_code or None,
            "hospital": hospital or None,
            "location_source": location_source,
            "limit": limit,
        },
    )


def _organization_frame(kind: str, hospitals_only: bool = False) -> pd.DataFrame:
    frame = STORE.clinics if kind == "clinic" else STORE.systems
    if frame is None:
        return pd.DataFrame()
    return frame[frame["is_hospital"]].copy() if hospitals_only else frame


def _organization_name_search(kind: str, hospitals_only: bool = False):
    parameter = "hospital" if hospitals_only else kind
    name = request.args.get(parameter, request.args.get("name", "")).strip()
    if not name:
        raise RequestValidationError(f"{parameter} parameter is required")
    city = request.args.get("city", "").strip()
    state = request.args.get("state", "").strip().upper()
    postal_code = request.args.get("postal_code", "").strip()[:5]
    limit = _parse_limit()
    frame = _organization_frame(kind, hospitals_only)
    mask = _literal_contains(frame["_search_name"], name)
    if city:
        mask &= frame["_search_city"].eq(city.upper())
    if state:
        mask &= frame["practice_state"].eq(state)
    if postal_code:
        mask &= frame["practice_zip5"].eq(postal_code)
    formatter = lambda row: format_organization(row, kind)
    return _response(
        frame[mask].head(limit),
        formatter,
        {
            parameter: name,
            "city": city or None,
            "state": state or None,
            "postal_code": postal_code or None,
            "limit": limit,
        },
    )


def _organization_location_search(kind: str, hospitals_only: bool = False):
    city = request.args.get("city", "").strip()
    state = request.args.get("state", "").strip().upper()
    postal_code = request.args.get("postal_code", "").strip()[:5]
    address = request.args.get("address", "").strip()
    if not postal_code and not address and not (city and state):
        raise RequestValidationError("provide city+state, postal_code, or address")
    limit = _parse_limit()
    frame = _organization_frame(kind, hospitals_only)
    mask = pd.Series(True, index=frame.index)
    if city:
        mask &= frame["_search_city"].eq(city.upper())
    if state:
        mask &= frame["practice_state"].eq(state)
    if postal_code:
        mask &= frame["practice_zip5"].eq(postal_code)
    if address:
        mask &= _literal_contains(frame["_search_address"], address)
    formatter = lambda row: format_organization(row, kind)
    return _response(
        frame[mask].head(limit),
        formatter,
        {
            "city": city or None,
            "state": state or None,
            "postal_code": postal_code or None,
            "address": address or None,
            "limit": limit,
        },
    )


@app.route("/api/clinics/search/name", methods=["GET"])
@require_organizations("clinic")
def search_clinics_by_name():
    return _organization_name_search("clinic")


@app.route("/api/clinics/search/location", methods=["GET"])
@require_organizations("clinic")
def search_clinics_by_location():
    return _organization_location_search("clinic")


@app.route("/api/systems/search/name", methods=["GET"])
@require_organizations("system")
def search_systems_by_name():
    return _organization_name_search("system")


@app.route("/api/systems/search/location", methods=["GET"])
@require_organizations("system")
def search_systems_by_location():
    return _organization_location_search("system")


@app.route("/api/hospitals/search/name", methods=["GET"])
@require_organizations("system")
def search_hospitals_by_name():
    return _organization_name_search("system", hospitals_only=True)


@app.route("/api/hospitals/search/location", methods=["GET"])
@require_organizations("system")
def search_hospitals_by_location():
    return _organization_location_search("system", hospitals_only=True)


@app.route("/", methods=["GET"])
def index():
    return """
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Unified CMS Provider API</title></head>
<body>
  <main>
    <h1>Unified CMS Provider API</h1>
    <p>Search current-active CMS providers, clinics, health systems, and hospitals.</p>
    <ul>
      <li><code>GET /api/health</code></li>
      <li><code>GET /api/providers/&lt;npi&gt;</code></li>
      <li><code>GET /api/providers/search/name</code></li>
      <li><code>GET /api/providers/search/specialty</code></li>
      <li><code>GET /api/providers/search/location</code></li>
      <li><code>GET /api/clinics/search/name</code></li>
      <li><code>GET /api/systems/search/name</code></li>
      <li><code>GET /api/hospitals/search/name</code></li>
    </ul>
    <p>See <code>src/API_REFERENCE.md</code> for the full endpoint contract.</p>
  </main>
</body>
</html>
"""


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "message": "The requested endpoint does not exist"}), 404


@app.errorhandler(500)
def internal_error(error):
    LOGGER.exception("Unhandled API error: %s", error)
    return jsonify({"error": "Internal server error", "message": "An unexpected error occurred"}), 500


if __name__ == "__main__":
    loaded = initialize_data()
    if not loaded:
        LOGGER.error("No provider data loaded; data endpoints will return 503")
    app.run(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "5000")),
        debug=os.getenv("FLASK_ENV") == "development",
    )
