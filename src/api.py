 
""" 
NPI Registry Flask API - Smart Loading with Entity Mapping 
A RESTful API for searching and querying healthcare provider data from the 
National Provider Identifier (NPI) Registry. 
Smart Loading Strategy (UPDATED): 
- Providers (individuals): 
 ✓ Loads processed individuals from /processed_data/npi_individuals_processed_*.parquet 
 → Falls back to /parquet/npi_individuals_*.parquet 
- Hospitals (organizations): 
 ✓ Loads processed organizations from /processed_data/npi_organizations_processed_*.parquet 
 (also supports npi_origanizations_processed_* as a safety net) 
 → Falls back to /parquet/npi_organizations_*.parquet 
Features: 
- Searches ALL 15 taxonomy code columns (not just Code_1) 
- Supports keyword search (e.g., "otolaryngology") 
- Returns all taxonomy codes + descriptions per provider 
- Clean separation: providers vs hospitals dataframes 
Optimizations: 
1. Vectorized taxonomy search (10-100x faster) 
2. Load only necessary columns from parquet 
3. Use exact match instead of regex where possible 
4. Better memory management 
""" 
from flask import Flask, request, jsonify 
from flask_cors import CORS 
import pandas as pd 
from pathlib import Path 
from typing import Optional, Dict, List, Any 
from math import radians, cos, sin, asin, sqrt 
import logging 
from functools import wraps 
import os 
import json 
from dotenv import load_dotenv 

# Load environment variables 
load_dotenv() 

# Configure logging 
logging.basicConfig( 
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
) 
logger = logging.getLogger(__name__) 

# Initialize Flask app 
app = Flask(__name__) 
CORS(app) 

# Configuration 
DATA_DIR = Path(os.getenv("DATA_DIR", "data")) 
PROCESSED_DATA_DIR = DATA_DIR / "processed_data" 
PARQUET_DIR = DATA_DIR / "parquet" 
MAX_RESULTS_DEFAULT = int(os.getenv("MAX_RESULTS_DEFAULT", "50")) 
MAX_RESULTS_LIMIT = int(os.getenv("MAX_RESULTS_LIMIT", "500")) 

# Global data storage 
# Providers (individuals) 
npi_data: Optional[pd.DataFrame] = None 
data_file_path: Optional[Path] = None 
data_source: Optional[str] = ( 
    None  # Track source: 'processed_individuals', 'individuals' 
) 

# Hospitals (organizations) 
hospitals_data: Optional[pd.DataFrame] = None 
hospitals_file_path: Optional[Path] = None 
hospitals_source: Optional[str] = None  # 'processed_orgs' or 'organizations' 

# Common taxonomy code mappings for keyword search 
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
    "family_np": ["363LF0000X"], 
    "acute_care_np": ["363LA2100X"], 
    "physician_assistant": ["363A00000X"], 
    "pharmacist": ["183500000X", "1835P1200X"], 
    "physical_therapist": ["225100000X", "2251E1300X"], 
    "psychologist": ["103T00000X", "103TC0700X"], 
} 

# Load taxonomy lookup if available 
TAXONOMY_LOOKUP = {} 
TAXONOMY_LOOKUP_FILE = DATA_DIR / "taxonomy_lookup.json" 
if TAXONOMY_LOOKUP_FILE.exists(): 
    try: 
        with open(TAXONOMY_LOOKUP_FILE, "r") as f: 
            TAXONOMY_LOOKUP = json.load(f) 
        logger.info(f"Loaded {len(TAXONOMY_LOOKUP)} taxonomy descriptions") 
    except Exception as e: 
        logger.warning(f"Could not load taxonomy lookup: {e}") 


def _read_parquet_columns(path: Path, columns: List[str]) -> pd.DataFrame: 
    """ 
    Best-effort column projection: try reading only requested columns; if that fails, 
    read all and then project available columns. 
    """ 
    try: 
        return pd.read_parquet(path, columns=columns) 
    except Exception: 
        df = pd.read_parquet(path) 
        keep = [c for c in columns if c in df.columns] 
        return df[keep] 


def search_across_taxonomy_codes(query: str) -> pd.Series: 
    """ 
    OPTIMIZED: Search across all 15 taxonomy columns using vectorized operations. 
    Args: 
        query: Taxonomy code (e.g., "207Y00000X") or keyword (e.g., "otolaryngology") 
    Returns: 
        Boolean series for matching records 
    """ 
    # Check if query is a known specialty keyword 
    query_lower = query.lower().replace(" ", "_") 
    if query_lower in TAXONOMY_CODES: 
        codes = TAXONOMY_CODES[query_lower] 
        _ = query.replace("\r", "").replace("\n", "") 
    else: 
        codes = [query] 

    # VECTORIZED: Use pandas isin() with concatenated columns 
    tax_cols = [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)] 
    tax_cols_exist = [col for col in tax_cols if col in npi_data.columns] 
    if not tax_cols_exist: 
        logger.warning("No taxonomy columns found!") 
        return pd.Series([False] * len(npi_data), index=npi_data.index) 

    stacked = npi_data[tax_cols_exist].stack() 
    matched = stacked.isin(codes) 
    mask = matched.groupby(level=0).any() 
    mask = mask.reindex(npi_data.index, fill_value=False) 
    return mask 


def load_npi_data(): 
    """ 
    Load PROVIDERS (individuals) with smart fallback strategy: 
    1) Try npi_individuals_processed_*.parquet from /processed_data (mapped individuals + geocoding) 
    2) Fall back to npi_individuals_*.parquet from /parquet (raw individuals) 
    """ 
    global npi_data, data_file_path, data_source 
    logger.info("=" * 80) 
    logger.info("LOADING NPI REGISTRY DATA (INDIVIDUALS)") 
    logger.info("=" * 80) 

    essential_cols = [ 
        "NPI", 
        "Entity Type Code", 
        "Provider Organization Name (Legal Business Name)", 
        "Provider First Name", 
        "Provider Last Name (Legal Name)", 
        "Provider Credential Text", 
        "Provider Business Practice Location Address City Name", 
        "Provider Business Practice Location Address State Name", 
        "Provider Business Practice Location Address Postal Code", 
        "Provider Business Practice Location Address Telephone Number", 
        "Provider First Line Business Practice Location Address", 
        "Provider Second Line Business Practice Location Address", 
        "mapped_org_npi", 
        "mapped_org_name", 
    ] + [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)] 

    # STRATEGY 1: Try processed individuals first 
    if PROCESSED_DATA_DIR.exists(): 
        processed_files = sorted(PROCESSED_DATA_DIR.glob("npi_individuals_processed_*.parquet")) 
        if processed_files: 
            data_file_path = processed_files[-1] 
            logger.info(f"✓ Found processed individuals: {data_file_path.name}") 
            try: 
                npi_data = _read_parquet_columns(data_file_path, essential_cols) 
                logger.info(f"✓ Loaded {len(npi_data):,} providers from PROCESSED INDIVIDUALS") 
                logger.info( 
                    f" Memory: {npi_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB" 
                ) 
                data_source = "processed_individuals" 
                logger.info("=" * 80) 
                return True 
            except Exception as e: 
                logger.warning(f"Failed to load processed individuals: {e}") 

    # STRATEGY 2: Fall back to raw individuals from parquet 
    if PARQUET_DIR.exists(): 
        individuals_files = sorted(PARQUET_DIR.glob("npi_individuals_*.parquet")) 
        if individuals_files: 
            data_file_path = individuals_files[-1] 
            logger.info(f"✓ Found raw individuals: {data_file_path.name}") 
            try: 
                npi_data = _read_parquet_columns(data_file_path, essential_cols) 
                logger.info( 
                    f"✓ Loaded {len(npi_data):,} providers from RAW INDIVIDUALS" 
                ) 
                logger.info( 
                    f" Memory: {npi_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB" 
                ) 
                data_source = "individuals" 
                logger.info("=" * 80) 
                return True 
            except Exception as e: 
                logger.warning(f"Failed to load individuals: {e}") 

    logger.error("✗ No individuals data found!") 
    logger.error(f" Searched in: {PROCESSED_DATA_DIR}, {PARQUET_DIR}") 
    logger.error(" Please run data download/processing scripts first") 
    logger.info("=" * 80) 
    return False 


def load_hospitals_data() -> bool: 
    """ 
    Load HOSPITALS (organizations) with fallback strategy: 
    1) Try processed org parquet from /processed_data 
       - npi_organizations_processed_*.parquet 
       - npi_origanizations_processed_* (accept misspelling) 
    2) Fall back to raw organizations parquet from /parquet (npi_organizations_*.parquet) 
    """ 
    global hospitals_data, hospitals_file_path, hospitals_source 
    logger.info("=" * 80) 
    logger.info("LOADING ORGANIZATIONS (HOSPITALS) DATA") 
    logger.info("=" * 80) 

    essential_cols = [ 
        "NPI", 
        "Entity Type Code", 
        "Provider Organization Name (Legal Business Name)", 
        "Provider First Line Business Practice Location Address", 
        "Provider Second Line Business Practice Location Address", 
        "Provider Business Practice Location Address City Name", 
        "Provider Business Practice Location Address State Name", 
        "Provider Business Practice Location Address Postal Code", 
        "Provider Business Practice Location Address Telephone Number", 
        "is_hospital", 
        "system_homepage_url", 
        "location_page_url", 
        "org_entity_id", 
        "provider_count", 
        "provider_count_entity", 
        # processed enrichments if present 
        "latitude", "longitude", "geohash", 
    ] 

    try: 
        # Strategy 1: processed organizations in /processed_data 
        processed_patterns = [ 
            "npi_organizations_processed_*.parquet", 
            "npi_origanizations_processed_*.parquet",  # misspelling tolerance 
        ] 
        processed_files = [] 
        if PROCESSED_DATA_DIR.exists(): 
            for pat in processed_patterns: 
                processed_files.extend(sorted(PROCESSED_DATA_DIR.glob(pat))) 
        if processed_files: 
            hospitals_file_path = processed_files[-1] 
            logger.info(f"✓ Found processed organizations: {hospitals_file_path.name}") 
            hospitals_data = _read_parquet_columns(hospitals_file_path, essential_cols) 
            hospitals_source = "processed_orgs" 
            # Keep only organizations (Entity Type Code == 2) if column exists 
            if "Entity Type Code" in hospitals_data.columns: 
                with pd.option_context('mode.use_inf_as_na', True): 
                    hospitals_data = hospitals_data[ 
                        pd.to_numeric(hospitals_data["Entity Type Code"], errors="coerce") == 2 
                    ] 
            logger.info(f"✓ Loaded {len(hospitals_data):,} organizations from PROCESSED ORGS") 
            logger.info(f" Memory: {hospitals_data.memory_usage(deep=True).sum()/1024**2:.1f} MB") 
            logger.info("=" * 80) 
            return True 

        # Strategy 2: raw organizations in /parquet 
        if PARQUET_DIR.exists(): 
            org_files = sorted(PARQUET_DIR.glob("npi_organizations_*.parquet")) 
            if org_files: 
                hospitals_file_path = org_files[-1] 
                logger.info(f"✓ Found raw organizations: {hospitals_file_path.name}") 
                hospitals_data = _read_parquet_columns(hospitals_file_path, essential_cols) 
                hospitals_source = "organizations" 
                if "Entity Type Code" in hospitals_data.columns: 
                    with pd.option_context('mode.use_inf_as_na', True): 
                        hospitals_data = hospitals_data[ 
                            pd.to_numeric(hospitals_data["Entity Type Code"], errors="coerce") == 2 
                        ] 
                logger.info(f"✓ Loaded {len(hospitals_data):,} organizations from RAW ORGS") 
                logger.info(f" Memory: {hospitals_data.memory_usage(deep=True).sum()/1024**2:.1f} MB") 
                logger.info("=" * 80) 
                return True 

        logger.error("✗ No organizations data found!") 
        logger.error(f" Searched in: {PROCESSED_DATA_DIR}, {PARQUET_DIR}") 
        logger.error(" Please ensure processed org parquet exists or raw organizations parquet is available.") 
        logger.info("=" * 80) 
        return False 
    except Exception as e: 
        logger.error(f"Failed to load hospitals data: {e}") 
        return False 


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float: 
    """Calculate the great circle distance between two points on earth in miles.""" 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2]) 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2 
    c = 2 * asin(sqrt(a)) 
    r = 3956  # Radius of earth in miles 
    return c * r 


def format_provider(row: pd.Series, include_distance: bool = False) -> Dict[str, Any]: 
    """Format a provider record for API response.""" 
    # Determine if organization or individual based on Entity Type Code 
    # 1 = Individual, 2 = Organization 
    entity_type_code = row.get("Entity Type Code") 
    is_org = False 
    if pd.notna(entity_type_code): 
        try: 
            code = int(float(entity_type_code)) 
            is_org = code == 2 
        except (ValueError, TypeError): 
            org_name = row.get("Provider Organization Name (Legal Business Name)") 
            is_org = ( 
                pd.notna(org_name) 
                and str(org_name).strip() 
                and str(org_name).lower() != "nan" 
            ) 

    # Always extract organization_name from mapped_org_name when available, fallback to legal business name 
    mapped_org_name_val = row.get("mapped_org_name") 
    if pd.notna(mapped_org_name_val) and str(mapped_org_name_val).strip() and str(mapped_org_name_val).lower() != 'nan': 
        organization_name = str(mapped_org_name_val).strip() 
    else: 
        organization_name = row.get("Provider Organization Name (Legal Business Name)") 
        organization_name = ( 
            str(organization_name).strip() 
            if pd.notna(organization_name) and str(organization_name).lower() != 'nan' 
            else None 
        ) 

    # Always extract provider_name from the provider name fields 
    first = row.get("Provider First Name", "") 
    last = row.get("Provider Last Name (Legal Name)", "") 
    credential = row.get("Provider Credential Text", "") 
    provider_name = f"{first} {last}".strip() 
    if credential and str(credential).lower() != "nan": 
        provider_name += f", {credential}" 

    # Extract address components 
    address_line1 = row.get( 
        "Provider First Line Business Practice Location Address", "" 
    ) 
    address_line2 = row.get( 
        "Provider Second Line Business Practice Location Address", "" 
    ) 
    city = row.get("Provider Business Practice Location Address City Name", "") 
    state = row.get("Provider Business Practice Location Address State Name", "") 
    postal_code = row.get("Provider Business Practice Location Address Postal Code", "") 
    phone = row.get("Provider Business Practice Location Address Telephone Number", "") 

    # Get all taxonomy codes 
    taxonomy_codes = [] 
    for i in range(1, 16): 
        code = row.get(f"Healthcare Provider Taxonomy Code_{i}") 
        if pd.notna(code) and str(code).lower() != "nan": 
            taxonomy_codes.append(str(code).strip()) 

    # Get specialty descriptions if lookup is available 
    specialty_descriptions = [] 
    if TAXONOMY_LOOKUP: 
        for code in taxonomy_codes: 
            desc = TAXONOMY_LOOKUP.get(code, code) 
            specialty_descriptions.append(desc) 

    result = { 
        "npi": str(row.get("NPI", "")), 
        "entity_type": "Organization" if is_org else "Individual", 
        "provider_name": provider_name, 
        "organization_name": organization_name,  # ALWAYS included 
        "taxonomy_codes": taxonomy_codes, 
        "primary_taxonomy": taxonomy_codes[0] if taxonomy_codes else None, 
        "address": { 
            "address_1": str(address_line1).strip() if address_line1 else "", 
            "address_2": str(address_line2).strip() if address_line2 else "", 
            "city": str(city).strip() if city else "", 
            "state": str(state).strip() if state else "", 
            "postal_code": str(postal_code)[:5].strip() if postal_code else "", 
            "phone": str(phone).strip() if phone else "", 
        }, 
    } 

    if specialty_descriptions: 
        result["specialty_descriptions"] = specialty_descriptions 
        result["primary_specialty"] = specialty_descriptions[0] 

    # Include geocoding if available 
    if "latitude" in row and pd.notna(row.get("latitude")):
        try:
            result["geocode"] = {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
        except (ValueError, TypeError):
            pass

    # Include processed data columns if available 
    if "geohash" in row and pd.notna(row.get("geohash")):
        result["geohash"] = str(row["geohash"]).strip()

    if (
        include_distance
        and "distance_miles" in row
        and pd.notna(row.get("distance_miles"))
    ):
        result["distance_miles"] = round(float(row["distance_miles"]), 2)

    return result 


def require_data(f): 
    """Decorator to ensure PROVIDERS data is loaded before processing request.""" 
    @wraps(f) 
    def decorated_function(*args, **kwargs): 
        if npi_data is None: 
            return ( 
                jsonify( 
                    { 
                        "error": "Data not loaded", 
                        "message": "NPI data has not been loaded. Please contact administrator.", 
                    } 
                ), 
                503, 
            ) 
        return f(*args, **kwargs) 

    return decorated_function 


def require_hospital_data(f): 
    """Decorator to ensure HOSPITALS (organizations) data is loaded before processing request.""" 
    @wraps(f) 
    def decorated_function(*args, **kwargs): 
        if hospitals_data is None or len(hospitals_data) == 0: 
            return ( 
                jsonify({ 
                    "error": "Hospitals data not loaded", 
                    "message": "Organizations dataset is not available. Please contact administrator.", 
                }), 
                503, 
            ) 
        return f(*args, **kwargs) 

    return decorated_function 

# ==================== API ENDPOINTS ==================== 
@app.route("/api/health", methods=["GET"]) 
def health_check(): 
    """Health check endpoint.""" 
    return jsonify( 
        { 
            "status": "healthy", 
            "providers_data_loaded": npi_data is not None, 
            "providers_data_source": data_source, 
            "total_providers": len(npi_data) if npi_data is not None else 0, 
            "providers_data_file": str(data_file_path) if data_file_path else None, 
            "hospitals_data_loaded": hospitals_data is not None and len(hospitals_data) > 0, 
            "hospitals_data_source": hospitals_source, 
            "total_hospitals": len(hospitals_data) if hospitals_data is not None else 0, 
            "hospitals_data_file": str(hospitals_file_path) if hospitals_file_path else None, 
            "taxonomy_lookup_loaded": len(TAXONOMY_LOOKUP) > 0, 
            "taxonomy_descriptions_count": len(TAXONOMY_LOOKUP), 
        } 
    ) 


@app.route("/api/taxonomy/codes", methods=["GET"]) 
@require_data 
def get_taxonomy_codes(): 
    """Get list of common taxonomy code mappings.""" 
    return jsonify( 
        { 
            "taxonomy_mappings": TAXONOMY_CODES, 
            "taxonomy_descriptions_loaded": len(TAXONOMY_LOOKUP) > 0, 
            "total_descriptions": len(TAXONOMY_LOOKUP), 
            "usage": "Use these keywords in the specialty parameter, or provide taxonomy codes directly", 
            "example": "/api/providers/search/specialty?specialty=otolaryngology&state=MN", 
        } 
    ) 


@app.route("/api/providers/search/location", methods=["GET"]) 
@require_data 
def search_by_location(): 
    """Search providers by location with optional distance filtering.""" 
    city = request.args.get("city", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    specialty = request.args.get("specialty", "").strip() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    if not city or not state: 
        return ( 
            jsonify( 
                { 
                    "error": "Missing required parameters", 
                    "message": "Both city and state are required", 
                } 
            ), 
            400, 
        ) 

    # Sanitize user inputs before logging 
    safe_city = city.replace("\r", "").replace("\n", "") 
    safe_state = state.replace("\r", "").replace("\n", "") 
    safe_specialty = specialty.replace("\r", "").replace("\n", "") 
    logger.info( 
        f"Location search: {safe_city}, {safe_state}, specialty={safe_specialty}, limit={limit}" 
    ) 

    city_col = "Provider Business Practice Location Address City Name" 
    state_col = "Provider Business Practice Location Address State Name" 

    mask = (npi_data[city_col].str.upper() == city.upper()) & ( 
        npi_data[state_col] == state 
    ) 

    if specialty: 
        specialty_mask = search_across_taxonomy_codes(specialty) 
        mask &= specialty_mask 
        if not specialty_mask.any(): 
            return ( 
                jsonify( 
                    { 
                        "count": 0, 
                        "search_params": { 
                            "city": city, 
                            "state": state, 
                            "specialty": specialty, 
                            "limit": limit, 
                        }, 
                        "message": f'No providers found with specialty "{specialty}". Check /api/taxonomy/codes.', 
                        "results": [], 
                    } 
                ), 
                200, 
            ) 

    results = npi_data[mask].head(limit) 
    formatted_results = [format_provider(row) for _, row in results.iterrows()] 
    return jsonify( 
        { 
            "count": len(formatted_results), 
            "search_params": { 
                "city": city, 
                "state": state, 
                "specialty": specialty or None, 
                "limit": limit, 
            }, 
            "results": formatted_results, 
        } 
    ) 


@app.route("/api/providers/search/specialty", methods=["GET"]) 
@require_data 
def search_by_specialty(): 
    """Search providers by specialty (taxonomy code or keyword).""" 
    specialty = request.args.get("specialty", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    # Sanitize user input for logging 
    specialty_clean = specialty.replace("\n", "").replace("\r", "") 
    state_clean = state.replace("\n", "").replace("\r", "") 

    if not specialty: 
        return ( 
            jsonify( 
                { 
                    "error": "Missing required parameter", 
                    "message": "specialty parameter is required", 
                } 
            ), 
            400, 
        ) 

    logger.info( 
        f"Specialty search: {specialty_clean}, state={state_clean}, limit={limit}" 
    ) 

    mask = search_across_taxonomy_codes(specialty) 
    if not mask.any(): 
        return ( 
            jsonify( 
                { 
                    "count": 0, 
                    "search_params": { 
                        "specialty": specialty, 
                        "state": state or None, 
                        "limit": limit, 
                    }, 
                    "message": f'No providers found with specialty "{specialty}". Check /api/taxonomy/codes.', 
                    "results": [], 
                } 
            ), 
            200, 
        ) 

    if state: 
        state_col = "Provider Business Practice Location Address State Name" 
        mask &= npi_data[state_col] == state 

    results = npi_data[mask].head(limit) 
    formatted_results = [format_provider(row) for _, row in results.iterrows()] 
    return jsonify( 
        { 
            "count": len(formatted_results), 
            "search_params": { 
                "specialty": specialty, 
                "state": state or None, 
                "limit": limit, 
            }, 
            "results": formatted_results, 
        } 
    ) 


@app.route("/api/providers/search/state/<state_code>", methods=["GET"]) 
@require_data 
def search_by_state(state_code: str): 
    """Get all providers in a specific state.""" 
    state_code = state_code.strip().upper() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    state_col = "Provider Business Practice Location Address State Name" 
    results = npi_data[npi_data[state_col] == state_code].head(limit) 
    formatted_results = [format_provider(row) for _, row in results.iterrows()] 
    return jsonify( 
        { 
            "count": len(formatted_results), 
            "state": state_code, 
            "limit": limit, 
            "results": formatted_results, 
        } 
    ) 


@app.route("/api/providers/<npi>", methods=["GET"]) 
@require_data 
def get_provider_by_npi(npi: str): 
    """Get detailed provider information by NPI number.""" 
    sanitized_npi = npi.replace("\r", "").replace("\n", "") 
    logger.info(f"NPI lookup: {sanitized_npi}") 

    result = npi_data[npi_data["NPI"].astype(str) == npi] 
    if len(result) == 0: 
        return ( 
            jsonify( 
                { 
                    "error": "Provider not found", 
                    "message": f"No provider found with NPI {npi}", 
                } 
            ), 
            404, 
        ) 

    provider = format_provider(result.iloc[0]) 
    return jsonify(provider) 


@app.route("/api/providers/search/postal_code/<postal_code>", methods=["GET"]) 
@require_data 
def search_by_postal_code(postal_code: str): 
    """Search providers by postal code.""" 
    postal_code = postal_code.strip()[:5] 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    postal_col = "Provider Business Practice Location Address Postal Code" 
    results = npi_data[npi_data[postal_col].str.startswith(postal_code, na=False)].head( 
        limit 
    ) 

    formatted_results = [format_provider(row) for _, row in results.iterrows()] 
    return jsonify( 
        { 
            "count": len(formatted_results), 
            "postal_code": postal_code, 
            "limit": limit, 
            "results": formatted_results, 
        } 
    ) 


@app.route("/api/hospitals/search/location", methods=["GET"]) 
@require_hospital_data 
def search_hospitals_by_location(): 
    """ 
    Search for hospitals by location (flexible parameters). 
    Draws from processed organizations when available. 
    Can search by: 
    1. City + State combination 
    2. Postal code 
    3. Exact address (address line + city + state) 

    Returns simplified hospital records with unique addresses. 
    Duplicates (same address + organization name) are removed. 

    Parameters: 
        city (str): City name 
        state (str): State code, e.g., 'MN' 
        postal_code (str): 5-digit postal code 
        address (str): Address line to search 
        limit (int): Max results to return (default: 50, max: 500) 
    """ 
    city = request.args.get("city", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    postal_code = request.args.get("postal_code", "").strip()[:5] 
    address = request.args.get("address", "").strip() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    # Validate that at least one search parameter is provided 
    if not (city or postal_code or address): 
        return ( 
            jsonify( 
                { 
                    "error": "Missing required parameters", 
                    "message": "Provide at least one of: city+state, postal_code, or address+city+state", 
                    "examples": [ 
                        "/api/hospitals/search/location?city=Minneapolis&state=MN", 
                        "/api/hospitals/search/location?postal_code=55401", 
                        "/api/hospitals/search/location?address=1000%205th%20Street&city=Minneapolis&state=MN", 
                    ], 
                } 
            ), 
            400, 
        ) 

    # Sanitize for logging 
    safe_city = city.replace("\r", "").replace("\n", "") 
    safe_state = state.replace("\r", "").replace("\n", "") 
    safe_postal = postal_code.replace("\r", "").replace("\n", "") 
    safe_address = address.replace("\r", "").replace("\n", "") 
    logger.info( 
        f"Hospital search: city={safe_city}, state={safe_state}, postal={safe_postal}, address={safe_address}, limit={limit}" 
    ) 

    df = hospitals_data  # <-- organizations 
    city_col = "Provider Business Practice Location Address City Name" 
    state_col = "Provider Business Practice Location Address State Name" 
    postal_col = "Provider Business Practice Location Address Postal Code" 
    address_col = "Provider First Line Business Practice Location Address" 
    org_col = "Provider Organization Name (Legal Business Name)" 

    # Start with all records 
    mask = pd.Series([True] * len(df), index=df.index) 

    # Apply location filters based on what was provided 
    if city and state: 
        # City + State search 
        mask &= (df[city_col].str.upper() == city.upper()) & (df[state_col] == state) 
    elif postal_code: 
        # Postal code search 
        mask &= df[postal_col].str.startswith(postal_code, na=False) 

    if address: 
        # Address search (can be combined with other filters) 
        mask &= (df[address_col].str.upper().str.contains(address.upper(), na=False)) 

    results = df[mask] 
    if len(results) == 0: 
        search_params = { 
            "city": city or None, 
            "state": state or None, 
            "postal_code": postal_code or None, 
            "address": address or None, 
            "limit": limit, 
        } 
        return ( 
            jsonify( 
                { 
                    "count": 0, 
                    "search_params": search_params, 
                    "message": "No hospitals found matching the search criteria", 
                    "results": [], 
                } 
            ), 
            200, 
        ) 

    # Format simplified hospital records 
    hospital_records = [] 
    seen = set()  # Track unique hospitals by organization name and full address 

    for _, row in results.iterrows(): 
        # Extract address components 
        address_1 = ( 
            str(row.get("Provider First Line Business Practice Location Address", "")).strip() 
            if pd.notna( 
                row.get("Provider First Line Business Practice Location Address") 
            ) 
            else "" 
        ) 
        address_2 = ( 
            str(row.get("Provider Second Line Business Practice Location Address", "")).strip() 
            if pd.notna( 
                row.get("Provider Second Line Business Practice Location Address") 
            ) 
            else "" 
        ) 
        city_val = ( 
            str(row.get("Provider Business Practice Location Address City Name", "")).strip() 
            if pd.notna(row.get("Provider Business Practice Location Address City Name")) 
            else "" 
        ) 
        state_val = ( 
            str(row.get("Provider Business Practice Location Address State Name", "")).strip() 
            if pd.notna(row.get("Provider Business Practice Location Address State Name")) 
            else "" 
        ) 
        postal_val = ( 
            str(row.get("Provider Business Practice Location Address Postal Code", ""))[:5].strip() 
            if pd.notna(row.get("Provider Business Practice Location Address Postal Code")) 
            else "" 
        ) 

        # Extract organization name 
        org_name = ( 
            str(row.get("Provider Organization Name (Legal Business Name)", "")).strip() 
            if pd.notna(row.get("Provider Organization Name (Legal Business Name)")) 
            and str(row.get("Provider Organization Name (Legal Business Name)")).lower() 
            != "nan" 
            else None 
        ) 

        # Create unique identifier for deduplication 
        unique_key = (org_name, address_1, address_2, city_val, state_val, postal_val) 

        # Skip if we've already seen this hospital at this address 
        if unique_key in seen: 
            continue 
        seen.add(unique_key) 

        # Create simplified hospital record 
        hospital_record = { 
            "address": { 
                "address_1": address_1, 
                "address_2": address_2, 
                "city": city_val, 
                "postal_code": postal_val, 
                "state": state_val, 
            }, 
            "organization_name": org_name, 
            "is_hospital": (bool(row.get("is_hospital")) if pd.notna(row.get("is_hospital")) else True), 
            "system_homepage_url": (str(row.get("system_homepage_url")).strip() if pd.notna(row.get("system_homepage_url")) and str(row.get("system_homepage_url")).lower() != "nan" else None), 
            "location_page_url": (str(row.get("location_page_url")).strip() if pd.notna(row.get("location_page_url")) and str(row.get("location_page_url")).lower() != "nan" else None), 
            "org_entity_id": (str(row.get("org_entity_id")).strip() if pd.notna(row.get("org_entity_id")) and str(row.get("org_entity_id")).lower() != "nan" else None), 
            "provider_count": (int(row.get("provider_count")) if pd.notna(row.get("provider_count")) else None), 
            "provider_count_entity": (int(row.get("provider_count_entity")) if pd.notna(row.get("provider_count_entity")) else None), 
        } 
        hospital_records.append(hospital_record) 

        # Stop if we've reached the limit 
        if len(hospital_records) >= limit: 
            break 

    search_params = { 
        "city": city or None, 
        "state": state or None, 
        "postal_code": postal_code or None, 
        "address": address or None, 
        "limit": limit, 
    } 

    return jsonify( 
        { 
            "count": len(hospital_records), 
            "search_params": search_params, 
            "results": hospital_records, 
        } 
    ) 


@app.route("/api/providers/search/hospital", methods=["GET"]) 
@require_data 
def search_providers_by_hospital(): 
    """ 
    Search for all providers affiliated with a specific hospital. 
    Providers can be matched to a hospital by: 
    1. Hospital name (searches organization_name field) 
    2. Hospital NPI (searches organization relationships if available) 

    Matching uses case-insensitive partial matching on organization name. 
    """ 
    hospital = request.args.get("hospital", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    if not hospital: 
        return ( 
            jsonify( 
                { 
                    "error": "Missing required parameter", 
                    "message": "hospital parameter is required", 
                    "example": "/api/providers/search/hospital?hospital=Mayo Clinic&state=MN", 
                } 
            ), 
            400, 
        ) 

    # Sanitize for logging 
    hospital_clean = hospital.replace("\n", "").replace("\r", "") 
    state_clean = state.replace("\n", "").replace("\r", "") 
    logger.info( 
        f"Hospital provider search: {hospital_clean}, state={state_clean}, limit={limit}" 
    ) 

    org_name_col = "Provider Organization Name (Legal Business Name)" 
    state_col = "Provider Business Practice Location Address State Name" 

    # Case-insensitive partial match on organization name 
    hospital_upper = hospital.upper() 
    mask = ( 
        npi_data[org_name_col].notna() 
        & (npi_data[org_name_col].astype(str) != "nan") 
        & (npi_data[org_name_col].str.upper().str.contains(hospital_upper, na=False)) 
    ) 

    if state: 
        mask &= npi_data[state_col] == state 

    results = npi_data[mask].head(limit) 
    if len(results) == 0: 
        return ( 
            jsonify( 
                { 
                    "count": 0, 
                    "search_params": { 
                        "hospital": hospital, 
                        "state": state or None, 
                        "limit": limit, 
                    }, 
                    "message": f'No providers found for hospital "{hospital}"', 
                    "results": [], 
                } 
            ), 
            200, 
        ) 

    formatted_results = [format_provider(row) for _, row in results.iterrows()] 
    return jsonify( 
        { 
            "count": len(formatted_results), 
            "search_params": { 
                "hospital": hospital, 
                "state": state or None, 
                "limit": limit, 
            }, 
            "results": formatted_results, 
        } 
    ) 


# ===== NEW ENDPOINT: Search providers by provider name with optional filters ===== 
@app.route("/api/providers/search/name", methods=["GET"]) 
@require_data 
def search_providers_by_name(): 
    """ 
    Search providers by provider name (case-insensitive partial match) with optional filters. 

    Required: 
        - name (str): Provider name (e.g., "Jane Doe" or "Doe") 

    Optional: 
        - city (str) 
        - state (str, 2-letter) 
        - postal_code (str, 5-digit prefix accepted) 
        - hospital (str): hospital/organization name to match provider's organization 
        - limit (int): default 50, max 500 
    """ 
    name = request.args.get("name", "").strip() 
    if not name:
        return (
            jsonify({
                "error": "Missing required parameter",
                "message": "name parameter is required",
                "example": "/api/providers/search/name?name=Jane%20Doe&city=Minneapolis&state=MN"
            }),
            400,
        )

    city = request.args.get("city", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    postal_code = request.args.get("postal_code", "").strip()[:5] 
    hospital = request.args.get("hospital", "").strip() 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    # Log (sanitized)
    logger.info(
        f"Provider name search: name={name.replace('\n',' ').replace('\r',' ')}, "
        f"city={city.replace('\n',' ').replace('\r',' ')}, state={state.replace('\n',' ').replace('\r',' ')}, "
        f"postal={postal_code.replace('\n',' ').replace('\r',' ')}, hospital={hospital.replace('\n',' ').replace('\r',' ')}, "
        f"limit={limit}"
    )

    # Columns
    first_col = "Provider First Name"
    last_col = "Provider Last Name (Legal Name)"
    city_col = "Provider Business Practice Location Address City Name" 
    state_col = "Provider Business Practice Location Address State Name" 
    postal_col = "Provider Business Practice Location Address Postal Code" 
    org_col = "Provider Organization Name (Legal Business Name)" 

    # Build full name series for matching
    full_name = (
        npi_data[first_col].fillna("").astype(str).str.strip() + " " +
        npi_data[last_col].fillna("").astype(str).str.strip()
    ).str.strip()

    mask = full_name.str.upper().str.contains(name.upper(), na=False)

    if city:
        mask &= npi_data[city_col].str.upper() == city.upper()
    if state:
        mask &= npi_data[state_col] == state
    if postal_code:
        mask &= npi_data[postal_col].str.startswith(postal_code, na=False)
    if hospital:
        mask &= (
            npi_data[org_col].notna() & (npi_data[org_col].astype(str) != "nan") &
            npi_data[org_col].str.upper().str.contains(hospital.upper(), na=False)
        )

    results = npi_data[mask].head(limit)

    if len(results) == 0:
        return (
            jsonify({
                "count": 0,
                "search_params": {
                    "name": name,
                    "city": city or None,
                    "state": state or None,
                    "postal_code": postal_code or None,
                    "hospital": hospital or None,
                    "limit": limit,
                },
                "message": "No providers found matching the specified criteria",
                "results": []
            }),
            200,
        )

    formatted_results = [format_provider(row) for _, row in results.iterrows()]
    return jsonify({
        "count": len(formatted_results),
        "search_params": {
            "name": name,
            "city": city or None,
            "state": state or None,
            "postal_code": postal_code or None,
            "hospital": hospital or None,
            "limit": limit,
        },
        "results": formatted_results,
    })


# ===== NEW ENDPOINT: Search hospitals by hospital name with optional filters ===== 
@app.route("/api/hospitals/search/name", methods=["GET"]) 
@require_hospital_data 
def search_hospitals_by_name(): 
    """ 
    Search hospitals by hospital name (case-insensitive partial match) with optional filters. 

    Required: 
        - hospital (str): Hospital/organization name or fragment 

    Optional: 
        - city (str) 
        - state (str, 2-letter) 
        - postal_code (str, 5-digit prefix accepted) 
        - limit (int): default 50, max 500 
    """ 
    hospital = request.args.get("hospital", "").strip() 
    if not hospital:
        return (
            jsonify({
                "error": "Missing required parameter",
                "message": "hospital parameter is required",
                "example": "/api/hospitals/search/name?hospital=Mayo%20Clinic&state=MN"
            }),
            400,
        )

    city = request.args.get("city", "").strip() 
    state = request.args.get("state", "").strip().upper() 
    postal_code = request.args.get("postal_code", "").strip()[:5] 
    limit = min(int(request.args.get("limit", MAX_RESULTS_DEFAULT)), MAX_RESULTS_LIMIT) 

    # Log (sanitized)
    logger.info(
        f"Hospital name search: hospital={hospital.replace('\n',' ').replace('\r',' ')}, "
        f"city={city.replace('\n',' ').replace('\r',' ')}, state={state.replace('\n',' ').replace('\r',' ')}, "
        f"postal={postal_code.replace('\n',' ').replace('\r',' ')}, limit={limit}"
    )

    df = hospitals_data
    org_col = "Provider Organization Name (Legal Business Name)" 
    city_col = "Provider Business Practice Location Address City Name" 
    state_col = "Provider Business Practice Location Address State Name" 
    postal_col = "Provider Business Practice Location Address Postal Code" 

    mask = (
        df[org_col].notna() & (df[org_col].astype(str) != "nan") &
        df[org_col].str.upper().str.contains(hospital.upper(), na=False)
    )

    if city:
        mask &= df[city_col].str.upper() == city.upper()
    if state:
        mask &= df[state_col] == state
    if postal_code:
        mask &= df[postal_col].str.startswith(postal_code, na=False)

    results = df[mask]

    if len(results) == 0:
        return (
            jsonify({
                "count": 0,
                "search_params": {
                    "hospital": hospital,
                    "city": city or None,
                    "state": state or None,
                    "postal_code": postal_code or None,
                    "limit": limit,
                },
                "message": "No hospitals found matching the specified criteria",
                "results": []
            }),
            200,
        )

    # Deduplicate records by org name + full address similar to /hospitals/search/location
    hospital_records = []
    seen = set()
    for _, row in results.iterrows():
        address_1 = (
            str(row.get("Provider First Line Business Practice Location Address", "")).strip()
            if pd.notna(row.get("Provider First Line Business Practice Location Address")) else ""
        )
        address_2 = (
            str(row.get("Provider Second Line Business Practice Location Address", "")).strip()
            if pd.notna(row.get("Provider Second Line Business Practice Location Address")) else ""
        )
        city_val = (
            str(row.get("Provider Business Practice Location Address City Name", "")).strip()
            if pd.notna(row.get("Provider Business Practice Location Address City Name")) else ""
        )
        state_val = (
            str(row.get("Provider Business Practice Location Address State Name", "")).strip()
            if pd.notna(row.get("Provider Business Practice Location Address State Name")) else ""
        )
        postal_val = (
            str(row.get("Provider Business Practice Location Address Postal Code", ""))[:5].strip()
            if pd.notna(row.get("Provider Business Practice Location Address Postal Code")) else ""
        )
        org_name = (
            str(row.get("Provider Organization Name (Legal Business Name)", "")).strip()
            if pd.notna(row.get("Provider Organization Name (Legal Business Name)")) and str(row.get("Provider Organization Name (Legal Business Name)")).lower() != "nan" else None
        )

        unique_key = (org_name, address_1, address_2, city_val, state_val, postal_val)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        record = {
            "address": {
                "address_1": address_1,
                "address_2": address_2,
                "city": city_val,
                "postal_code": postal_val,
                "state": state_val,
            },
            "organization_name": org_name,
            "is_hospital": (bool(row.get("is_hospital")) if pd.notna(row.get("is_hospital")) else True),
            "system_homepage_url": (str(row.get("system_homepage_url")).strip() if pd.notna(row.get("system_homepage_url")) and str(row.get("system_homepage_url")).lower() != "nan" else None),
            "location_page_url": (str(row.get("location_page_url")).strip() if pd.notna(row.get("location_page_url")) and str(row.get("location_page_url")).lower() != "nan" else None),
            "org_entity_id": (str(row.get("org_entity_id")).strip() if pd.notna(row.get("org_entity_id")) and str(row.get("org_entity_id")).lower() != "nan" else None),
            "provider_count": (int(row.get("provider_count")) if pd.notna(row.get("provider_count")) else None),
            "provider_count_entity": (int(row.get("provider_count_entity")) if pd.notna(row.get("provider_count_entity")) else None),
        }
        hospital_records.append(record)
        if len(hospital_records) >= limit:
            break

    return jsonify({
        "count": len(hospital_records),
        "search_params": {
            "hospital": hospital,
            "city": city or None,
            "state": state or None,
            "postal_code": postal_code or None,
            "limit": limit,
        },
        "results": hospital_records,
    })


@app.route("/", methods=["GET"]) 
def index(): 
    """API documentation endpoint.""" 
    return """ 
<!DOCTYPE html> 
<html> 
<head> 
<title>NPI Registry API</title> 
<style> 
body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; } 
.container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; } 
h1 { color: #333; } 
h2 { color: #666; margin-top: 30px; border-bottom: 2px solid #007bff; padding-bottom: 10px; } 
.endpoint { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; } 
.method { font-weight: bold; color: #007bff; } 
code { background-color: #f0f0f0; padding: 2px 6px; border-radius: 3px; } 
.example { background-color: #e7f3ff; padding: 10px; margin: 10px 0; border-radius: 4px; } 
</style> 
</head> 
<body> 
<div class="container"> 
<h1>🏥 NPI Registry API</h1> 
<p>Fast healthcare provider search API for millions of providers.</p> 

<h2>Data Loading Strategy</h2> 
<ul> 
<li>✓ Providers: Loads processed individuals from <code>/processed_data/npi_individuals_processed_*</code>, falls back to <code>/parquet/npi_individuals_*</code></li> 
<li>✓ Hospitals: Loads processed organizations from <code>/processed_data/npi_organizations_processed_*</code>, falls back to <code>/parquet/npi_organizations_*</code></li> 
<li>✓ Always includes organization_name in responses</li> 
</ul> 

<h2>Optimizations</h2> 
<ul> 
<li>✓ Vectorized taxonomy search (10-100x faster)</li> 
<li>✓ Only loads essential columns (optimized memory)</li> 
<li>✓ Uses exact match instead of regex</li> 
</ul> 

<h2>Endpoints</h2> 
<div class="endpoint"> 
<div class="method">GET /api/health</div> 
<p>Health check and data status (providers & hospitals)</p> 
<div class="example"><code>/api/health</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/taxonomy/codes</div> 
<p>List of taxonomy codes and keyword mappings</p> 
<div class="example"><code>/api/taxonomy/codes</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/&lt;npi&gt;</div> 
<p>Get provider by NPI number</p> 
<div class="example"><code>/api/providers/1234567890</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/specialty</div> 
<p>Search by specialty (keyword or code)</p> 
<div class="example"><code>/api/providers/search/specialty?specialty=otolaryngology&state=MN&limit=50</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/location</div> 
<p>Search by location</p> 
<div class="example"><code>/api/providers/search/location?city=Minneapolis&state=MN&limit=50</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/state/&lt;state_code&gt;</div> 
<p>Get all providers in a state</p> 
<div class="example"><code>/api/providers/search/state/MN?limit=50</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/postal_code/&lt;postal_code&gt;</div> 
<p>Search by postal code</p> 
<div class="example"><code>/api/providers/search/postal_code/55401?limit=50</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/hospital</div> 
<p>Search for providers by hospital name</p> 
<div class="example"><code>/api/providers/search/hospital?hospital=Mayo Clinic&state=MN&limit=25</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/providers/search/name</div> 
<p>Search for providers by provider name with optional city/state/postal/hospital filters</p> 
<div class="example"><code>/api/providers/search/name?name=Jane%20Doe&city=Minneapolis&state=MN&limit=25</code></div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/hospitals/search/location</div> 
<p>Search for hospitals by location (city+state, postal code, or address). Returns simplified list with duplicates removed.</p> 
<div class="example"> 
<code>/api/hospitals/search/location?city=Minneapolis&state=MN&limit=25</code><br> 
<code>/api/hospitals/search/location?postal_code=55401&limit=25</code><br> 
<code>/api/hospitals/search/location?address=1000%205th%20Street&city=Minneapolis&state=MN</code> 
</div> 
</div> 

<div class="endpoint"> 
<div class="method">GET /api/hospitals/search/name</div> 
<p>Search for hospitals by hospital name with optional city/state/postal filters</p> 
<div class="example"><code>/api/hospitals/search/name?hospital=Mayo Clinic&state=MN&limit=25</code></div> 
</div> 

</div> 
</body> 
</html> 
""" 


@app.errorhandler(404) 
def not_found(error): 
    """Handle 404 errors.""" 
    return ( 
        jsonify({"error": "Not found", "message": "The requested endpoint does not exist"}), 
        404, 
    ) 


@app.errorhandler(500) 
def internal_error(error): 
    """Handle 500 errors.""" 
    logger.error(f"Internal error: {error}") 
    return ( 
        jsonify( 
            { 
                "error": "Internal server error", 
                "message": "An unexpected error occurred", 
            } 
        ), 
        500, 
    ) 


# ==================== STARTUP ==================== 
if __name__ == "__main__": 
    # Providers: processed individuals -> raw individuals 
    providers_ok = load_npi_data() 

    # Hospitals: processed orgs -> raw orgs 
    hospitals_ok = load_hospitals_data() 

    if not providers_ok: 
        logger.error("Failed to load providers (individuals) data. Provider endpoints will not function properly.") 
    if not hospitals_ok: 
        logger.error("Failed to load hospitals (organizations) data. Hospital endpoints will not function properly.") 

    port = int(os.getenv("API_PORT", "5000")) 
    host = os.getenv("API_HOST", "0.0.0.0") 
    debug = os.getenv("FLASK_ENV") == "development" 

    logger.info(f"\n{'=' * 80}") 
    logger.info(f"🚀 Starting NPI Registry API") 
    logger.info(f"{'=' * 80}") 
    logger.info(f"Host: {host}") 
    logger.info(f"Port: {port}") 
    logger.info(f"Debug: {debug}") 
    logger.info(f"Providers loaded: {npi_data is not None} (source: {data_source})") 
    logger.info(f"Hospitals loaded: {hospitals_data is not None and len(hospitals_data) > 0} (source: {hospitals_source})") 
    logger.info(f"{'=' * 80}\n") 
    app.run(host=host, port=port, debug=debug)
