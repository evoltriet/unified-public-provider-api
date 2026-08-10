# Unified CMS Provider API Reference

The Flask API serves the CMS provider baseline built from PPEF, PECOS, and NPI Registry data. It loads the `current_active_primary` provider view by default and uses the canonical PECOS clinic and system rollups.

## Start Server

```bash
# Development
python src/api.py

# Production
gunicorn -c src/gunicorn_config.py src.api:app
```

## Data Loading

The default CMS source loads:

- `data/processed_data/ppef_individuals_comparison_baseline_*.parquet`
- `data/processed_data/pecos_clinic_rollup_*.parquet`
- `data/processed_data/pecos_system_rollup_*.parquet`

Set `API_DATA_SOURCE=legacy_npi` to use the older NPPES API artifacts directly. If CMS artifacts are absent, `ALLOW_LEGACY_NPI_FALLBACK=true` permits automatic fallback.

## Provider Endpoints

### Health

```text
GET /api/health
```

Reports source, baseline view, snapshot dates, artifact paths, and provider/clinic/system counts.

### Provider by NPI

```text
GET /api/providers/{10-digit-npi}
```

### Search by Name

```text
GET /api/providers/search/name?name=Jane%20Doe&city=Minneapolis&state=MN&hospital=North%20Health&location_source=any&limit=25
```

`name` is required. Optional filters are `city`, `state`, `postal_code`, `hospital`, `location_source`, and `limit`.

### Search by Specialty

```text
GET /api/providers/search/specialty?specialty=Cardiology&state=MN&limit=25
```

Specialty search considers taxonomy codes, taxonomy descriptions, normalized specialty, specialty group, and provider type.

### Search by Location

```text
GET /api/providers/search/location?city=Minneapolis&state=MN&location_source=any&limit=25
```

`city` and `state` are required. `location_source` accepts:

- `any`: provider practice or mapped affiliation location, the default
- `provider`: NPI Registry provider-practice location only
- `affiliation`: mapped PECOS clinic location only

### Search by State or ZIP

```text
GET /api/providers/search/state/MN?limit=25
GET /api/providers/search/postal_code/55401?limit=25
```

### Search by Hospital, System, or Clinic Affiliation

```text
GET /api/providers/search/hospital?hospital=North%20Health&state=MN&limit=25
```

The hospital query is matched against both mapped system and clinic names.

## Organization Endpoints

```text
GET /api/clinics/search/name?clinic=North%20Clinic&state=MN&limit=25
GET /api/clinics/search/location?postal_code=55401&limit=25

GET /api/systems/search/name?system=North%20Health&state=MN&limit=25
GET /api/systems/search/location?city=Minneapolis&state=MN&limit=25

GET /api/hospitals/search/name?hospital=North%20Health&state=MN&limit=25
GET /api/hospitals/search/location?postal_code=55401&limit=25
```

Hospital routes are compatibility routes over PECOS systems marked as hospitals. System routes include all canonical system entities.

Organization location search accepts `city+state`, `postal_code`, or `address`.

## Provider Response

Legacy keys remain available, including `organization_name`, `taxonomy_codes`, `primary_taxonomy`, `primary_specialty`, and `address`. CMS fields are additive:

```json
{
  "npi": "1234567890",
  "provider_id": "provider_1234567890",
  "provider_name": "Jane Q Doe",
  "organization_name": "North Clinic",
  "primary_specialty": "Cardiology",
  "address": {
    "address_1": "10 Old Road",
    "city": "St Paul",
    "state": "MN",
    "postal_code": "55101",
    "phone": "6515550100"
  },
  "specialty": {
    "taxonomy_code": "207RC0000X",
    "taxonomy_description": "Cardiovascular Disease",
    "normalized": "Cardiology",
    "group": "Cardiology",
    "provider_type": "Physician"
  },
  "affiliation": {
    "clinic": {"id": "clinic_1", "name": "North Clinic"},
    "system": {"id": "system_1", "name": "North Health"},
    "practice_location": {
      "address_1": "100 Main Street",
      "city": "Minneapolis",
      "state": "MN",
      "postal_code": "55401",
      "phone": "6125550100"
    },
    "relationship": {"source": "PPEF_REASSIGNMENT", "active": true}
  },
  "currentness": {
    "active": true,
    "tier": "current_active_primary",
    "as_of_date": "2026-05-05"
  },
  "mapping": {
    "confidence_tier": "high",
    "method": "reassignment",
    "score": 12.5,
    "ambiguous": false
  },
  "provenance": {
    "ppef_snapshot_date": "2026-05-05",
    "pecos_snapshot_date": "2026-05-05",
    "npi_registry_snapshot_date": "2026-05-05"
  }
}
```

## Environment Variables

- `DATA_DIR`, default `data`
- `API_DATA_SOURCE`, `cms` or `legacy_npi`, default `cms`
- `CMS_BASELINE_VIEW`, default `current_active_primary`
- `CMS_PROVIDER_PATH`, optional explicit baseline parquet
- `CMS_CLINIC_PATH`, optional explicit clinic-rollup parquet
- `CMS_SYSTEM_PATH`, optional explicit system-rollup parquet
- `ALLOW_LEGACY_NPI_FALLBACK`, default `true`
- `API_HOST`, default `0.0.0.0`
- `API_PORT`, default `5000`
- `MAX_RESULTS_DEFAULT`, default `50`
- `MAX_RESULTS_LIMIT`, default `500`
- `GUNICORN_WORKERS`, default `2`

## Status Codes

- `200`: success, including valid searches with no results
- `400`: invalid or missing query parameter
- `404`: NPI or route not found
- `500`: unexpected server error
- `503`: required dataset not loaded
