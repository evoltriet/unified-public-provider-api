# NPI Registry API – Reference

## 🚀 Start Server
```bash
# Development
python api.py

# Production (example)
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

## 📦 Data Loading
- Providers load from `data/processed_data/npi_individuals_processed_*.parquet` (fallback `data/parquet/npi_individuals_*.parquet`).
- Hospitals load from `data/processed_data/npi_organizations_processed_*.parquet` (fallback `data/parquet/npi_organizations_*.parquet`).
- `MAX_RESULTS_DEFAULT` (default **50**) and `MAX_RESULTS_LIMIT` (default **500**) control paging.

## 📡 Endpoints

### Health Check
```
GET /api/health
```

### Taxonomy Keywords → Codes
```
GET /api/taxonomy/codes
```

### Provider – Get by NPI
```
GET /api/providers/{npi}
```

### Provider – Search by Specialty (keyword or taxonomy code)
```
GET /api/providers/search/specialty?specialty={value}&state={STATE}&limit={N}
```

### Provider – Search by Location (city & state, optional specialty)
```
GET /api/providers/search/location?city={City}&state={STATE}&specialty={value}&limit={N}
```

### Provider – Search by State
```
GET /api/providers/search/state/{STATE}?limit={N}
```

### Provider – Search by Postal Code
```
GET /api/providers/search/postal_code/{ZIP5}?limit={N}
```

### Provider – **Search by Provider Name** (NEW)
```
GET /api/providers/search/name?name={Jane%20Doe}&city={City}&state={STATE}&postal_code={ZIP5}&hospital={Org}&limit={N}
```
- **Required**: `name`
- **Optional**: `city`, `state`, `postal_code` (prefix match), `hospital` (organization name), `limit`

### Provider – Search by Hospital Name
```
GET /api/providers/search/hospital?hospital={Org}&state={STATE}&limit={N}
```

### Hospital – Search by Location (city+state OR postal_code OR address+city+state)
```
GET /api/hospitals/search/location?city={City}&state={STATE}&postal_code={ZIP5}&address={Line1}&limit={N}
```

### Hospital – **Search by Hospital Name** (NEW)
```
GET /api/hospitals/search/name?hospital={Org}&city={City}&state={STATE}&postal_code={ZIP5}&limit={N}
```
- **Required**: `hospital`
- **Optional**: `city`, `state`, `postal_code` (prefix match), `limit`

## 🔧 Common Parameters
- `state`: two-letter uppercase (normalized internally)
- `postal_code`: first **5** digits are used for matching
- `limit`: default **50**, max **500** (capped server-side)

## 🧾 Response Shapes (high level)

### Provider Object
```json
{
  "npi": "1234567890",
  "entity_type": "Individual",
  "provider_name": "Jane Doe, MD",
  "organization_name": "Mayo Clinic",
  "taxonomy_codes": ["207Y00000X", "207YX0901X"],
  "primary_taxonomy": "207Y00000X",
  "specialty_descriptions": ["Otolaryngology"],
  "primary_specialty": "Otolaryngology",
  "address": {
    "address_1": "123 Main St",
    "address_2": "Suite 100",
    "city": "Rochester",
    "state": "MN",
    "postal_code": "55901",
    "phone": "5075550000"
  },
  "geocode": {"latitude": 44.0, "longitude": -93.0},
  "geohash": "9zvx..."
}
```

### Hospital Record (simplified)
```json
{
  "organization_name": "Mayo Clinic Hospital",
  "address": {
    "address_1": "1216 Second St SW",
    "address_2": "",
    "city": "Rochester",
    "state": "MN",
    "postal_code": "55902"
  },
  "is_hospital": true,
  "system_homepage_url": "https://www.mayoclinic.org/",
  "location_page_url": "https://www.mayoclinic.org/patient-visitor-guide/minnesota",
  "org_entity_id": "org_123",
  "provider_count": 2500,
  "provider_count_entity": 1800
}
```

## 🐛 Error Codes
- **200** Success (may include empty `results` with explanatory `message`)
- **400** Bad Request (missing required parameters)
- **404** Not Found (e.g., provider by NPI)
- **500** Server Error
- **503** Service Unavailable (dataset not loaded)

## ⚙️ Environment Variables
- `DATA_DIR` (default `data`)
- `API_HOST` (default `0.0.0.0`)
- `API_PORT` (default `5000`)
- `FLASK_ENV` (`development` enables debug)
- `MAX_RESULTS_DEFAULT` (default `50`)
- `MAX_RESULTS_LIMIT` (default `500`)

## 🔎 Examples
```bash
# Provider by name
curl "http://localhost:5000/api/providers/search/name?name=Jane%20Doe&city=Minneapolis&state=MN&limit=25"

# Hospital by name
curl "http://localhost:5000/api/hospitals/search/name?hospital=Mayo%20Clinic&state=MN&limit=25"

# Providers by hospital name
curl "http://localhost:5000/api/providers/search/hospital?hospital=Mayo%20Clinic&state=MN&limit=25"

# Hospitals by postal code (location search)
curl "http://localhost:5000/api/hospitals/search/location?postal_code=55401&limit=25"
```

---
**Notes**
- City and state comparisons are case-insensitive for city and normalized uppercase for state.
- Postal code matches on the prefix of the practice ZIP (first 5 digits).
