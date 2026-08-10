# Unified Public Provider API

A Flask API and reproducible data pipeline for building a searchable provider, clinic, and health-system dataset from public CMS sources.

## Overview

Raw provider data is distributed across multiple CMS files with different identifiers, update schedules, and organization semantics. This repository turns those files into a unified CMS baseline by combining:

- Medicare Fee-For-Service Public Provider Enrollment data for provider enrollment and reassignment relationships
- PECOS-style organization and practice-location records derived from the public enrollment assets
- NPI Registry data for provider identity, taxonomy, address, and phone enrichment

The resulting baseline supports current-active provider analysis, clinic and system affiliation mapping, aggregate quality review, and API search.

## Architecture

```text
CMS enrollment assets       NPI Registry
        |                         |
        v                         v
 PPEF individuals          NPI parquet splits
 PECOS organizations              |
        |                         |
        +-----------+-------------+
                    |
                    v
       Enriched organizations and providers
                    |
        +-----------+------------+
        |                        |
        v                        v
 CMS comparison baseline   Clinic/system rollups
        |                        |
        +-----------+------------+
                    |
                    v
              Flask API
```

The API defaults to the `current_active_primary` provider view and canonical PECOS clinic/system rollups. The older NPPES-only API path remains available as a fallback.

## Repository Layout

```text
data/
  config/                 reviewed, public-safe normalization inputs
  raw/                    downloaded source files, ignored by Git
  parquet/                converted source parquet, ignored by Git
  processed_data/         enriched outputs, ignored by Git
  analysis/               aggregate analysis outputs, ignored by Git
scripts/
  cms_enrollment_common.py
  ppef_dump.py
  pecos_dump.py
  process_individuals.py
  process_orgs.py
  analyze_cms_baseline.py
src/
  api.py
  data_store.py
  API_REFERENCE.md
tests/
docs/
```

## Environment Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Build The CMS Data

### 1. Download NPI Registry Data

```bash
python scripts/npi_registry_dump.py
```

This creates individual and organization parquet files under `data/parquet/`.

### 2. Download CMS Enrollment Assets

```bash
python scripts/ppef_dump.py
python scripts/pecos_dump.py
```

The dump scripts resolve the current CMS catalog distribution, download enrollment/reassignment/practice-location assets, handle common CSV encodings, and convert them to parquet.

### 3. Process Organizations

```bash
python scripts/process_orgs.py npi_enrichment
```

Organization processing:

- enriches PECOS rows with NPI organization fields
- creates deterministic clinic and system identifiers
- canonicalizes names and locations
- writes clinic and system rollups
- applies approved alias controls when configured
- adds generic-name, duplicate-address, and rollup-quality diagnostics

Optional organization enrichment modes include `hgi_enrichment`, `provider_count`, and `website_mapping`.

### 4. Process Individuals

```bash
python scripts/process_individuals.py npi_enrichment
python scripts/process_individuals.py clinic_mapping
python scripts/process_individuals.py quality_exports
```

Individual processing:

- enriches PPEF identities and contact fields with NPI Registry data
- normalizes taxonomy into specialty and provider-type groups
- builds scored provider-to-clinic/system affiliation candidates
- selects a primary affiliation using active relationships, recency, continuity, and corroborating location signals
- records confidence, ambiguity, provenance, and currentness metadata
- writes analysis-ready baseline views and quality audits

Run `python scripts/process_individuals.py all` when all prerequisites are already available.

### 5. Add Provider Counts To Organizations

```bash
python scripts/process_orgs.py provider_count
```

This counts unique mapped providers at clinic and system levels and refreshes the rollups.

See [docs/CMS_PIPELINE.md](docs/CMS_PIPELINE.md) for detailed artifact and sequencing notes.

## Baseline Views

`ppef_individuals_comparison_baseline_YYYYMMDD.parquet` contains several explicit views:

- `current_active_primary`: one high-confidence, non-ambiguous active primary affiliation per provider; default API view
- `clean_primary`: strict primary-affiliation sensitivity view
- `all_mapped`: broader mapped-provider coverage view
- `candidate_graph`: expanded provider-affiliation candidates for analytical use, not API serving

The API requires its selected view to contain one nonblank provider ID and NPI per row with no duplicates.

## Run The API

Development:

```bash
python src/api.py
```

Production-style:

```bash
gunicorn -c src/gunicorn_config.py src.api:app
```

Common endpoints:

```text
GET /api/health
GET /api/providers/{npi}
GET /api/providers/search/name
GET /api/providers/search/specialty
GET /api/providers/search/location
GET /api/providers/search/hospital
GET /api/clinics/search/name
GET /api/clinics/search/location
GET /api/systems/search/name
GET /api/systems/search/location
GET /api/hospitals/search/name
GET /api/hospitals/search/location
```

See [src/API_REFERENCE.md](src/API_REFERENCE.md) for query parameters and response examples.

### API Configuration

```bash
export DATA_DIR=data
export API_DATA_SOURCE=cms
export CMS_BASELINE_VIEW=current_active_primary
export ALLOW_LEGACY_NPI_FALLBACK=true
export API_HOST=0.0.0.0
export API_PORT=5000
export MAX_RESULTS_DEFAULT=50
export MAX_RESULTS_LIMIT=500
```

Optional explicit file overrides:

- `CMS_PROVIDER_PATH`
- `CMS_CLINIC_PATH`
- `CMS_SYSTEM_PATH`

Set `API_DATA_SOURCE=legacy_npi` to use the older processed NPPES files directly.

## Aggregate CMS Analysis

Generate aggregate, non-row-level baseline analysis with:

```bash
python scripts/analyze_cms_baseline.py
```

The analyzer writes ignored artifacts under `data/analysis/`:

- baseline summary in JSON and Markdown
- provider counts and completeness by state
- high-volume specialty groups
- high-volume mapped systems
- organization rollup quality totals

No row-level provider data is emitted by this analyzer.

## Durable Configuration

`data/config/specialty_rollup.csv` maps CMS/NUCC taxonomy descriptions and codes into normalized specialties, specialty groups, and provider-type groups.

`data/config/pecos_system_alias_overrides.csv` is an intentionally empty template. Add reviewed overrides locally and commit only decisions suitable for public release. Supported actions are implemented by `process_orgs.py`; unapproved rows do not affect canonical mappings.

## Tests

```bash
python -m pytest tests/test_api.py tests/test_cms_enrollment_common.py
python -m py_compile src/*.py scripts/*.py
```

The API tests use synthetic parquet fixtures and Flask's test client. CMS download tests mock catalog responses and do not require production data.

## Data And Git Hygiene

CMS source files and generated parquet datasets can be large. The repository intentionally ignores:

- downloaded raw assets
- converted and processed parquet files
- generated quality and analysis outputs
- caches, hashes, logs, and local environment files

Commit source code, documentation, public-safe configuration, and small synthetic fixtures only.

## Limitations

- CMS source schemas and distribution URLs can change; catalog discovery and schema warnings reduce but do not eliminate maintenance.
- Enrollment and reassignment relationships are administrative signals, not a guarantee that a provider is currently accepting patients at a location.
- Clinic and system mapping is probabilistic when authoritative identifiers are absent. Confidence and ambiguity fields should remain visible to downstream users.
- Organization websites are optional enrichments and may not be available for every rollup.
- The in-memory Flask implementation is intended for local and moderate-scale use. High-concurrency deployments should materialize search indexes or use a database/search service.
