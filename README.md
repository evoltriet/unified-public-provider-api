# NPI Registry API

Local Flask API and data-processing toolkit for working with CMS provider and organization data.

The repository has two parallel tracks:

- a current API/data path built on NPPES/NPI split and processed parquet files
- a newer parallel pipeline built around PPEF, PECOS-style org/location data, and HGI enrichment

The Flask API currently reads the legacy processed NPI artifacts. The newer PPEF/PECOS pipeline is available in `scripts/`, but it is not yet the API's primary input path.

## Overview

This repo is designed to:

- download and split NPPES/NPI data into individual and organization parquet files
- enrich those datasets with organization mapping, website mapping, and provider counts
- expose provider and hospital search endpoints through a Flask API
- experiment with improved clinic/system mapping via PPEF, PECOS-style org/location data, and HGI

The API loads:

- providers from `data/processed_data/npi_individuals_processed_*.parquet`
- hospitals from `data/processed_data/npi_organizations_processed_*.parquet`

If those files do not exist, it falls back to the corresponding raw split parquet files in `data/parquet/`.

## Repository Layout

```text
.
├── data/
│   ├── raw/              # downloaded CMS source files
│   ├── parquet/          # split / converted base parquet files
│   ├── processed_data/   # enriched parquet outputs used by the API and pipeline work
│   ├── hashes/           # caches and hash/state files used by enrichment scripts
│   └── meta/             # schema manifests and metadata
├── scripts/
│   ├── npi_registry_dump.py
│   ├── process_npi.py
│   ├── hgi_dump.py
│   ├── process_hgi.py
│   ├── ppef_dump.py
│   ├── pecos_dump.py
│   ├── process_individuals.py
│   └── process_orgs.py
├── src/
│   ├── api.py
│   ├── gunicorn_config.py
│   └── API_REFERENCE.md
├── tests/
└── requirements.txt
```

## Environment Setup

Use a Python 3.10+ environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
export DATA_DIR=data
export API_HOST=0.0.0.0
export API_PORT=5000
export FLASK_ENV=development
export MAX_RESULTS_DEFAULT=50
export MAX_RESULTS_LIMIT=500

# gunicorn-specific
export GUNICORN_BIND=0.0.0.0:5000
export GUNICORN_WORKERS=2
```

If you need taxonomy descriptions locally, generate or refresh the lookup file:

```bash
python scripts/create_specialty_lookup.py
```

## Quick Start

This is the shortest path to a working local API on top of the current NPI artifacts.

### 1. Download and split NPPES/NPI data

```bash
python scripts/npi_registry_dump.py
```

This produces split parquet files such as:

- `data/parquet/npi_individuals_YYYYMMDD.parquet`
- `data/parquet/npi_organizations_YYYYMMDD.parquet`

### 2. Build the processed API inputs

Run `process_npi.py` and use the interactive modes you need.

At minimum, the useful modes are:

- `entity_mapping`
- `website_mapping_osm` or `website_mapping_overpass`
- `provider_count`

```bash
python scripts/process_npi.py
```

This produces:

- `data/processed_data/npi_individuals_processed_YYYYMMDD.parquet`
- `data/processed_data/npi_organizations_processed_YYYYMMDD.parquet`

### 3. Start the API

```bash
python src/api.py
```

Or with gunicorn:

```bash
gunicorn -c src/gunicorn_config.py src.api:app
```

## Data Pipelines

### A. Current API pipeline

This is the path the API uses today.

1. `python scripts/npi_registry_dump.py`
2. `python scripts/process_npi.py`

Key `process_npi.py` modes:

- `entity_mapping`
  - maps individuals to organizations
  - writes `mapped_org_npi` and `mapped_org_name` on the processed individuals dataset
- `website_mapping_osm`
  - maps hospital orgs to websites using OSM/Nominatim
- `website_mapping_overpass`
  - maps hospital orgs to websites using the cached Overpass hospital dataset
- `provider_count`
  - adds `provider_count` and `provider_count_entity` to the processed organizations dataset

Main outputs:

- `npi_individuals_processed_YYYYMMDD.parquet`
- `npi_organizations_processed_YYYYMMDD.parquet`

### B. New parallel PPEF / PECOS / HGI pipeline

This pipeline is available for improved clinic and hospital-system mapping, but it is not yet wired into `src/api.py` as the primary input source.

Core scripts:

- `scripts/ppef_dump.py`
- `scripts/pecos_dump.py`
- `scripts/process_individuals.py`
- `scripts/process_orgs.py`
- `scripts/hgi_dump.py`
- `scripts/process_hgi.py`

Recommended run order:

```bash
python scripts/pecos_dump.py
python scripts/ppef_dump.py
python scripts/process_orgs.py npi_enrichment
python scripts/process_orgs.py hgi_enrichment
python scripts/process_individuals.py all
python scripts/process_orgs.py provider_count
python scripts/process_orgs.py website_mapping
```

What this parallel pipeline produces:

- `pecos_orgs_YYYYMMDD.parquet`
- `pecos_orgs_processed_YYYYMMDD.parquet`
- `ppef_individuals_YYYYMMDD.parquet`
- `ppef_individuals_processed_YYYYMMDD.parquet`
- `ppef_individual_affiliation_links_YYYYMMDD.parquet`
- `hgi_processed_YYYYMMDD.parquet`

## Run the API

Development:

```bash
python src/api.py
```

Production-style:

```bash
gunicorn -c src/gunicorn_config.py src.api:app
```

The API exposes:

- health check
- taxonomy keyword/code lookup
- provider lookup by NPI
- provider search by name, specialty, location, state, ZIP, and hospital
- hospital search by name and location

See [src/API_REFERENCE.md](src/API_REFERENCE.md) for the endpoint catalog and example requests.

## Data Artifacts

### Raw and intermediate

- `data/raw/`
  - downloaded CMS source files and extracted CSVs
- `data/parquet/npi_individuals_*.parquet`
  - split raw NPI individual provider records
- `data/parquet/npi_organizations_*.parquet`
  - split raw NPI organization records
- `data/parquet/hgi_*.parquet`
  - HGI hospital base data
- `data/parquet/ppef_individuals_*.parquet`
  - PPEF-derived individual-provider base rows
- `data/parquet/pecos_orgs_*.parquet`
  - PECOS-style org/location base rows built from public PPEF enrollment assets

### Processed outputs

- `data/processed_data/npi_individuals_processed_*.parquet`
  - processed NPI individual rows used by the API
- `data/processed_data/npi_organizations_processed_*.parquet`
  - processed NPI organization rows used by the API
- `data/processed_data/hgi_processed_*.parquet`
  - HGI rows enriched with website/provider-count data
- `data/processed_data/ppef_individuals_processed_*.parquet`
  - processed PPEF individual rows with best mapped clinic/system fields
- `data/processed_data/ppef_individual_affiliation_links_*.parquet`
  - scored affiliation link table for all candidate individual-to-clinic/system relationships
- `data/processed_data/pecos_orgs_processed_*.parquet`
  - processed clinic/system org dataset derived from the PECOS-style org/location base

### State and metadata

- `data/hashes/`
  - OSM and related caches / hash files used by enrichment scripts
- `data/meta/`
  - schema manifests for downloaded and converted source files
- `data/processed_data/*_metadata.json`
  - per-run processing metadata and completed-mode tracking

## Notes and Current Limitations

- The Flask API currently reads the legacy processed NPI datasets, not the newer PPEF/PECOS outputs.
- The PPEF/PECOS pipeline is parallel work intended to improve clinic and hospital-system mapping quality before any downstream cutover.
- Several scripts are interactive or mode-driven instead of fully declarative CLI tools.
- CMS public source schemas can drift. The dump/process scripts include best-effort schema handling, but new source releases may still require adjustment.
- The newer `process_individuals.py` and `process_orgs.py` pipelines are implemented as evolving local workflows and should be validated against fresh source extracts before treating them as stable production inputs.
