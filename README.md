# Unified Public Provider API

A Flask API and public-data pipeline for building searchable provider and hospital datasets from CMS NPI/NPPES files, with parallel work toward stronger clinic and health-system mapping.

## Overview

This repository has two connected goals:

- turn large, messy public provider files into parquet datasets that are practical to query and enrich
- expose those datasets through a lightweight API for provider and hospital search

Today, the API serves the processed NPI/NPPES path. In parallel, the repo also contains a newer PPEF/PECOS/HGI pipeline intended to improve organization, clinic, and system mapping before any future API cutover.

## Problem

Public provider data is useful, but it is not easy to work with directly.

The raw sources are:

- large
- schema-drifting over time
- split across multiple public systems
- weak on organization normalization and clinic/system relationships
- not immediately usable for application-style search

If the goal is to answer questions such as:

- "Which cardiologists are near this city?"
- "Which providers are affiliated with this hospital?"
- "Which hospital locations belong to this organization?"
- "What website or system homepage is associated with this hospital?"

then raw CMS files alone are not enough. They need to be downloaded, split, normalized, enriched, and indexed into a search-friendly representation.

## Approach

The repo is structured around two tracks.

### 1. Current API track: NPI/NPPES -> processed parquet -> Flask API

This is the path the API uses today.

- `scripts/npi_registry_dump.py` downloads the official NPPES dissemination file from CMS and splits it into separate individual and organization parquet files.
- `scripts/process_npi.py` enriches those parquet files with fields such as entity mapping, website mapping, provider counts, geocoding, and metadata tracking.
- `src/api.py` loads the newest processed individual and organization parquet files and exposes search endpoints for providers and hospitals.

The API prefers:

- `data/processed_data/npi_individuals_processed_*.parquet`
- `data/processed_data/npi_organizations_processed_*.parquet`

and falls back to the corresponding raw split parquet files under `data/parquet/` if the processed versions are not available.

### 2. Parallel enrichment track: PPEF / PECOS / HGI

This is the newer mapping-oriented workflow in `scripts/`, designed to improve affiliation quality and organization modeling.

- `scripts/ppef_dump.py` and `scripts/pecos_dump.py` pull newer public enrollment-style source data
- `scripts/process_individuals.py` builds enriched individual rows and scored affiliation links
- `scripts/process_orgs.py` builds clinic/system-style organization entities and enriches them with provider counts and website signals
- `scripts/hgi_dump.py` and `scripts/process_hgi.py` bring in hospital/system enrichment data

This path is important architecturally, but it is not yet the primary input source for the Flask API.

## Outcome

The repo already delivers a working local API on top of processed public provider data.

Current outcome:

- provider search by location, specialty, state, ZIP, hospital, and provider name
- provider lookup by NPI
- hospital search by location and hospital name
- taxonomy keyword/code lookup
- health endpoint showing dataset load state

Pipeline outcome:

- raw CMS downloads stored under `data/raw/`
- split parquet outputs under `data/parquet/`
- enriched API-ready artifacts under `data/processed_data/`
- metadata and caches that support resumable processing and enrichment workflows

In short, the project is already useful as a searchable public-provider API, while also serving as a staging ground for better clinic/system mapping logic.

## Repository Layout

```text
unified-public-provider-api/
  data/
    raw/              downloaded source files
    parquet/          split and converted parquet datasets
    processed_data/   enriched outputs used by the API and mapping workflows
    hashes/           processing caches and hash/state files
    meta/             schema and processing metadata
  scripts/
    npi_registry_dump.py
    process_npi.py
    ppef_dump.py
    pecos_dump.py
    process_individuals.py
    process_orgs.py
    hgi_dump.py
    process_hgi.py
  src/
    api.py
    API_REFERENCE.md
    gunicorn_config.py
  tests/
    test_api.py
    compare_provider_data.py
    provider_compare_methodology.md
  requirements.txt
```

## API Surface

The Flask API exposes:

- `GET /api/health`
- `GET /api/taxonomy/codes`
- `GET /api/providers/<npi>`
- `GET /api/providers/search/location`
- `GET /api/providers/search/specialty`
- `GET /api/providers/search/state/<state_code>`
- `GET /api/providers/search/postal_code/<postal_code>`
- `GET /api/providers/search/hospital`
- `GET /api/providers/search/name`
- `GET /api/hospitals/search/location`
- `GET /api/hospitals/search/name`

See [src/API_REFERENCE.md](src/API_REFERENCE.md) for endpoint-level examples and response shapes.

## Quick Start

Use Python 3.10+.

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
export GUNICORN_BIND=0.0.0.0:5000
export GUNICORN_WORKERS=2
```

If taxonomy descriptions are missing, generate them locally:

```bash
python scripts/create_specialty_lookup.py
```

## Current API Workflow

### 1. Download and split NPI/NPPES data

```bash
python scripts/npi_registry_dump.py
```

This creates split parquet files such as:

- `data/parquet/npi_individuals_YYYYMMDD.parquet`
- `data/parquet/npi_organizations_YYYYMMDD.parquet`

### 2. Build processed API inputs

```bash
python scripts/process_npi.py
```

Useful modes in the current workflow include:

- `entity_mapping`
- `website_mapping_osm` or `website_mapping_overpass`
- `provider_count`

That step produces the processed parquet files the API prefers to load.

### 3. Start the API

Development:

```bash
python src/api.py
```

Production-style:

```bash
gunicorn -c src/gunicorn_config.py src.api:app
```

`src/gunicorn_config.py` is tuned for large in-memory datasets and loads provider data per worker.

## Parallel Mapping Workflow

For the newer PPEF/PECOS/HGI path, the recommended sequence is:

```bash
python scripts/pecos_dump.py
python scripts/ppef_dump.py
python scripts/process_orgs.py npi_enrichment
python scripts/process_orgs.py hgi_enrichment
python scripts/process_individuals.py all
python scripts/process_orgs.py provider_count
python scripts/process_orgs.py website_mapping
```

This path produces artifacts such as:

- `pecos_orgs_processed_YYYYMMDD.parquet`
- `ppef_individuals_processed_YYYYMMDD.parquet`
- `ppef_individual_affiliation_links_YYYYMMDD.parquet`
- `hgi_processed_YYYYMMDD.parquet`

## Data Quality and Evaluation

The repo also includes evaluation utilities in `tests/`:

- `tests/test_api.py` exercises the API endpoints against a running local server
- `tests/compare_provider_data.py` compares provider outputs against curated datasets
- `tests/provider_compare_methodology.md` documents the matching, scoping, and scoring methodology used for those comparisons

That makes the project more than just an API wrapper. It also includes tooling for validating retrieval quality and affiliation quality as the pipeline evolves.

## Current Limitations

- The Flask API currently serves the legacy processed NPI/NPPES datasets, not the newer PPEF/PECOS outputs.
- Several pipeline scripts are interactive or mode-driven, so the full workflow is not yet a fully declarative batch pipeline.
- Public CMS schemas can drift, and fresh source releases may require adjustments in the dump or processing scripts.
- The API reads large parquet datasets into memory, which is simple and fast for local use but not a full distributed serving architecture.
- The parallel PPEF/PECOS/HGI workflow is clearly in progress and should be treated as an evolving enrichment path rather than a finalized production cutover.

## Why This Repo Is Interesting

This repo sits at the intersection of data engineering, public-data normalization, and applied API design.

It is not just a Flask service. It is a full workflow for:

- acquiring public healthcare provider data
- reshaping it into analysis- and application-friendly formats
- enriching it with hospital, website, and affiliation signals
- exposing it through queryable endpoints
- measuring how well the resulting mappings hold up against reference datasets

That makes it a strong foundation for provider search, open-scheduling discovery, clinic/system resolution, and future healthcare-directory products built on public data.
