# Data Directory

This directory contains local CMS source assets, converted parquet files, enriched outputs, durable configuration, and aggregate analysis artifacts.

## Directories

- `raw/`: downloaded CMS CSV/ZIP assets
- `parquet/`: converted source parquet files
- `processed_data/`: enriched providers, organizations, rollups, baselines, and quality audits
- `analysis/`: aggregate output from `scripts/analyze_cms_baseline.py`
- `config/`: durable specialty and organization normalization inputs
- `hashes/`: local enrichment caches
- `meta/`: source schema manifests and processing metadata

Raw, parquet, processed, analysis, hash, and metadata outputs are ignored by Git.

## API Inputs

The default CMS API source loads:

- `processed_data/ppef_individuals_comparison_baseline_*.parquet`
- `processed_data/pecos_clinic_rollup_*.parquet`
- `processed_data/pecos_system_rollup_*.parquet`

The provider parquet is filtered to `current_active_primary` unless `CMS_BASELINE_VIEW` is changed.

The optional legacy source loads processed NPPES individual and organization parquet files, with raw split parquet as fallback.

## Main CMS Outputs

- `ppef_individuals_processed_YYYYMMDD.parquet`: enriched PPEF provider rows
- `ppef_individual_affiliation_links_YYYYMMDD.parquet`: scored provider-affiliation links
- `ppef_individual_affiliation_candidates_YYYYMMDD.parquet`: candidate affiliation graph
- `ppef_individuals_comparison_baseline_YYYYMMDD.parquet`: flattened baseline views
- `pecos_orgs_processed_YYYYMMDD.parquet`: enriched organization/location rows
- `pecos_clinic_rollup_YYYYMMDD.parquet`: canonical clinic entities
- `pecos_system_rollup_YYYYMMDD.parquet`: canonical system entities
- `ppef_pecos_mapping_quality_YYYYMMDD.csv`: aggregate mapping metrics
- `ppef_pecos_mapping_quality_systems_YYYYMMDD.csv`: system-level review priorities
- `ppef_specialty_rollup_audit_YYYYMMDD.csv`: normalized specialty coverage
- `ppef_unresolved_taxonomy_audit_YYYYMMDD.csv`: unresolved taxonomy review queue
- `ppef_pecos_field_provenance_summary_YYYYMMDD.csv`: field-source coverage
- `ppef_comparison_readiness_summary_YYYYMMDD.csv`: baseline readiness tiers

## Baseline Views

- `current_active_primary`: API-serving view with one active, high-confidence primary affiliation per provider
- `clean_primary`: strict primary-affiliation sensitivity view
- `all_mapped`: broad mapped-provider view
- `candidate_graph`: expanded candidate affiliations for analysis

## Configuration

- `config/specialty_rollup.csv`: CMS/NUCC taxonomy normalization
- `config/pecos_system_alias_overrides.csv`: reviewed system alias actions; the committed file is an empty template

Only approved alias rows affect organization processing.

## Retention

Do not commit downloaded or generated provider datasets. Commit only public-safe configuration, documentation, and intentionally small synthetic fixtures.
