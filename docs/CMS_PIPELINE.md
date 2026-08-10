# CMS Provider Pipeline

## Purpose

The pipeline creates a provider-centric CMS baseline with normalized specialties, practice locations, and clinic/system affiliations. It combines public enrollment, reassignment, practice-location, and NPI Registry records while preserving source dates and mapping confidence.

## Sources

### Public Provider Enrollment File

The enrollment extract supplies individual and organization enrollment rows. Reassignment and practice-location assets provide evidence for provider-to-organization relationships and current practice geography.

### PECOS-Style Organizations

`pecos_dump.py` derives organization/location records from the same public enrollment distribution. `process_orgs.py` turns those rows into canonical clinic and system entities.

### NPI Registry

NPI Registry data supplies stable NPIs, provider names, taxonomy codes, practice contact fields, and organization enrichment.

## Processing Order

```bash
python scripts/npi_registry_dump.py
python scripts/ppef_dump.py
python scripts/pecos_dump.py
python scripts/process_orgs.py npi_enrichment
python scripts/process_individuals.py npi_enrichment
python scripts/process_individuals.py clinic_mapping
python scripts/process_individuals.py quality_exports
python scripts/process_orgs.py provider_count
```

The dump scripts are interactive. They can download fresh assets or rebuild from existing enrollment parquet files.

## CMS Catalog Resolution

`cms_enrollment_common.py` queries the CMS `data.json` catalog, selects the latest matching distribution, classifies enrollment/reassignment/practice-location assets, and records schema manifests. CSV conversion attempts UTF-8, UTF-8 with BOM, Windows-1252, and Latin-1 encodings.

## Organization Model

Organizations are represented at two levels:

- clinic: a practice/location entity with a canonical clinic ID
- system: a broader organization/system entity with a canonical system ID

Rollup selection prefers complete representative rows and records:

- source row and NPI counts
- mapped provider counts
- generic-name warnings
- duplicate-address cluster sizes
- website provenance and confidence when available
- reviewed alias actions

Alias overrides are optional. Only rows marked `approved` are allowed to change canonical system behavior.

## Individual Model

Provider identity is anchored by PPEF enrollment records and enriched with NPI Registry fields. The pipeline resolves primary taxonomy, normalized specialty, specialty group, provider type, practice address, and phone provenance.

Affiliation candidates are built from reassignment relationships and scored with signals including:

- active relationship status
- relationship recency and continuity
- state and ZIP agreement
- city/state agreement
- address-line agreement
- phone agreement and phone rarity
- repeated relationship evidence
- organization-name evidence
- penalties for blocked generic systems

The selected primary row retains its score, margin, signal count, confidence tier, tie status, ambiguity status, relationship dates, and source provenance.

## Current-Active Definition

`current_active_primary` includes one provider row when the selected relationship is active, the primary mapping is not ambiguous, and mapping confidence is high or medium. The row records the CMS active-as-of date and the reason it qualified.

This is an administrative currentness signal. It should not be interpreted as evidence that the provider is accepting patients or has appointment availability.

## Baseline Views

The baseline parquet materializes:

- `current_active_primary` for API serving and current-active reporting
- `clean_primary` for strict primary-mapping sensitivity analysis
- `all_mapped` for broad source coverage
- `candidate_graph` for expanded affiliation analysis

Only one-row-per-provider views are suitable for the API.

## Quality Outputs

Quality exports include mapping coverage, field provenance, readiness distributions, specialty rollup coverage, unresolved taxonomies, system-level review priorities, and alias summaries.

Run aggregate analysis with:

```bash
python scripts/analyze_cms_baseline.py --baseline-view current_active_primary
```

The analyzer emits only aggregate counts and completeness statistics.

## Operational Checks

Before serving a new snapshot:

1. Confirm the selected baseline view is nonempty and unique by provider ID and NPI.
2. Review unresolved taxonomy volume and high-priority system quality rows.
3. Confirm clinic and system rollups have unique IDs.
4. Confirm snapshot dates appear in the API health response.
5. Run API and CMS catalog tests.
6. Inspect generated schema manifests for source drift.
