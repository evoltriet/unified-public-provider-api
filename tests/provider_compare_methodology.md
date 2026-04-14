# Provider Comparison Methodology

## Datasets
- Test datasets:
  - `data/test_data/non_optum_clinic_open_scheduling_database_*.csv`
  - `data/test_data/optum_clinic_open_scheduling_database_*.csv`
- Input dataset:
  - any provider parquet or csv passed to the script CLI

## Provider Identity Matching
Providers are compared in two gates after hospital scoping.

1. Hospital scoping
   - input providers are first restricted to the mapped hospital subset for the active dataset

2. Multi-field retrieval

   Retrieval still runs inside the hospital-scoped input subset, but it is no longer purely name-led.

   - `name_strong`
   - exact normalized full-name key
   - exact normalized first-name-plus-last-name key with middle tokens removed
   - normalized last name plus first 3 characters of first name
   - normalized last name plus canonicalized first 3 characters of first name
   - normalized last name plus canonicalized first 4 characters of first name
   - exact token-set match on normalized first and last name

   - `name_plus_field`
   - normalized last name plus first initial, validated by specialty, phone, or ZIP/state
   - exact last name plus high first-name similarity, validated by specialty, phone, or ZIP/state

   - `field_led`
   - exact phone match
   - exact last name plus loose specialty plus ZIP/state
   - exact last name plus loose specialty inside the mapped hospital set

The `name_plus_field` and `field_led` buckets only run if `name_strong` does not already produce a match.

Field-led retrieval never uses specialty or ZIP/state alone. It still requires a strong anchor such as exact phone or exact last name.

If multiple retrieved candidates remain, the best candidate must beat the second-best candidate by a minimum score gap for weaker retrieval buckets or the test row is left unmatched.

If multiple input candidates remain in the chosen block, the script ranks them using:
- exact full-name bonus
- mapped-hospital bonus from the hospital crosswalk
- hospital-name overlap
- strict and loose specialty bonuses
- address-match and ZIP/state bonuses
- exact phone bonus
- reuse penalty when an input row has already been matched earlier in the run

If any candidates belong to the mapped hospital set for the test hospital, out-of-set candidates are dropped before ranking.

## Field Comparison Rules

### Hospital
- exact normalized equality first
- then canonicalized equality after removing generic organization words
- then token-overlap fallback for close naming variants

### Clinic
- exact normalized equality first
- then canonicalized equality
- then token-overlap fallback

`clinic_match` remains a diagnostic metric only. It does not affect `fully_matched`.

### Affiliation
- if the test row has a clinic name:
  - `affiliation_match = hospital_match and clinic_match`
- otherwise:
  - `affiliation_match = hospital_match`

`affiliation_match` remains a diagnostic metric only. It does not affect `fully_matched`.

### Specialty
- strict specialty match uses canonical specialty equality
- loose specialty match uses canonical equality or strong token overlap/containment

### Phone
- exact normalized digit equality

`phone_match` remains a diagnostic metric only. It does not affect `fully_matched`.

### Address
- exact normalized equality first
- else same state plus same ZIP5 plus street-token overlap of at least `0.5`
- else freeform token-overlap fallback of at least `0.6`

Input-side address precedence is:
- practice-location style fields first
- mapped facility address second
- organization or mailing style address fields last

The report includes both:
- full address match
- ZIP/state address match

## Classification Definitions
- `fully_matched`
  - provider appears in both sources and these fields match:
    - hospital
    - specialty using loose specialty matching
    - address
- `partially_matched`
  - provider appears in both sources, but one or more of those three fields differ or are missing
- `only_in_test`
  - provider appears only in the test dataset
- `only_in_input`
  - provider appears only in the input dataset

## Metric Definitions
The main `provider_compare_report.csv` keeps only these columns:
- `dataset`
- `scope_level`
- `hospital_name`
- `clinic_name`
- `test_provider_count`
- `input_provider_count_in_scope`
- `retrieval_reached_input_provider_count`
- `matched_provider_count`
- `only_in_test_count`
- `only_in_input_count`
- `union_provider_count`
- `fully_matched_pct`
- `partially_matched_pct`
- `only_in_test_pct`
- `only_in_input_pct`
- `hospital_match_pct`
- `clinic_match_pct`
- `affiliation_match_pct`
- `specialty_match_pct_loose`
- `phone_match_pct`
- `address_match_pct`

Metric notes:
- `input_provider_count_in_scope` is the gate-1 hospital-scoped unique input-provider count for that row
- `retrieval_reached_input_provider_count` is the gate-2 retrieval reach count:
  - overall rows use overall retrieval reach
  - hospital rows use hospital-level retrieval reach
  - clinic rows are blank unless specifically populated
- `matched_provider_count` is test-row based
- `only_in_input_count` is computed from the gate-1 hospital-scoped input set, not from the narrower retrieval subset
- `fully_matched_pct`, `partially_matched_pct`, `only_in_test_pct`, and `only_in_input_pct` use `union_provider_count` as denominator
- `hospital_match_pct`, `clinic_match_pct`, `affiliation_match_pct`, `specialty_match_pct_loose`, `phone_match_pct`, and `address_match_pct` use matched providers as denominator

Expanded analysis-only metrics are written to `provider_compare_report_diagnostics.csv`.
That diagnostics CSV includes the dropped counts, retrieval-bucket metrics, source-count comparisons, exact/relaxed/fallback breakdowns, reuse metrics, and stricter field diagnostics.

## Defined Hospital Subset
Hospital scoping is built in two steps:
- normalize and canonicalize hospital names on both sides
- build a hospital crosswalk using:
  - exact normalized or canonical matches
  - distinctive core-token overlap after generic and high-frequency hospital/system words are removed
  - sequence similarity on canonical and core-token names
- assign crosswalk confidence tiers:
  - `exact`
  - `strong`
  - `weak`
- never treat single-token core equality by itself as `exact`
- keep `exact` matches and only a capped number of `strong` matches per test hospital for scope
- only keep `strong` matches when they have at least one distinctive shared core token
- record weaker candidates in the audit CSV without using them for scope

Providers are in scope only if:
- they belong to a test hospital that maps to at least one input hospital, or
- they belong to an input hospital that maps back to one of those test hospitals

## Reproduction
Example command:

```bash
python tests/compare_provider_data.py data/processed_data/npi_individuals_processed_20251206.parquet --output-csv data/test_data/provider_compare_report.csv --output-diagnostics-csv data/test_data/provider_compare_report_diagnostics.csv --output-hospital-crosswalk-csv data/test_data/provider_compare_hospital_crosswalk.csv --output-retrieval-audit-csv data/test_data/provider_compare_retrieval_audit.csv
```

Additional audit outputs:
- `provider_compare_report_diagnostics.csv`
- `provider_compare_hospital_crosswalk.csv`
- `provider_compare_retrieval_audit.csv`
