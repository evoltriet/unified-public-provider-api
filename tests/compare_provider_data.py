#!/usr/bin/env python3
"""
Compare a provider dataset against curated test datasets.

Example:
    python tests/compare_provider_data.py data/processed_data/npi_individuals_processed_20251206.parquet

The script compares the input provider dataset against:
 - data/test_data/non_optum_clinic_open_scheduling_database_*.csv
 - data/test_data/optum_clinic_open_scheduling_database_*.csv

It writes a combined CSV with:
 - separate metrics for non_optum and optum_clinic
 - combined metrics across both datasets
 - overall stats plus per-hospital and per-clinic breakdowns
"""

from __future__ import annotations

import argparse
import ast
from difflib import SequenceMatcher
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

MIN_PYTHON = (3, 9)

if sys.version_info < MIN_PYTHON:
    version = ".".join(str(part) for part in sys.version_info[:3])
    required = ".".join(str(part) for part in MIN_PYTHON)
    raise SystemExit(
        f"compare_provider_data.py requires Python {required}+ because this repo pins "
        f"pandas/pyarrow versions that do not support Python {version}. "
        f"Use a newer interpreter, for example `python3`."
    )

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "compare_provider_data.py requires pandas. Install the repo dependencies in the "
        "Python interpreter you are using, for example `python3 -m pip install -r requirements.txt`."
    ) from exc


def _log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_text(value: Any) -> str:
    text = _safe_str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


HOSPITAL_NAME_STOPWORDS = {
    "and",
    "at",
    "care",
    "center",
    "centers",
    "centre",
    "clinic",
    "clinics",
    "company",
    "corp",
    "corporation",
    "for",
    "group",
    "health",
    "healthcare",
    "hospital",
    "hospitals",
    "inc",
    "llc",
    "ltd",
    "medical",
    "medicine",
    "network",
    "of",
    "pa",
    "pc",
    "physician",
    "physicians",
    "pllc",
    "service",
    "services",
    "system",
    "systems",
    "the",
}

HOSPITAL_GENERIC_TOKENS = {
    "associate",
    "associates",
    "alliance",
    "care",
    "center",
    "centers",
    "children",
    "clinic",
    "clinics",
    "community",
    "general",
    "health",
    "healthcare",
    "hospital",
    "hospitals",
    "medical",
    "memorial",
    "partner",
    "partners",
    "physician",
    "physicians",
    "premier",
    "regional",
    "group",
    "groups",
    "system",
    "systems",
    "university",
}

HOSPITAL_TOKEN_FREQUENCY_LIMIT = 40

PLACEHOLDER_CLINIC_NAMES = {
    "accepts new patients",
    "new patients",
}

SPECIALTY_REPLACEMENTS = {
    "orthopaedic": "orthopedic",
    "internal med": "internal medicine",
    "family practice": "family medicine",
    "general practice": "family medicine",
    "ob gyn": "obstetrics gynecology",
    "peds": "pediatrics",
    "ent": "otolaryngology",
}

PERSON_NAME_SUFFIXES = {
    "agacnp",
    "apnp",
    "aprn",
    "cnm",
    "cns",
    "crna",
    "dc",
    "dmd",
    "dnp",
    "do",
    "dds",
    "edd",
    "esq",
    "facp",
    "facs",
    "fnp",
    "ii",
    "iii",
    "iv",
    "jd",
    "jr",
    "lcsw",
    "ma",
    "mba",
    "md",
    "mhs",
    "mhsa",
    "mph",
    "ms",
    "msn",
    "msw",
    "np",
    "od",
    "pa",
    "pac",
    "phd",
    "psyd",
    "rn",
    "rnfa",
    "sr",
}

PERSON_FIRST_NAME_CANONICAL = {
    "abby": "abigail",
    "abbie": "abigail",
    "alex": "alexander",
    "andy": "andrew",
    "ben": "benjamin",
    "beth": "elizabeth",
    "bill": "william",
    "billy": "william",
    "bob": "robert",
    "bobby": "robert",
    "cathy": "catherine",
    "chris": "christopher",
    "dan": "daniel",
    "danny": "daniel",
    "dave": "david",
    "deb": "deborah",
    "debbie": "deborah",
    "jen": "jennifer",
    "jenny": "jennifer",
    "jim": "james",
    "jimmy": "james",
    "joe": "joseph",
    "jon": "jonathan",
    "kate": "katherine",
    "katie": "katherine",
    "kathy": "katherine",
    "liz": "elizabeth",
    "lizzy": "elizabeth",
    "maggie": "margaret",
    "matt": "matthew",
    "mike": "michael",
    "nick": "nicholas",
    "pat": "patrick",
    "patty": "patricia",
    "peggy": "margaret",
    "rick": "richard",
    "rob": "robert",
    "robbie": "robert",
    "ron": "ronald",
    "sam": "samuel",
    "steve": "steven",
    "sue": "susan",
    "susie": "susan",
    "tom": "thomas",
    "tommy": "thomas",
    "tony": "anthony",
    "vicki": "victoria",
    "vicky": "victoria",
    "will": "william",
}


def _canonical_hospital_name(value: Any) -> str:
    tokens = [token for token in _normalize_text(value).split() if token and token not in HOSPITAL_NAME_STOPWORDS]
    return " ".join(tokens)


def _canonical_clinic_name(value: Any) -> str:
    return _canonical_hospital_name(value)


def _hospital_core_tokens(value: Any) -> list[str]:
    if isinstance(value, str) and " " in value and value == value.strip():
        tokens = [token for token in value.split() if token]
    else:
        tokens = [token for token in _canonical_hospital_name(value).split() if token]
    return [token for token in tokens if token not in HOSPITAL_GENERIC_TOKENS and len(token) >= 3]


def _hospital_core_name(value: Any) -> str:
    return " ".join(_hospital_core_tokens(value))


def _build_hospital_core_token_frequencies(canonical_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for canonical in canonical_names:
        for token in set(_hospital_core_tokens(canonical)):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _is_distinctive_hospital_token(token: str, token_frequencies: dict[str, int]) -> bool:
    return bool(
        token
        and token not in HOSPITAL_GENERIC_TOKENS
        and len(token) >= 4
        and token_frequencies.get(token, 0) <= HOSPITAL_TOKEN_FREQUENCY_LIMIT
    )


def _zip5(value: Any) -> str:
    match = re.search(r"(\d{5})", _safe_str(value))
    return match.group(1) if match else ""


def _extract_state(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    match = re.search(r"\b([A-Z]{2})\b(?:\s+\d{5}(?:-\d{4})?)?\s*$", text)
    if match:
        return match.group(1).lower()
    return ""


def _address_tokens(value: Any) -> set[str]:
    return {token for token in _normalize_text(value).split() if token}


def _address_parts(value: Any) -> dict[str, str]:
    text = _safe_str(value)
    return {
        "normalized": _normalize_text(text),
        "zip5": _zip5(text),
        "state": _extract_state(text),
    }


def _canonical_specialty_name(value: Any) -> str:
    text = _normalize_text(value)
    for source, target in SPECIALTY_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    text = re.sub(r"\b(physician|doctor|dept|department|specialist|specialty)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _specialty_match_loose(left: Any, right: Any) -> bool:
    left_canonical = _canonical_specialty_name(left)
    right_canonical = _canonical_specialty_name(right)
    if not left_canonical or not right_canonical:
        return False
    if left_canonical == right_canonical:
        return True
    left_tokens = set(left_canonical.split())
    right_tokens = set(right_canonical.split())
    overlap = _token_overlap_score(left_tokens, right_tokens)
    return (
        (left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens))
        or (overlap >= 0.67 and len(left_tokens & right_tokens) >= 1)
    )


def _address_match_zip_state(test_address: Any, input_address: Any) -> bool:
    test_parts = _address_parts(test_address)
    input_parts = _address_parts(input_address)
    return bool(
        test_parts["zip5"]
        and input_parts["zip5"]
        and test_parts["zip5"] == input_parts["zip5"]
        and test_parts["state"]
        and input_parts["state"]
        and test_parts["state"] == input_parts["state"]
    )


def _hospital_names_match(
    test_hospital_name: Any,
    input_hospital_name: Any,
    test_hospital_canonical: Optional[str] = None,
    input_hospital_canonical: Optional[str] = None,
) -> bool:
    test_norm = _normalize_text(test_hospital_name)
    input_norm = _normalize_text(input_hospital_name)
    if test_norm and test_norm == input_norm:
        return True

    test_canonical = test_hospital_canonical if test_hospital_canonical is not None else _canonical_hospital_name(test_hospital_name)
    input_canonical = input_hospital_canonical if input_hospital_canonical is not None else _canonical_hospital_name(input_hospital_name)
    if test_canonical and input_canonical and test_canonical == input_canonical:
        return True

    test_tokens = set(test_canonical.split()) if test_canonical else set()
    input_tokens = set(input_canonical.split()) if input_canonical else set()
    if not test_tokens or not input_tokens:
        return False
    overlap = len(test_tokens & input_tokens) / max(1, len(test_tokens | input_tokens))
    if len(test_tokens & input_tokens) >= 2 and overlap >= 0.5:
        return True
    return False


def _clinic_names_match(
    test_clinic_name: Any,
    input_clinic_name: Any,
    test_clinic_canonical: Optional[str] = None,
    input_clinic_canonical: Optional[str] = None,
) -> bool:
    test_norm = _normalize_text(test_clinic_name)
    input_norm = _normalize_text(input_clinic_name)
    if not test_norm or not input_norm:
        return False
    if test_norm == input_norm:
        return True

    test_canonical = test_clinic_canonical if test_clinic_canonical is not None else _canonical_clinic_name(test_clinic_name)
    input_canonical = input_clinic_canonical if input_clinic_canonical is not None else _canonical_clinic_name(input_clinic_name)
    if test_canonical and input_canonical and test_canonical == input_canonical:
        return True

    overlap = _overlap_score(test_clinic_name, input_clinic_name)
    return overlap >= 0.5


def _address_match(test_address: Any, input_address: Any) -> bool:
    test_parts = _address_parts(test_address)
    input_parts = _address_parts(input_address)
    test_norm = test_parts["normalized"]
    input_norm = input_parts["normalized"]
    if not test_norm or not input_norm:
        return False
    if test_norm == input_norm:
        return True

    test_zip = test_parts["zip5"]
    input_zip = input_parts["zip5"]
    test_state = test_parts["state"]
    input_state = input_parts["state"]
    if test_zip and input_zip and test_zip == input_zip and test_state and input_state and test_state == input_state:
        return _overlap_score(test_address, input_address) >= 0.5

    return _overlap_score(test_address, input_address) >= 0.6


def _normalize_person_name(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    for pattern, replacement in [
        (r"\bpa c\b", "pac"),
        (r"\ba p r n\b", "aprn"),
        (r"\bf n p\b", "fnp"),
    ]:
        text = re.sub(pattern, replacement, text)
    parts = [part for part in text.split() if part and part not in PERSON_NAME_SUFFIXES]
    return " ".join(parts)


def _canonical_first_name(value: Any) -> str:
    token = _safe_str(value).strip().lower()
    return PERSON_FIRST_NAME_CANONICAL.get(token, token)


def _person_name_features(value: Any) -> tuple[str, str, str, str, str, str, str, str, str]:
    normalized = _normalize_person_name(value)
    if not normalized:
        return "", "", "", "", "", "", "", "", ""
    parts = [part for part in normalized.split() if part]
    if not parts:
        return "", "", "", "", "", "", "", "", ""
    first_name = parts[0]
    last_name = parts[-1]
    first3_name = first_name[:3]
    first4_name = first_name[:4]
    first_initial = first_name[:1]
    canonical_first_name = _canonical_first_name(first_name)
    canonical_first3_name = canonical_first_name[:3]
    canonical_first4_name = canonical_first_name[:4]
    no_middle_name = " ".join([parts[0], parts[-1]]) if len(parts) > 1 else parts[0]
    token_set_name_key = " ".join(sorted({token for token in no_middle_name.split() if token}))
    return (
        first_name,
        last_name,
        first3_name,
        first4_name,
        first_initial,
        canonical_first_name,
        canonical_first3_name,
        canonical_first4_name,
        token_set_name_key,
    )


def _split_name_parts(value: Any) -> tuple[str, str, str]:
    features = _person_name_features(value)
    return features[0], features[1], features[2]


def _name_without_middle_key(value: Any) -> str:
    normalized = _normalize_person_name(value)
    if not normalized:
        return ""
    parts = [part for part in normalized.split() if part]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1]}"


def _first_name_similarity(left: Any, right: Any) -> float:
    left_name = _canonical_first_name(_safe_str(left))
    right_name = _canonical_first_name(_safe_str(right))
    if not left_name or not right_name:
        return 0.0
    if left_name == right_name:
        return 1.0
    if len(left_name) >= 4 and len(right_name) >= 4 and (
        left_name.startswith(right_name) or right_name.startswith(left_name)
    ):
        return 0.92
    return SequenceMatcher(None, left_name, right_name).ratio()


def _normalize_phone(value: Any) -> str:
    digits = re.sub(r"\D", "", _safe_str(value))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def _token_set(value: Any) -> set[str]:
    return {token for token in _normalize_text(value).split() if token}


def _overlap_score(left: Any, right: Any) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _token_overlap_score(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _hospital_pair_evaluation(
    test_name: str,
    input_name: str,
    test_canonical: str,
    input_canonical: str,
    token_frequencies: dict[str, int],
) -> dict[str, Any]:
    test_norm = _normalize_text(test_name)
    input_norm = _normalize_text(input_name)
    test_tokens = set(test_canonical.split()) if test_canonical else set()
    input_tokens = set(input_canonical.split()) if input_canonical else set()
    shared_tokens = test_tokens & input_tokens
    token_overlap = _token_overlap_score(test_tokens, input_tokens)
    sequence = SequenceMatcher(None, test_canonical, input_canonical).ratio() if test_canonical and input_canonical else 0.0

    test_core_tokens = set(_hospital_core_tokens(test_canonical))
    input_core_tokens = set(_hospital_core_tokens(input_canonical))
    shared_core_tokens = test_core_tokens & input_core_tokens
    shared_distinctive_core_tokens = {
        token for token in shared_core_tokens if _is_distinctive_hospital_token(token, token_frequencies)
    }
    core_overlap = _token_overlap_score(test_core_tokens, input_core_tokens)
    test_core_name = " ".join(sorted(test_core_tokens))
    input_core_name = " ".join(sorted(input_core_tokens))
    core_sequence = SequenceMatcher(None, test_core_name, input_core_name).ratio() if test_core_name and input_core_name else 0.0
    generic_only = not test_core_tokens or not input_core_tokens

    score = 0.0
    confidence = ""

    if test_norm and input_norm and test_norm == input_norm:
        score = 1.0
        confidence = "exact"
    elif test_canonical and input_canonical and test_canonical == input_canonical:
        score = 0.96
        confidence = "exact"
    elif len(shared_distinctive_core_tokens) >= 2 and (core_overlap >= 0.5 or core_sequence >= 0.82 or sequence >= 0.88):
        score = round(max(0.9, core_overlap * 0.55 + core_sequence * 0.25 + sequence * 0.2), 4)
        confidence = "strong"
    elif len(shared_distinctive_core_tokens) == 1:
        token = next(iter(shared_distinctive_core_tokens))
        if len(token) >= 8 and sequence >= 0.94 and (core_sequence >= 0.9 or token_overlap >= 0.5):
            score = round(max(0.82, core_sequence * 0.5 + sequence * 0.5), 4)
            confidence = "strong"
    elif not generic_only and len(shared_tokens) >= 2 and token_overlap >= 0.8 and sequence >= 0.96:
        score = round(max(0.84, token_overlap * 0.45 + sequence * 0.55), 4)
        confidence = "strong"
    elif not generic_only and len(shared_tokens) >= 2 and token_overlap >= 0.67 and sequence >= 0.85:
        score = round(max(0.78, token_overlap * 0.45 + sequence * 0.55), 4)
        confidence = "weak"

    return {
        "score": score,
        "confidence": confidence,
        "shared_core_tokens": sorted(shared_core_tokens),
        "shared_distinctive_core_tokens": sorted(shared_distinctive_core_tokens),
        "shared_all_tokens": sorted(shared_tokens),
        "shared_token_count": len(shared_tokens),
        "shared_distinctive_token_count": len(shared_distinctive_core_tokens),
        "single_token_match": len(shared_core_tokens) == 1,
        "distinctive_token_match": bool(shared_distinctive_core_tokens),
        "sequence_similarity": round(sequence, 4),
        "core_sequence_similarity": round(core_sequence, 4),
        "generic_only": generic_only,
    }


def _build_hospital_crosswalk(
    test_df: pd.DataFrame,
    input_df: pd.DataFrame,
    dataset_label: str,
) -> tuple[dict[str, set[str]], dict[str, Any], list[dict[str, Any]]]:
    test_unique = (
        test_df[["hospital_name", "hospital_name_canonical"]]
        .dropna(subset=["hospital_name"])
        .drop_duplicates()
        .copy()
    )
    input_unique = (
        input_df[["hospital_name", "hospital_name_canonical"]]
        .dropna(subset=["hospital_name"])
        .drop_duplicates()
        .copy()
    )

    input_by_canonical = {}
    core_token_index: dict[str, set[str]] = {}
    normalized_name_index: dict[str, set[str]] = {}
    canonical_names = [
        canonical
        for canonical in pd.concat(
            [
                test_unique["hospital_name_canonical"],
                input_unique["hospital_name_canonical"],
            ],
            ignore_index=True,
        ).dropna().astype(str).tolist()
        if canonical
    ]
    core_token_frequencies = _build_hospital_core_token_frequencies(canonical_names)
    for _, row in input_unique.iterrows():
        raw_name = _safe_str(row.get("hospital_name"))
        canonical = _safe_str(row.get("hospital_name_canonical"))
        if not raw_name or not canonical:
            continue
        input_by_canonical[canonical] = raw_name
        for token in _hospital_core_tokens(canonical):
            if _is_distinctive_hospital_token(token, core_token_frequencies):
                core_token_index.setdefault(token, set()).add(canonical)
        normalized_name_index.setdefault(_normalize_text(raw_name), set()).add(canonical)

    crosswalk: dict[str, set[str]] = {}
    audit_rows: list[dict[str, Any]] = []
    for _, row in test_unique.iterrows():
        test_raw = _safe_str(row.get("hospital_name"))
        test_canonical = _safe_str(row.get("hospital_name_canonical"))
        if not test_raw or not test_canonical:
            continue

        candidate_canonicals: set[str] = set()
        test_norm = _normalize_text(test_raw)
        test_core_tokens = _hospital_core_tokens(test_canonical)
        if test_norm in normalized_name_index:
            candidate_canonicals.update(normalized_name_index.get(test_norm, set()))
        if test_canonical in input_by_canonical:
            candidate_canonicals.add(test_canonical)
        distinctive_test_core_tokens = [
            token for token in test_core_tokens if _is_distinctive_hospital_token(token, core_token_frequencies)
        ]
        if distinctive_test_core_tokens:
            for token in distinctive_test_core_tokens:
                candidate_canonicals.update(core_token_index.get(token, set()))

        if not candidate_canonicals:
            continue

        exact_rows: list[dict[str, Any]] = []
        strong_rows: list[dict[str, Any]] = []
        weak_rows: list[dict[str, Any]] = []
        for candidate_canonical in candidate_canonicals:
            evaluation = _hospital_pair_evaluation(
                test_name=test_raw,
                input_name=input_by_canonical.get(candidate_canonical, candidate_canonical),
                test_canonical=test_canonical,
                input_canonical=candidate_canonical,
                token_frequencies=core_token_frequencies,
            )
            if not evaluation["confidence"]:
                continue
            row_data = {
                "dataset": dataset_label,
                "test_hospital_name": test_raw,
                "test_hospital_canonical": test_canonical,
                "input_hospital_name": input_by_canonical.get(candidate_canonical, candidate_canonical),
                "input_hospital_canonical": candidate_canonical,
                "crosswalk_score": evaluation["score"],
                "match_confidence": evaluation["confidence"],
                "shared_core_tokens": "|".join(evaluation["shared_core_tokens"]),
                "shared_distinctive_core_tokens": "|".join(evaluation["shared_distinctive_core_tokens"]),
                "shared_all_tokens": "|".join(evaluation["shared_all_tokens"]),
                "shared_token_count": evaluation["shared_token_count"],
                "shared_distinctive_token_count": evaluation["shared_distinctive_token_count"],
                "single_token_match": evaluation["single_token_match"],
                "distinctive_token_match": evaluation["distinctive_token_match"],
                "sequence_similarity": evaluation["sequence_similarity"],
            }
            if evaluation["confidence"] == "exact":
                exact_rows.append(row_data)
            elif evaluation["confidence"] == "strong":
                strong_rows.append(row_data)
            else:
                weak_rows.append(row_data)

        exact_rows = sorted(exact_rows, key=lambda item: (-float(item["crosswalk_score"]), item["input_hospital_name"]))
        strong_rows = sorted(strong_rows, key=lambda item: (-float(item["crosswalk_score"]), item["input_hospital_name"]))
        weak_rows = sorted(weak_rows, key=lambda item: (-float(item["crosswalk_score"]), item["input_hospital_name"]))

        kept_rows = list(exact_rows)
        for item in strong_rows:
            if len(kept_rows) >= 10:
                break
            if int(item["shared_distinctive_token_count"]) <= 0:
                continue
            kept_rows.append(item)

        kept_canonicals = {item["input_hospital_canonical"] for item in kept_rows}
        if kept_canonicals:
            crosswalk[test_canonical] = kept_canonicals

        ranked_rows = exact_rows + strong_rows + weak_rows
        for rank, item in enumerate(ranked_rows, start=1):
            audit_rows.append(
                {
                    **item,
                    "candidate_rank_for_test_hospital": rank,
                    "kept_for_scope": item["input_hospital_canonical"] in kept_canonicals,
                }
            )

    diagnostics = {
        "test_hospital_count": int(test_unique["hospital_name_canonical"].nunique()),
        "input_hospital_count": int(input_unique["hospital_name_canonical"].nunique()),
        "mapped_test_hospital_count": len(crosswalk),
        "mapped_input_hospital_count": len({item for values in crosswalk.values() for item in values}),
        "mapped_hospital_pairs": sum(len(values) for values in crosswalk.values()),
        "unmapped_test_hospital_count": int(test_unique["hospital_name_canonical"].nunique()) - len(crosswalk),
    }
    return crosswalk, diagnostics, audit_rows


def _scope_dataset_to_hospitals(
    test_df: pd.DataFrame,
    input_df: pd.DataFrame,
    dataset_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, Any]]:
    crosswalk, diagnostics, audit_rows = _build_hospital_crosswalk(test_df, input_df, dataset_label)
    mapped_test_hospitals = set(crosswalk.keys())
    mapped_input_hospitals = {item for values in crosswalk.values() for item in values}
    fanouts = sorted(len(values) for values in crosswalk.values())

    scoped_test_df = test_df[test_df["hospital_name_canonical"].isin(mapped_test_hospitals)].copy()
    scoped_input_df = input_df[input_df["hospital_name_canonical"].isin(mapped_input_hospitals)].copy()

    diagnostics = {
        **diagnostics,
        "scoped_test_provider_count": len(scoped_test_df),
        "scoped_input_provider_count": len(scoped_input_df),
        "avg_fanout": round(sum(fanouts) / len(fanouts), 2) if fanouts else 0.0,
        "median_fanout": fanouts[len(fanouts) // 2] if fanouts else 0,
        "max_fanout": max(fanouts) if fanouts else 0,
        "fanout_gt_10": sum(1 for item in fanouts if item > 10),
        "fanout_gt_25": sum(1 for item in fanouts if item > 25),
        "fanout_gt_50": sum(1 for item in fanouts if item > 50),
        "kept_single_token_match_count": sum(
            1
            for item in audit_rows
            if item.get("kept_for_scope") and item.get("single_token_match")
        ),
        "kept_single_token_distinctive_match_count": sum(
            1
            for item in audit_rows
            if item.get("kept_for_scope") and item.get("single_token_match") and item.get("distinctive_token_match")
        ),
        "hospital_crosswalk_rows": audit_rows,
    }
    _log(
        f"[{dataset_label}] Hospital scope: "
        f"test_hospitals={diagnostics['test_hospital_count']:,}, "
        f"input_hospitals={diagnostics['input_hospital_count']:,}, "
        f"mapped_test_hospitals={diagnostics['mapped_test_hospital_count']:,}, "
        f"unmapped_test_hospitals={diagnostics['unmapped_test_hospital_count']:,}, "
        f"mapped_input_hospitals={diagnostics['mapped_input_hospital_count']:,}, "
        f"scoped_test_providers={diagnostics['scoped_test_provider_count']:,}, "
        f"scoped_input_providers={diagnostics['scoped_input_provider_count']:,}, "
        f"avg_fanout={diagnostics['avg_fanout']}, median_fanout={diagnostics['median_fanout']}, "
        f"max_fanout={diagnostics['max_fanout']}, "
        f"kept_single_token_matches={diagnostics['kept_single_token_match_count']:,}"
    )
    return scoped_test_df, scoped_input_df, crosswalk, diagnostics


def _first_present(row: pd.Series, columns: list[str]) -> str:
    for column in columns:
        if column in row.index:
            value = _safe_str(row.get(column))
            if value:
                return value
    return ""


def _resolve_specialty(row: pd.Series, taxonomy_lookup: dict[str, str]) -> str:
    direct = _first_present(
        row,
        [
            "primary_specialty",
            "specialty",
            "primary_specialty_description",
            "Healthcare Provider Primary Taxonomy Description",
        ],
    )
    if direct:
        return direct

    for column in [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)]:
        code = _safe_str(row.get(column))
        if code:
            return taxonomy_lookup.get(code, code)
    return ""


def _parse_listish(value: Any) -> list[str]:
    text = _safe_str(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [_safe_str(item) for item in parsed if _safe_str(item)]
        except Exception:
            pass
    return [item.strip() for item in text.split("|") if item.strip()] or [text]


def _clean_clinic_name(value: Any) -> str:
    text = _safe_str(value)
    if not text:
        return ""
    normalized = _normalize_text(text)
    if normalized in PLACEHOLDER_CLINIC_NAMES:
        return ""
    return text


def _read_dataset(path: Path) -> pd.DataFrame:
    _log(f"Loading dataset: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        _log(f"Loaded parquet with {len(df):,} rows and {len(df.columns):,} columns")
        return df
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str, low_memory=False)
        _log(f"Loaded csv with {len(df):,} rows and {len(df.columns):,} columns")
        return df
    raise ValueError(f"Unsupported input file type: {path}")


def _read_parquet_projected(path: Path, candidate_columns: list[str]) -> pd.DataFrame:
    _log(f"Inspecting parquet schema for projection: {path}")
    try:
        import pyarrow.parquet as pq

        schema_columns = pq.ParquetFile(path).schema.names
        columns = [column for column in candidate_columns if column in schema_columns]
        _log(
            f"Reading parquet with column projection: selected {len(columns):,} of "
            f"{len(candidate_columns):,} candidate columns"
        )
        df = pd.read_parquet(path, columns=columns)
        _log(f"Loaded projected parquet with {len(df):,} rows and {len(df.columns):,} columns")
        return df
    except Exception as exc:
        _log(f"Projection failed, falling back to full parquet read: {exc}")
        return _read_dataset(path)


def _find_test_file(test_data_dir: Path, prefix: str) -> Path:
    matches = sorted(test_data_dir.glob(f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find test data file under {test_data_dir} with prefix '{prefix}'")
    return matches[-1]


def _prepare_test_df(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    _log(f"Preparing test dataset '{source_label}'")
    out = df.copy()
    out["source_dataset"] = source_label
    out["provider_name_norm"] = out["name"].map(_normalize_person_name)
    name_features = out["name"].map(_person_name_features)
    out["first_name_norm"] = name_features.map(lambda item: item[0])
    out["last_name_norm"] = name_features.map(lambda item: item[1])
    out["first3_name"] = name_features.map(lambda item: item[2])
    out["first4_name"] = name_features.map(lambda item: item[3])
    out["first_initial"] = name_features.map(lambda item: item[4])
    out["canonical_first_name_norm"] = name_features.map(lambda item: item[5])
    out["canonical_first3_name"] = name_features.map(lambda item: item[6])
    out["canonical_first4_name"] = name_features.map(lambda item: item[7])
    out["name_token_set_key"] = name_features.map(lambda item: item[8])
    out["full_name_key"] = out["provider_name_norm"]
    out["full_name_no_middle_key"] = out["name"].map(_name_without_middle_key)
    out["relaxed_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["first3_name"].fillna("").astype("string").str.strip()
    )
    out["canonical_relaxed_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["canonical_first3_name"].fillna("").astype("string").str.strip()
    )
    out["first4_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["canonical_first4_name"].fillna("").astype("string").str.strip()
    )
    out["first_initial_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["first_initial"].fillna("").astype("string").str.strip()
    )
    out["hospital_name_norm"] = out["hospital_name"].map(_normalize_text)
    out["hospital_name_canonical"] = out["hospital_name"].map(_canonical_hospital_name)
    out["specialty_norm"] = out["specialty"].map(_normalize_text)
    out["specialty_canonical"] = out["specialty"].map(_canonical_specialty_name)
    out["phone_norm"] = out["phone"].map(_normalize_phone)
    out["clinic_list"] = out["clinic_names"].map(_parse_listish)
    out["clinic_name"] = out["clinic_list"].map(lambda items: _clean_clinic_name(items[0] if items else ""))
    out["clinic_name_norm"] = out["clinic_name"].map(_normalize_text)
    out["clinic_name_canonical"] = out["clinic_name"].map(_canonical_clinic_name)
    clinic_address_list = out["clinic_addresses"].map(_parse_listish) if "clinic_addresses" in out.columns else pd.Series([[]] * len(out), index=out.index)
    out["clinic_address"] = clinic_address_list.map(lambda items: _safe_str(items[0] if items else ""))
    out["clinic_address_norm"] = out["clinic_address"].map(_normalize_text)
    out["clinic_address_zip_state_match_key"] = out["clinic_address"].map(
        lambda value: f"{_address_parts(value)['state']}|{_address_parts(value)['zip5']}"
    )
    _log(
        f"Prepared test dataset '{source_label}': "
        f"{len(out):,} providers, {out['hospital_name'].nunique():,} hospitals, "
        f"{out['clinic_name'].nunique():,} clinics"
    )
    return out


def _build_input_name(row: pd.Series) -> str:
    full = _first_present(
        row,
        [
            "PROVIDER_FULL_NAME",
            "provider_name",
            "name",
            "full_name",
        ],
    )
    if full:
        return full

    first = _first_present(row, ["Provider First Name", "FIRST_NAME", "first_name"])
    middle = _first_present(row, ["Provider Middle Name", "MDL_NAME", "middle_name"])
    last = _first_present(row, ["Provider Last Name (Legal Name)", "LAST_NAME", "last_name"])
    combined = " ".join(part for part in [first, middle, last] if part)
    return combined.strip()


def _coalesce_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.Series("", index=df.index, dtype="string")
    values = df[existing].astype("string").fillna("")
    values = values.replace(r"^\s*$", pd.NA, regex=True)
    return values.bfill(axis=1).iloc[:, 0].fillna("").str.strip()


def _build_input_name_vectorized(df: pd.DataFrame) -> pd.Series:
    full_name = _coalesce_columns(df, ["PROVIDER_FULL_NAME", "provider_name", "name", "full_name"])
    if (full_name != "").all():
        return full_name

    first = _coalesce_columns(df, ["Provider First Name", "FIRST_NAME", "first_name"])
    middle = _coalesce_columns(df, ["Provider Middle Name", "MDL_NAME", "middle_name"])
    last = _coalesce_columns(df, ["Provider Last Name (Legal Name)", "LAST_NAME", "last_name"])
    combined = (first + " " + middle + " " + last).str.replace(r"\s+", " ", regex=True).str.strip()
    return full_name.where(full_name != "", combined)


def _resolve_specialty_vectorized(df: pd.DataFrame, taxonomy_lookup: dict[str, str]) -> pd.Series:
    direct = _coalesce_columns(
        df,
        [
            "primary_specialty",
            "specialty",
            "primary_specialty_description",
            "Healthcare Provider Primary Taxonomy Description",
        ],
    )
    if (direct != "").all():
        return direct

    taxonomy_cols = [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)]
    taxonomy_code = _coalesce_columns(df, taxonomy_cols)
    taxonomy_desc = taxonomy_code.map(lambda code: taxonomy_lookup.get(code, code) if code else "")
    return direct.where(direct != "", taxonomy_desc)


def _prepare_input_identity_df(df: pd.DataFrame) -> pd.DataFrame:
    _log("Preparing input provider identity fields")
    out = df.copy()
    _log("Vectorizing provider name normalization")
    out["provider_name"] = _build_input_name_vectorized(out)
    out["provider_name_norm"] = out["provider_name"].map(_normalize_person_name)
    name_features = out["provider_name"].map(_person_name_features)
    out["first_name_norm"] = name_features.map(lambda item: item[0])
    out["last_name_norm"] = name_features.map(lambda item: item[1])
    out["first3_name"] = name_features.map(lambda item: item[2])
    out["first4_name"] = name_features.map(lambda item: item[3])
    out["first_initial"] = name_features.map(lambda item: item[4])
    out["canonical_first_name_norm"] = name_features.map(lambda item: item[5])
    out["canonical_first3_name"] = name_features.map(lambda item: item[6])
    out["canonical_first4_name"] = name_features.map(lambda item: item[7])
    out["name_token_set_key"] = name_features.map(lambda item: item[8])
    out["full_name_key"] = out["provider_name_norm"]
    out["full_name_no_middle_key"] = out["provider_name"].map(_name_without_middle_key)
    out["relaxed_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["first3_name"].fillna("").astype("string").str.strip()
    )
    out["canonical_relaxed_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["canonical_first3_name"].fillna("").astype("string").str.strip()
    )
    out["first4_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["canonical_first4_name"].fillna("").astype("string").str.strip()
    )
    out["first_initial_name_key"] = (
        out["last_name_norm"].fillna("").astype("string").str.strip()
        + "|"
        + out["first_initial"].fillna("").astype("string").str.strip()
    )
    out["provider_id_for_count"] = _coalesce_columns(
        out,
        ["NPI", "provider_id", "ENRLMT_ID"],
    )
    _log(
        "Prepared input identity fields: "
        f"{len(out):,} providers, "
        f"{out['full_name_key'].nunique():,} full-name keys, "
        f"{out['relaxed_name_key'].nunique():,} relaxed-name keys"
    )
    return out


def _enrich_input_candidate_df(df: pd.DataFrame, taxonomy_lookup: dict[str, str]) -> pd.DataFrame:
    _log("Preparing comparison fields on hospital-scoped input pool")
    out = df.copy()
    _log("Vectorizing hospital field selection on hospital-scoped input pool")
    out["hospital_name"] = _coalesce_columns(
        out,
        [
            "mapped_facility_name",
            "mapped_org_name",
            "Provider Organization Name (Legal Business Name)",
            "organization_name",
            "hospital_name",
        ],
    )
    out["hospital_name_norm"] = out["hospital_name"].map(_normalize_text)
    out["hospital_name_canonical"] = out["hospital_name"].map(_canonical_hospital_name)
    out["clinic_name"] = _coalesce_columns(
        out,
        [
            "mapped_clinic_name",
            "clinic_name",
            "practice_location_name",
            "location_name",
        ],
    )
    out["clinic_name_norm"] = out["clinic_name"].map(_normalize_text)
    out["clinic_name_canonical"] = out["clinic_name"].map(_canonical_clinic_name)
    _log("Vectorizing specialty resolution on hospital-scoped input pool")
    out["specialty"] = _resolve_specialty_vectorized(out, taxonomy_lookup)
    out["specialty_norm"] = out["specialty"].map(_normalize_text)
    out["specialty_canonical"] = out["specialty"].map(_canonical_specialty_name)
    _log("Vectorizing phone field selection on hospital-scoped input pool")
    out["phone"] = _coalesce_columns(
        out,
        [
            "Provider Business Practice Location Address Telephone Number",
            "Provider Business Mailing Address Telephone Number",
            "phone",
            "Phone",
            "PHONE_NUM",
        ],
    )
    out["phone_norm"] = out["phone"].map(_normalize_phone)
    address_line_1 = _coalesce_columns(
        out,
        [
            "practice_location_address",
            "clinic_address",
            "Provider First Line Business Practice Location Address",
            "mapped_facility_address",
            "mapped_org_address",
            "address",
            "Address",
        ],
    )
    address_line_2 = _coalesce_columns(
        out,
        [
            "Provider Second Line Business Practice Location Address",
        ],
    )
    address_city = _coalesce_columns(
        out,
        [
            "Provider Business Practice Location Address City Name",
            "CITY_NAME",
            "city",
        ],
    )
    address_state = _coalesce_columns(
        out,
        [
            "Provider Business Practice Location Address State Name",
            "STATE_CD",
            "state",
        ],
    )
    address_zip = _coalesce_columns(
        out,
        [
            "Provider Business Practice Location Address Postal Code",
            "ZIP_CD",
            "zip",
        ],
    )
    out["address"] = (
        address_line_1
        + " "
        + address_line_2
        + " "
        + address_city
        + " "
        + address_state
        + " "
        + address_zip
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    out["address_norm"] = out["address"].map(_normalize_text)
    out["address_zip_state_match_key"] = out["address"].map(
        lambda value: f"{_address_parts(value)['state']}|{_address_parts(value)['zip5']}"
    )
    _log(
        "Prepared hospital-scoped comparison pool: "
        f"{len(out):,} providers, "
        f"{out['provider_name_norm'].nunique():,} normalized names, "
        f"{out['hospital_name_norm'].nunique():,} hospitals"
    )
    return out


def _enrich_input_scope_df(df: pd.DataFrame) -> pd.DataFrame:
    _log("Preparing input hospital fields for scope computation")
    out = df.copy()
    out["hospital_name"] = _coalesce_columns(
        out,
        [
            "mapped_facility_name",
            "mapped_org_name",
            "Provider Organization Name (Legal Business Name)",
            "organization_name",
            "hospital_name",
        ],
    )
    out["hospital_name_norm"] = out["hospital_name"].map(_normalize_text)
    out["hospital_name_canonical"] = out["hospital_name"].map(_canonical_hospital_name)
    return out


def _build_group_index(
    df: pd.DataFrame,
    column: str,
    invalid_values: Optional[set[str]] = None,
) -> dict[str, pd.Index]:
    invalid_values = invalid_values or {""}
    if column not in df.columns:
        return {}
    groups = {}
    for name, group_index in df.groupby(column, sort=False).groups.items():
        key = _safe_str(name)
        if not key or key in invalid_values:
            continue
        groups[key] = group_index
    return groups


def _score_candidates(
    test_row: pd.Series,
    candidates: pd.DataFrame,
    mapped_input_hospital_canonicals: Optional[set[str]] = None,
    matched_input_use_counts: Optional[dict[int, int]] = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.head(0).copy()

    scoped_candidates = candidates
    if mapped_input_hospital_canonicals:
        in_scope_candidates = candidates[
            candidates["hospital_name_canonical"].fillna("").astype("string").isin(mapped_input_hospital_canonicals)
        ]
        if not in_scope_candidates.empty:
            scoped_candidates = in_scope_candidates

    scored = scoped_candidates.copy()
    exact_name_bonus = (
        scored["full_name_key"].fillna("").astype("string") == _safe_str(test_row.get("full_name_key"))
    ).astype("float64") * 100.0
    exact_specialty_bonus = scored["specialty_canonical"].map(
        lambda value: 20.0 if value and value == _safe_str(test_row.get("specialty_canonical")) else 0.0
    )
    loose_specialty_bonus = scored["specialty"].map(
        lambda value: 10.0 if _specialty_match_loose(test_row.get("specialty"), value) else 0.0
    )
    exact_phone_bonus = (
        (scored["phone_norm"].fillna("").astype("string") != "")
        & (scored["phone_norm"].fillna("").astype("string") == _safe_str(test_row.get("phone_norm")))
    ).astype("float64") * 15.0
    mapped_hospital_bonus = pd.Series(0.0, index=scored.index)
    if mapped_input_hospital_canonicals:
        mapped_hospital_bonus = (
            scored["hospital_name_canonical"].fillna("").astype("string").isin(mapped_input_hospital_canonicals)
        ).astype("float64") * 25.0
    address_bonus = scored["address"].map(
        lambda value: 10.0 if _address_match(test_row.get("clinic_address"), value) else 0.0
    )
    address_zip_state_bonus = scored["address"].map(
        lambda value: 5.0 if _address_match_zip_state(test_row.get("clinic_address"), value) else 0.0
    )
    reuse_penalty = pd.Series(0.0, index=scored.index)
    if matched_input_use_counts:
        reuse_penalty = scored.index.to_series().map(lambda idx: float(matched_input_use_counts.get(int(idx), 0)) * 8.0)
    scored["candidate_score"] = scored.apply(
        lambda row: (
            3.0 * _overlap_score(test_row.get("hospital_name"), row.get("hospital_name"))
            + 1.5 * _overlap_score(test_row.get("clinic_name"), row.get("clinic_name"))
        ),
        axis=1,
    )
    scored["candidate_score"] = (
        scored["candidate_score"]
        + exact_name_bonus
        + exact_specialty_bonus
        + loose_specialty_bonus
        + exact_phone_bonus
        + mapped_hospital_bonus
        + address_bonus
        + address_zip_state_bonus
        - reuse_penalty
    )
    scored = scored.sort_values(
        by=["candidate_score", "hospital_name", "specialty", "provider_id_for_count"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return scored


def _choose_best_candidate(
    test_row: pd.Series,
    candidates: pd.DataFrame,
    mapped_input_hospital_canonicals: Optional[set[str]] = None,
    matched_input_use_counts: Optional[dict[int, int]] = None,
    require_score_gap: bool = False,
    min_score_gap: float = 5.0,
) -> Optional[dict[str, Any]]:
    scored = _score_candidates(
        test_row,
        candidates,
        mapped_input_hospital_canonicals,
        matched_input_use_counts=matched_input_use_counts,
    )
    if scored.empty:
        return None
    best = scored.iloc[0]
    second_best_score = float(scored.iloc[1]["candidate_score"]) if len(scored) > 1 else None
    score_gap = float(best["candidate_score"] - second_best_score) if second_best_score is not None else None
    if require_score_gap and second_best_score is not None and score_gap is not None and score_gap < min_score_gap:
        return None
    return {
        "row": best,
        "candidate_score": float(best["candidate_score"]),
        "second_best_candidate_score": second_best_score,
        "score_gap": score_gap,
        "candidate_count": int(len(scored)),
    }


def _name_similarity_score(left: Any, right: Any) -> float:
    left_name = _normalize_person_name(left)
    right_name = _normalize_person_name(right)
    if not left_name or not right_name:
        return 0.0
    if left_name == right_name:
        return 1.0
    return SequenceMatcher(None, left_name, right_name).ratio()


def _annotate_candidate_support(
    test_row: pd.Series,
    candidates: pd.DataFrame,
    mapped_input_hospital_canonicals: Optional[set[str]],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.head(0).copy()
    annotated = candidates.copy()
    if mapped_input_hospital_canonicals:
        annotated["_mapped_hospital"] = (
            annotated["hospital_name_canonical"].fillna("").astype("string").isin(mapped_input_hospital_canonicals)
        )
    else:
        annotated["_mapped_hospital"] = True
    test_phone = _safe_str(test_row.get("phone_norm"))
    test_first_name = _safe_str(test_row.get("first_name_norm"))
    test_last_name = _safe_str(test_row.get("last_name_norm"))
    annotated["_specialty_loose"] = annotated["specialty"].map(
        lambda value: _specialty_match_loose(test_row.get("specialty"), value)
    )
    annotated["_address_zip_state"] = annotated["address"].map(
        lambda value: _address_match_zip_state(test_row.get("clinic_address"), value)
    )
    annotated["_phone_match"] = (
        (annotated["phone_norm"].fillna("").astype("string") != "")
        & (annotated["phone_norm"].fillna("").astype("string") == test_phone)
    )
    annotated["_first_name_similarity"] = annotated["first_name_norm"].map(
        lambda value: _first_name_similarity(test_first_name, value)
    )
    annotated["_last_name_exact"] = (
        annotated["last_name_norm"].fillna("").astype("string") == test_last_name
    )
    annotated["_name_similarity"] = annotated["provider_name"].map(
        lambda value: _name_similarity_score(test_row.get("name"), value)
    )
    return annotated


def _name_plus_field_candidates(
    test_row: pd.Series,
    candidates: pd.DataFrame,
    mapped_input_hospital_canonicals: Optional[set[str]],
    require_first_name_similarity: bool = False,
) -> pd.DataFrame:
    annotated = _annotate_candidate_support(test_row, candidates, mapped_input_hospital_canonicals)
    if annotated.empty:
        return annotated
    annotated = annotated[annotated["_mapped_hospital"]].copy()
    if annotated.empty:
        return annotated
    if require_first_name_similarity:
        annotated = annotated[annotated["_first_name_similarity"] >= 0.84].copy()
        if annotated.empty:
            return annotated
    support_mask = annotated["_specialty_loose"] | annotated["_address_zip_state"] | annotated["_phone_match"]
    return annotated[support_mask].copy()


def _field_led_candidates(
    test_row: pd.Series,
    candidates: pd.DataFrame,
    mapped_input_hospital_canonicals: Optional[set[str]],
    mode: str,
) -> pd.DataFrame:
    annotated = _annotate_candidate_support(test_row, candidates, mapped_input_hospital_canonicals)
    if annotated.empty:
        return annotated
    if mode == "phone_exact":
        if mapped_input_hospital_canonicals:
            in_scope = annotated[annotated["_mapped_hospital"]].copy()
            if not in_scope.empty:
                annotated = in_scope
        return annotated[annotated["_phone_match"]].copy()
    if mode == "last_name_specialty_zip":
        annotated = annotated[annotated["_mapped_hospital"]].copy()
        if annotated.empty:
            return annotated
        return annotated[annotated["_last_name_exact"] & annotated["_specialty_loose"] & annotated["_address_zip_state"]].copy()
    if mode == "last_name_specialty_hospital":
        annotated = annotated[annotated["_mapped_hospital"]].copy()
        if annotated.empty:
            return annotated
        return annotated[annotated["_last_name_exact"] & annotated["_specialty_loose"]].copy()
    return annotated.head(0).copy()


def _compare_single_dataset(
    test_df: pd.DataFrame,
    hospital_scoped_input_df: pd.DataFrame,
    dataset_label: str,
    hospital_crosswalk: Optional[dict[str, set[str]]] = None,
    scope_diagnostics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    _log(
        f"Starting comparison for '{dataset_label}' with "
        f"{len(test_df):,} test providers against {len(hospital_scoped_input_df):,} hospital-scoped input providers"
    )
    results = []
    matched_input_indices = set()
    matched_input_use_counts: dict[int, int] = {}
    candidate_input_indices = set()
    candidate_input_indices_by_hospital: dict[str, set[int]] = {}
    candidate_input_indices_by_bucket = {
        "name_strong": set(),
        "name_plus_field": set(),
        "field_led": set(),
    }
    candidate_input_indices_by_stage = {
        "exact": set(),
        "relaxed": set(),
        "fallback_weak": set(),
    }
    matched_counts_by_bucket = {
        "name_strong": 0,
        "name_plus_field": 0,
        "field_led": 0,
    }
    retrieval_audit_rows: list[dict[str, Any]] = []

    input_by_full_name = _build_group_index(hospital_scoped_input_df, "full_name_key")
    input_by_no_middle_name = _build_group_index(hospital_scoped_input_df, "full_name_no_middle_key")
    input_by_relaxed_name = _build_group_index(hospital_scoped_input_df, "relaxed_name_key", invalid_values={"", "|"})
    input_by_canonical_relaxed_name = _build_group_index(
        hospital_scoped_input_df,
        "canonical_relaxed_name_key",
        invalid_values={"", "|"},
    )
    input_by_first4_name = _build_group_index(hospital_scoped_input_df, "first4_name_key", invalid_values={"", "|"})
    input_by_name_token_set = _build_group_index(hospital_scoped_input_df, "name_token_set_key")
    input_by_first_initial_name = _build_group_index(
        hospital_scoped_input_df,
        "first_initial_name_key",
        invalid_values={"", "|"},
    )
    input_by_last_name = _build_group_index(hospital_scoped_input_df, "last_name_norm")
    input_by_phone = _build_group_index(hospital_scoped_input_df, "phone_norm")
    _log(
        f"Built candidate indexes for '{dataset_label}': "
        f"{len(input_by_full_name):,} full-name keys, "
        f"{len(input_by_relaxed_name):,} relaxed-name keys, "
        f"{len(input_by_last_name):,} last-name keys, "
        f"{len(input_by_phone):,} phone keys"
    )

    hospital_crosswalk = hospital_crosswalk or {}

    exact_count = 0
    relaxed_count = 0
    fallback_weak_count = 0
    unmatched_after_exact_relaxed_count = 0

    def _register_candidates(
        hospital_name: str,
        retrieval_bucket: str,
        candidates: pd.DataFrame,
        stage_bucket: Optional[str] = None,
    ):
        if candidates.empty:
            return
        index_values = set(candidates.index.tolist())
        candidate_input_indices.update(index_values)
        candidate_input_indices_by_bucket.setdefault(retrieval_bucket, set()).update(index_values)
        if stage_bucket:
            candidate_input_indices_by_stage.setdefault(stage_bucket, set()).update(index_values)
        candidate_input_indices_by_hospital.setdefault(hospital_name, set()).update(index_values)

    for _, test_row in test_df.iterrows():
        test_hospital_name = _safe_str(test_row.get("hospital_name"))
        mapped_input_hospitals = hospital_crosswalk.get(_safe_str(test_row.get("hospital_name_canonical")), set())
        best_info = None
        match_strategy = "unmatched"
        match_strategy_bucket = ""
        retrieval_bucket = ""
        seen_stage_keys: set[tuple[str, str]] = set()

        for strategy_name, stage_bucket, key_column, index_map, invalid_values in [
            ("exact_full_name", "exact", "full_name_key", input_by_full_name, {"", "|"}),
            ("exact_no_middle_name", "exact", "full_name_no_middle_key", input_by_no_middle_name, {"", "|"}),
            ("relaxed_name", "relaxed", "relaxed_name_key", input_by_relaxed_name, {"", "|"}),
            (
                "relaxed_name_canonical",
                "relaxed",
                "canonical_relaxed_name_key",
                input_by_canonical_relaxed_name,
                {"", "|"},
            ),
            ("relaxed_name_first4", "relaxed", "first4_name_key", input_by_first4_name, {"", "|"}),
            ("token_set_name", "relaxed", "name_token_set_key", input_by_name_token_set, {""}),
        ]:
            key_value = _safe_str(test_row.get(key_column))
            if not key_value or key_value in invalid_values:
                continue
            dedupe_key = (strategy_name, key_value)
            if dedupe_key in seen_stage_keys:
                continue
            seen_stage_keys.add(dedupe_key)
            candidate_index = index_map.get(key_value)
            if candidate_index is None:
                continue
            candidates = hospital_scoped_input_df.loc[candidate_index]
            _register_candidates(test_hospital_name, "name_strong", candidates, stage_bucket=stage_bucket)
            best_info = _choose_best_candidate(
                test_row,
                candidates,
                mapped_input_hospitals,
                matched_input_use_counts=matched_input_use_counts,
            )
            if best_info is not None:
                match_strategy = strategy_name
                match_strategy_bucket = stage_bucket
                retrieval_bucket = "name_strong"
                break

        if best_info is None:
            unmatched_after_exact_relaxed_count += 1

        if best_info is None:
            first_initial_key = _safe_str(test_row.get("first_initial_name_key"))
            if first_initial_key and first_initial_key != "|":
                candidate_index = input_by_first_initial_name.get(first_initial_key)
                if candidate_index is not None:
                    retrieval_candidates = hospital_scoped_input_df.loc[candidate_index]
                    retrieval_candidates = _name_plus_field_candidates(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        require_first_name_similarity=False,
                    )
                    _register_candidates(
                        test_hospital_name,
                        "name_plus_field",
                        retrieval_candidates,
                        stage_bucket="fallback_weak",
                    )
                    best_info = _choose_best_candidate(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        matched_input_use_counts=matched_input_use_counts,
                        require_score_gap=True,
                    )
                    if best_info is not None:
                        match_strategy = "validated_first_initial"
                        match_strategy_bucket = "fallback_weak"
                        retrieval_bucket = "name_plus_field"

        if best_info is None:
            last_name_key = _safe_str(test_row.get("last_name_norm"))
            if last_name_key:
                candidate_index = input_by_last_name.get(last_name_key)
                if candidate_index is not None:
                    retrieval_candidates = hospital_scoped_input_df.loc[candidate_index]
                    retrieval_candidates = _name_plus_field_candidates(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        require_first_name_similarity=True,
                    )
                    _register_candidates(
                        test_hospital_name,
                        "name_plus_field",
                        retrieval_candidates,
                        stage_bucket="fallback_weak",
                    )
                    best_info = _choose_best_candidate(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        matched_input_use_counts=matched_input_use_counts,
                        require_score_gap=True,
                    )
                    if best_info is not None:
                        match_strategy = "validated_last_name_similarity"
                        match_strategy_bucket = "fallback_weak"
                        retrieval_bucket = "name_plus_field"

        if best_info is None:
            phone_key = _safe_str(test_row.get("phone_norm"))
            if phone_key:
                candidate_index = input_by_phone.get(phone_key)
                if candidate_index is not None:
                    retrieval_candidates = hospital_scoped_input_df.loc[candidate_index]
                    retrieval_candidates = _field_led_candidates(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        mode="phone_exact",
                    )
                    _register_candidates(
                        test_hospital_name,
                        "field_led",
                        retrieval_candidates,
                        stage_bucket="fallback_weak",
                    )
                    best_info = _choose_best_candidate(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        matched_input_use_counts=matched_input_use_counts,
                        require_score_gap=True,
                    )
                    if best_info is not None:
                        match_strategy = "field_led_phone_exact"
                        match_strategy_bucket = "fallback_weak"
                        retrieval_bucket = "field_led"

        if best_info is None:
            last_name_key = _safe_str(test_row.get("last_name_norm"))
            if last_name_key:
                candidate_index = input_by_last_name.get(last_name_key)
                if candidate_index is not None:
                    retrieval_candidates = hospital_scoped_input_df.loc[candidate_index]
                    retrieval_candidates = _field_led_candidates(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        mode="last_name_specialty_zip",
                    )
                    _register_candidates(
                        test_hospital_name,
                        "field_led",
                        retrieval_candidates,
                        stage_bucket="fallback_weak",
                    )
                    best_info = _choose_best_candidate(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        matched_input_use_counts=matched_input_use_counts,
                        require_score_gap=True,
                    )
                    if best_info is not None:
                        match_strategy = "field_led_last_name_specialty_zip"
                        match_strategy_bucket = "fallback_weak"
                        retrieval_bucket = "field_led"

        if best_info is None:
            last_name_key = _safe_str(test_row.get("last_name_norm"))
            if last_name_key:
                candidate_index = input_by_last_name.get(last_name_key)
                if candidate_index is not None:
                    retrieval_candidates = hospital_scoped_input_df.loc[candidate_index]
                    retrieval_candidates = _field_led_candidates(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        mode="last_name_specialty_hospital",
                    )
                    _register_candidates(
                        test_hospital_name,
                        "field_led",
                        retrieval_candidates,
                        stage_bucket="fallback_weak",
                    )
                    best_info = _choose_best_candidate(
                        test_row,
                        retrieval_candidates,
                        mapped_input_hospitals,
                        matched_input_use_counts=matched_input_use_counts,
                        require_score_gap=True,
                    )
                    if best_info is not None:
                        match_strategy = "field_led_last_name_specialty_hospital"
                        match_strategy_bucket = "fallback_weak"
                        retrieval_bucket = "field_led"

        if best_info is None:
            results.append(
                {
                    "source_dataset": dataset_label,
                    "provider_name": test_row.get("name", ""),
                    "hospital_name": test_row.get("hospital_name", ""),
                    "clinic_name": test_row.get("clinic_name", ""),
                    "clinic_address": test_row.get("clinic_address", ""),
                    "matched_in_input": False,
                    "hospital_match": False,
                    "clinic_match": False,
                    "affiliation_match": False,
                    "specialty_match_strict": False,
                    "specialty_match_loose": False,
                    "phone_match": False,
                    "address_match": False,
                    "address_match_zip_state": False,
                    "fully_matched": False,
                    "partially_matched": False,
                    "match_strategy": "unmatched",
                    "match_strategy_bucket": "unmatched",
                    "retrieval_bucket": "unmatched",
                }
            )
            continue

        best = best_info["row"]
        matched_counts_by_bucket[retrieval_bucket] = matched_counts_by_bucket.get(retrieval_bucket, 0) + 1
        if match_strategy_bucket == "exact":
            exact_count += 1
        elif match_strategy_bucket == "relaxed":
            relaxed_count += 1
        else:
            fallback_weak_count += 1
        matched_input_indices.add(best.name)
        matched_input_use_counts[int(best.name)] = matched_input_use_counts.get(int(best.name), 0) + 1
        hospital_match = bool(_hospital_names_match(
            test_hospital_name=test_row.get("hospital_name"),
            input_hospital_name=best.get("hospital_name"),
            test_hospital_canonical=_safe_str(test_row.get("hospital_name_canonical")),
            input_hospital_canonical=_safe_str(best.get("hospital_name_canonical")),
        ))
        clinic_match = bool(_clinic_names_match(
            test_clinic_name=test_row.get("clinic_name"),
            input_clinic_name=best.get("clinic_name"),
            test_clinic_canonical=_safe_str(test_row.get("clinic_name_canonical")),
            input_clinic_canonical=_safe_str(best.get("clinic_name_canonical")),
        ))
        affiliation_match = bool(hospital_match and (
            clinic_match if _safe_str(test_row.get("clinic_name")) else True
        ))
        test_specialty_canonical = _safe_str(test_row.get("specialty_canonical"))
        input_specialty_canonical = _safe_str(best.get("specialty_canonical"))
        specialty_match_strict = bool(
            test_specialty_canonical
            and input_specialty_canonical
            and test_specialty_canonical == input_specialty_canonical
        )
        specialty_match_loose = bool(_specialty_match_loose(test_row.get("specialty"), best.get("specialty")))
        test_phone = _normalize_phone(test_row.get("phone"))
        input_phone = _normalize_phone(best.get("phone"))
        phone_match = bool(test_phone and input_phone and test_phone == input_phone)
        address_match = bool(_address_match(test_row.get("clinic_address"), best.get("address")))
        address_match_zip_state = bool(_address_match_zip_state(test_row.get("clinic_address"), best.get("address")))
        fully_matched = bool(hospital_match and specialty_match_loose and address_match)
        partially_matched = bool(not fully_matched)

        results.append(
            {
                "source_dataset": dataset_label,
                "provider_name": test_row.get("name", ""),
                "hospital_name": test_row.get("hospital_name", ""),
                "clinic_name": test_row.get("clinic_name", ""),
                "clinic_address": test_row.get("clinic_address", ""),
                "matched_in_input": True,
                "hospital_match": hospital_match,
                "clinic_match": clinic_match,
                "affiliation_match": affiliation_match,
                "specialty_match_strict": specialty_match_strict,
                "specialty_match_loose": specialty_match_loose,
                "phone_match": phone_match,
                "address_match": address_match,
                "address_match_zip_state": address_match_zip_state,
                "fully_matched": fully_matched,
                "partially_matched": partially_matched,
                "input_provider_name": best.get("provider_name", ""),
                "input_hospital_name": best.get("hospital_name", ""),
                "input_clinic_name": best.get("clinic_name", ""),
                "input_specialty": best.get("specialty", ""),
                "input_phone": best.get("phone", ""),
                "input_address": best.get("address", ""),
                "match_strategy": match_strategy,
                "match_strategy_bucket": match_strategy_bucket,
                "retrieval_bucket": retrieval_bucket,
                "candidate_score": best_info["candidate_score"],
                "second_best_candidate_score": best_info["second_best_candidate_score"],
                "candidate_score_gap": best_info["score_gap"],
                "candidate_count": best_info["candidate_count"],
            }
        )
        if retrieval_bucket in {"name_plus_field", "field_led"}:
            retrieval_audit_rows.append(
                {
                    "dataset": dataset_label,
                    "test_provider_name": test_row.get("name", ""),
                    "test_hospital_name": test_row.get("hospital_name", ""),
                    "test_specialty": test_row.get("specialty", ""),
                    "test_clinic_address": test_row.get("clinic_address", ""),
                    "retrieval_bucket": retrieval_bucket,
                    "match_strategy": match_strategy,
                    "input_provider_name": best.get("provider_name", ""),
                    "input_hospital_name": best.get("hospital_name", ""),
                    "input_specialty": best.get("specialty", ""),
                    "input_phone": best.get("phone", ""),
                    "input_address": best.get("address", ""),
                    "name_similarity": round(_name_similarity_score(test_row.get("name"), best.get("provider_name")), 4),
                    "specialty_match_loose": specialty_match_loose,
                    "address_match_zip_state": address_match_zip_state,
                    "phone_match": phone_match,
                    "candidate_score": best_info["candidate_score"],
                    "second_best_candidate_score": best_info["second_best_candidate_score"],
                    "candidate_score_gap": best_info["score_gap"],
                    "candidate_count": best_info["candidate_count"],
                }
            )

    matches_df = pd.DataFrame(results)
    if matches_df.empty:
        matches_df = pd.DataFrame(
            columns=[
                "source_dataset",
                "provider_name",
                "hospital_name",
                "clinic_name",
                "clinic_address",
                "matched_in_input",
                "hospital_match",
                "clinic_match",
                "affiliation_match",
                "specialty_match_strict",
                "specialty_match_loose",
                "phone_match",
                "address_match",
                "address_match_zip_state",
                "fully_matched",
                "partially_matched",
                "match_strategy",
                "match_strategy_bucket",
                "retrieval_bucket",
            ]
        )

    relevant_input_df = hospital_scoped_input_df.copy()
    relevant_input_indices = set(relevant_input_df.index.tolist())
    hospital_scoped_input_count = len(relevant_input_df)
    candidate_input_count = len(candidate_input_indices)
    unique_matched_input_provider_count = len(relevant_input_indices & matched_input_indices)
    input_only_count = len(relevant_input_indices - matched_input_indices)
    test_only_count = int((~matches_df["matched_in_input"]).sum()) if not matches_df.empty else len(test_df)
    matched_count = int(matches_df["matched_in_input"].sum()) if not matches_df.empty else 0
    fully_matched_count = int(matches_df["fully_matched"].sum()) if not matches_df.empty else 0
    partially_matched_count = int(matches_df["partially_matched"].sum()) if not matches_df.empty else 0
    only_in_test_count = test_only_count
    only_in_input_count = input_only_count
    reused_input_match_count = matched_count - unique_matched_input_provider_count
    union_count = fully_matched_count + partially_matched_count + only_in_test_count + only_in_input_count

    def _pct(numerator: int, denominator: int) -> float:
        return round((100.0 * numerator / denominator), 2) if denominator else 0.0

    matched_pct_of_test = _pct(matched_count, len(test_df))

    def _group_stats(group_cols: list[str]) -> list[dict[str, Any]]:
        rows = []
        for keys, group in matches_df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_cols, keys))
            total = len(group)
            matched = int(group["matched_in_input"].sum())
            test_only_group = int((~group["matched_in_input"]).sum())
            fully_group = int(group["fully_matched"].sum())
            partial_group = int(group["partially_matched"].sum())
            input_only_group = ""
            union_group = ""
            input_scope_group = ""
            retrieval_reached_group = ""
            unique_matched_input_group = ""
            if group_cols == ["hospital_name"]:
                hospital_canonical = _canonical_hospital_name(key_map.get("hospital_name", ""))
                mapped_input_canonicals = hospital_crosswalk.get(hospital_canonical, {hospital_canonical})
                scoped_input_group = relevant_input_df[
                    relevant_input_df["hospital_name_canonical"].isin(mapped_input_canonicals)
                ]
                scoped_input_indices = set(scoped_input_group.index.tolist())
                unmatched = relevant_input_df[
                    (relevant_input_df["hospital_name_canonical"].isin(mapped_input_canonicals))
                    & (~relevant_input_df.index.isin(list(matched_input_indices)))
                ]
                input_scope_group = int(len(scoped_input_group))
                retrieval_reached_group = int(len(candidate_input_indices_by_hospital.get(key_map.get("hospital_name", ""), set())))
                unique_matched_input_group = int(len(scoped_input_indices & matched_input_indices))
                input_only_group = int(len(unmatched))
                union_group = fully_group + partial_group + test_only_group + input_only_group
            elif group_cols == ["hospital_name", "clinic_name"]:
                hospital_canonical = _canonical_hospital_name(key_map.get("hospital_name", ""))
                clinic_canonical = _canonical_clinic_name(key_map.get("clinic_name", ""))
                mapped_input_canonicals = hospital_crosswalk.get(hospital_canonical, {hospital_canonical})
                unmatched = relevant_input_df[
                    (relevant_input_df["hospital_name_canonical"].isin(mapped_input_canonicals))
                    & (relevant_input_df["clinic_name_canonical"] == clinic_canonical)
                    & (~relevant_input_df.index.isin(list(matched_input_indices)))
                ]
                input_only_group = int(len(unmatched))
                union_group = fully_group + partial_group + test_only_group + input_only_group
            rows.append(
                {
                    **key_map,
                    "test_provider_count": total,
                    "input_provider_count_in_scope": input_scope_group,
                    "candidate_input_provider_count": "",
                    "retrieval_reached_input_provider_count": retrieval_reached_group,
                    "unique_matched_input_provider_count": unique_matched_input_group,
                    "matched_provider_count": matched,
                    "matched_provider_pct_of_test": _pct(matched, total),
                    "fully_matched_count": fully_group,
                    "partially_matched_count": partial_group,
                    "only_in_test_count": test_only_group,
                    "only_in_input_count": input_only_group,
                    "union_provider_count": union_group,
                    "fully_matched_pct": _pct(fully_group, union_group if union_group != "" else total),
                    "partially_matched_pct": _pct(partial_group, union_group if union_group != "" else total),
                    "only_in_test_pct": _pct(test_only_group, union_group if union_group != "" else total),
                    "only_in_input_pct": _pct(input_only_group if input_only_group != "" else 0, union_group if union_group != "" else total),
                    "hospital_match_pct": _pct(int(group["hospital_match"].sum()), matched) if matched else 0.0,
                    "clinic_match_pct": _pct(int(group["clinic_match"].sum()), matched) if matched else 0.0,
                    "affiliation_match_pct": _pct(int(group["affiliation_match"].sum()), matched) if matched else 0.0,
                    "specialty_match_pct_strict": _pct(int(group["specialty_match_strict"].sum()), matched) if matched else 0.0,
                    "specialty_match_pct_loose": _pct(int(group["specialty_match_loose"].sum()), matched) if matched else 0.0,
                    "phone_match_pct": _pct(int(group["phone_match"].sum()), matched) if matched else 0.0,
                    "address_match_pct": _pct(int(group["address_match"].sum()), matched) if matched else 0.0,
                    "address_match_zip_state_pct": _pct(int(group["address_match_zip_state"].sum()), matched) if matched else 0.0,
                }
            )
        return sorted(rows, key=lambda item: (-item["test_provider_count"], str(item.get(group_cols[0], ""))))

    exact_match_pct = _pct(exact_count, matched_count) if matched_count else 0.0
    relaxed_match_pct = _pct(relaxed_count, matched_count) if matched_count else 0.0
    fallback_weak_match_pct = _pct(fallback_weak_count, matched_count) if matched_count else 0.0

    per_hospital_rows = _group_stats(["hospital_name"])
    per_clinic_source = matches_df[matches_df["clinic_name"].astype("string").fillna("").str.strip() != ""]
    per_clinic_rows = _group_stats(["hospital_name", "clinic_name"]) if len(per_clinic_source) == len(matches_df) else [
        {
            **(
                dict(zip(["hospital_name", "clinic_name"], keys if isinstance(keys, tuple) else (keys,)))
            ),
            "test_provider_count": len(group),
            "input_provider_count_in_scope": "",
            "candidate_input_provider_count": "",
            "unique_matched_input_provider_count": "",
            "matched_provider_count": int(group["matched_in_input"].sum()),
            "matched_provider_pct_of_test": _pct(int(group["matched_in_input"].sum()), len(group)),
            "fully_matched_count": int(group["fully_matched"].sum()),
            "partially_matched_count": int(group["partially_matched"].sum()),
            "only_in_test_count": int((~group["matched_in_input"]).sum()),
            "only_in_input_count": "",
            "union_provider_count": len(group),
            "fully_matched_pct": _pct(int(group["fully_matched"].sum()), len(group)),
            "partially_matched_pct": _pct(int(group["partially_matched"].sum()), len(group)),
            "only_in_test_pct": _pct(int((~group["matched_in_input"]).sum()), len(group)),
            "only_in_input_pct": 0.0,
            "hospital_match_pct": _pct(int(group["hospital_match"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "clinic_match_pct": _pct(int(group["clinic_match"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "affiliation_match_pct": _pct(int(group["affiliation_match"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "specialty_match_pct_strict": _pct(int(group["specialty_match_strict"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "specialty_match_pct_loose": _pct(int(group["specialty_match_loose"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "phone_match_pct": _pct(int(group["phone_match"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "address_match_pct": _pct(int(group["address_match"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
            "address_match_zip_state_pct": _pct(int(group["address_match_zip_state"].sum()), int(group["matched_in_input"].sum())) if int(group["matched_in_input"].sum()) else 0.0,
        }
        for keys, group in per_clinic_source.groupby(["hospital_name", "clinic_name"], dropna=False)
    ]
    per_clinic_rows = sorted(per_clinic_rows, key=lambda item: (-item["test_provider_count"], str(item.get("hospital_name", ""))))

    summary = {
        "dataset": dataset_label,
        "test_provider_count": len(test_df),
        "input_provider_count_in_scope": hospital_scoped_input_count,
        "candidate_input_provider_count": candidate_input_count,
        "retrieved_input_provider_count": candidate_input_count,
        "retrieved_input_provider_count_name_strong": len(candidate_input_indices_by_bucket["name_strong"]),
        "retrieved_input_provider_count_name_plus_field": len(candidate_input_indices_by_bucket["name_plus_field"]),
        "retrieved_input_provider_count_field_led": len(candidate_input_indices_by_bucket["field_led"]),
        "candidate_input_provider_count_exact": len(candidate_input_indices_by_stage["exact"]),
        "candidate_input_provider_count_relaxed": len(candidate_input_indices_by_stage["relaxed"]),
        "candidate_input_provider_count_fallback_weak": len(candidate_input_indices_by_stage["fallback_weak"]),
        "unique_matched_input_provider_count": unique_matched_input_provider_count,
        "input_scope_counts_by_hospital": {
            item.get("hospital_name", ""): int(item.get("input_provider_count_in_scope", 0) or 0)
            for item in per_hospital_rows
        },
        "retrieval_reached_input_counts_by_hospital": {
            hospital_name: int(len(indexes)) for hospital_name, indexes in candidate_input_indices_by_hospital.items()
        },
        "unique_matched_input_counts_by_hospital": {
            item.get("hospital_name", ""): int(item.get("unique_matched_input_provider_count", 0) or 0)
            for item in per_hospital_rows
        },
        "matched_provider_count": matched_count,
        "matched_provider_count_name_strong": matched_counts_by_bucket["name_strong"],
        "matched_provider_count_name_plus_field": matched_counts_by_bucket["name_plus_field"],
        "matched_provider_count_field_led": matched_counts_by_bucket["field_led"],
        "matched_provider_count_exact": exact_count,
        "matched_provider_count_relaxed": relaxed_count,
        "matched_provider_count_fallback_weak": fallback_weak_count,
        "matched_provider_pct_of_test": matched_pct_of_test,
        "fully_matched_count": fully_matched_count,
        "partially_matched_count": partially_matched_count,
        "only_in_test_count": only_in_test_count,
        "only_in_input_count": only_in_input_count,
        "union_provider_count": union_count,
        "fully_matched_pct_of_union": _pct(fully_matched_count, union_count),
        "partially_matched_pct_of_union": _pct(partially_matched_count, union_count),
        "only_in_test_pct_of_union": _pct(only_in_test_count, union_count),
        "only_in_input_pct_of_union": _pct(only_in_input_count, union_count),
        "reused_input_match_count": reused_input_match_count,
        "reused_input_match_pct_of_matched": _pct(reused_input_match_count, matched_count),
        "hospital_match_pct_of_matched": _pct(int(matches_df["hospital_match"].sum()), matched_count) if matched_count else 0.0,
        "clinic_match_pct_of_matched": _pct(int(matches_df["clinic_match"].sum()), matched_count) if matched_count else 0.0,
        "affiliation_match_pct_of_matched": _pct(int(matches_df["affiliation_match"].sum()), matched_count) if matched_count else 0.0,
        "specialty_match_pct_strict_of_matched": _pct(int(matches_df["specialty_match_strict"].sum()), matched_count) if matched_count else 0.0,
        "specialty_match_pct_loose_of_matched": _pct(int(matches_df["specialty_match_loose"].sum()), matched_count) if matched_count else 0.0,
        "phone_match_pct_of_matched": _pct(int(matches_df["phone_match"].sum()), matched_count) if matched_count else 0.0,
        "address_match_pct_of_matched": _pct(int(matches_df["address_match"].sum()), matched_count) if matched_count else 0.0,
        "address_match_zip_state_pct_of_matched": _pct(int(matches_df["address_match_zip_state"].sum()), matched_count) if matched_count else 0.0,
        "exact_match_count": exact_count,
        "relaxed_match_count": relaxed_count,
        "fallback_weak_match_count": fallback_weak_count,
        "exact_match_pct_of_matched": exact_match_pct,
        "relaxed_match_pct_of_matched": relaxed_match_pct,
        "fallback_weak_match_pct_of_matched": fallback_weak_match_pct,
        "unmatched_after_exact_relaxed_count": unmatched_after_exact_relaxed_count,
        "rescued_by_fallback_count": fallback_weak_count,
        "scope_diagnostics": scope_diagnostics or {},
        "retrieval_audit_rows": retrieval_audit_rows,
        "fallback_match_audit_rows": retrieval_audit_rows,
        "per_hospital": per_hospital_rows,
        "per_clinic": per_clinic_rows,
    }
    _log(
        f"Finished comparison for '{dataset_label}': "
        f"hospital_scoped_input={hospital_scoped_input_count:,}, candidate_input={candidate_input_count:,}, "
        f"unique_matched_input={unique_matched_input_provider_count:,}, "
        f"matched={matched_count:,}, only_in_test={only_in_test_count:,}, only_in_input={only_in_input_count:,}, "
        f"union={union_count:,}, exact={exact_count:,} ({exact_match_pct:.2f}%), "
        f"relaxed={relaxed_count:,} ({relaxed_match_pct:.2f}%), "
        f"fallback_weak={fallback_weak_count:,} ({fallback_weak_match_pct:.2f}%)"
    )
    return summary


def _load_taxonomy_lookup(repo_root: Path) -> dict[str, str]:
    path = repo_root / "data" / "taxonomy_lookup.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _main_report_columns() -> list[str]:
    return [
        "dataset",
        "scope_level",
        "hospital_name",
        "clinic_name",
        "test_provider_count",
        "input_provider_count_in_scope",
        "retrieval_reached_input_provider_count",
        "matched_provider_count",
        "only_in_test_count",
        "only_in_input_count",
        "union_provider_count",
        "fully_matched_pct",
        "partially_matched_pct",
        "only_in_test_pct",
        "only_in_input_pct",
        "hospital_match_pct",
        "clinic_match_pct",
        "affiliation_match_pct",
        "specialty_match_pct_loose",
        "phone_match_pct",
        "address_match_pct",
    ]


def _diagnostics_columns() -> list[str]:
    return [
        "dataset",
        "scope_level",
        "hospital_name",
        "clinic_name",
        "test_provider_count",
        "input_provider_count_in_scope",
        "retrieved_input_provider_count",
        "retrieval_reached_input_provider_count",
        "candidate_input_provider_count",
        "retrieved_input_provider_count_name_strong",
        "retrieved_input_provider_count_name_plus_field",
        "retrieved_input_provider_count_field_led",
        "candidate_input_provider_count_exact",
        "candidate_input_provider_count_relaxed",
        "candidate_input_provider_count_fallback_weak",
        "unique_matched_input_provider_count",
        "source_provider_count_org_npi",
        "source_provider_count_org_entity",
        "missing_vs_source_org_npi",
        "missing_vs_source_org_entity",
        "coverage_vs_source_org_npi_pct",
        "coverage_vs_source_org_entity_pct",
        "matched_provider_count",
        "matched_provider_count_name_strong",
        "matched_provider_count_name_plus_field",
        "matched_provider_count_field_led",
        "matched_provider_count_exact",
        "matched_provider_count_relaxed",
        "matched_provider_count_fallback_weak",
        "reused_input_match_count",
        "reused_input_match_pct_of_matched",
        "unmatched_after_exact_relaxed_count",
        "rescued_by_fallback_count",
        "matched_provider_pct_of_test",
        "fully_matched_count",
        "partially_matched_count",
        "only_in_test_count",
        "only_in_input_count",
        "union_provider_count",
        "fully_matched_pct",
        "partially_matched_pct",
        "only_in_test_pct",
        "only_in_input_pct",
        "hospital_match_pct",
        "clinic_match_pct",
        "affiliation_match_pct",
        "specialty_match_pct_strict",
        "specialty_match_pct_loose",
        "phone_match_pct",
        "address_match_pct",
        "address_match_zip_state_pct",
        "exact_match_count",
        "exact_match_pct",
        "relaxed_match_count",
        "relaxed_match_pct",
    ]


def _flatten_main_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    rows.append(
        {
            "dataset": summary["dataset"],
            "scope_level": "overall",
            "hospital_name": "",
            "clinic_name": "",
            "test_provider_count": summary["test_provider_count"],
            "input_provider_count_in_scope": summary.get("input_provider_count_in_scope", ""),
            "retrieval_reached_input_provider_count": summary.get("retrieved_input_provider_count", ""),
            "matched_provider_count": summary["matched_provider_count"],
            "only_in_test_count": summary["only_in_test_count"],
            "only_in_input_count": summary["only_in_input_count"],
            "union_provider_count": summary["union_provider_count"],
            "fully_matched_pct": summary["fully_matched_pct_of_union"],
            "partially_matched_pct": summary["partially_matched_pct_of_union"],
            "only_in_test_pct": summary["only_in_test_pct_of_union"],
            "only_in_input_pct": summary["only_in_input_pct_of_union"],
            "hospital_match_pct": summary["hospital_match_pct_of_matched"],
            "clinic_match_pct": summary["clinic_match_pct_of_matched"],
            "affiliation_match_pct": summary["affiliation_match_pct_of_matched"],
            "specialty_match_pct_loose": summary["specialty_match_pct_loose_of_matched"],
            "phone_match_pct": summary["phone_match_pct_of_matched"],
            "address_match_pct": summary["address_match_pct_of_matched"],
        }
    )

    for item in summary["per_hospital"]:
        rows.append(
            {
                "dataset": summary["dataset"],
                "scope_level": "hospital",
                "hospital_name": item.get("hospital_name", ""),
                "clinic_name": "",
                "test_provider_count": item["test_provider_count"],
                "input_provider_count_in_scope": item.get("input_provider_count_in_scope", ""),
                "retrieval_reached_input_provider_count": item.get("retrieval_reached_input_provider_count", ""),
                "matched_provider_count": item["matched_provider_count"],
                "only_in_test_count": item["only_in_test_count"],
                "only_in_input_count": item["only_in_input_count"],
                "union_provider_count": item["union_provider_count"],
                "fully_matched_pct": item["fully_matched_pct"],
                "partially_matched_pct": item["partially_matched_pct"],
                "only_in_test_pct": item["only_in_test_pct"],
                "only_in_input_pct": item["only_in_input_pct"],
                "hospital_match_pct": item["hospital_match_pct"],
                "clinic_match_pct": item["clinic_match_pct"],
                "affiliation_match_pct": item["affiliation_match_pct"],
                "specialty_match_pct_loose": item["specialty_match_pct_loose"],
                "phone_match_pct": item["phone_match_pct"],
                "address_match_pct": item["address_match_pct"],
            }
        )

    for item in summary["per_clinic"]:
        rows.append(
            {
                "dataset": summary["dataset"],
                "scope_level": "clinic",
                "hospital_name": item.get("hospital_name", ""),
                "clinic_name": item.get("clinic_name", ""),
                "test_provider_count": item["test_provider_count"],
                "input_provider_count_in_scope": item.get("input_provider_count_in_scope", ""),
                "retrieval_reached_input_provider_count": item.get("retrieval_reached_input_provider_count", ""),
                "matched_provider_count": item["matched_provider_count"],
                "only_in_test_count": item["only_in_test_count"],
                "only_in_input_count": item["only_in_input_count"],
                "union_provider_count": item["union_provider_count"],
                "fully_matched_pct": item["fully_matched_pct"],
                "partially_matched_pct": item["partially_matched_pct"],
                "only_in_test_pct": item["only_in_test_pct"],
                "only_in_input_pct": item["only_in_input_pct"],
                "hospital_match_pct": item["hospital_match_pct"],
                "clinic_match_pct": item["clinic_match_pct"],
                "affiliation_match_pct": item["affiliation_match_pct"],
                "specialty_match_pct_loose": item["specialty_match_pct_loose"],
                "phone_match_pct": item["phone_match_pct"],
                "address_match_pct": item["address_match_pct"],
            }
        )

    return rows


def _flatten_diagnostics_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    def _source_gap_fields(input_count: Any, source_org_npi: Any, source_org_entity: Any) -> dict[str, Any]:
        if input_count == "" or input_count is None:
            return {
                "missing_vs_source_org_npi": "",
                "missing_vs_source_org_entity": "",
                "coverage_vs_source_org_npi_pct": "",
                "coverage_vs_source_org_entity_pct": "",
            }
        input_count = int(input_count)
        source_org_npi = int(source_org_npi or 0)
        source_org_entity = int(source_org_entity or 0)
        return {
            "missing_vs_source_org_npi": max(source_org_npi - input_count, 0),
            "missing_vs_source_org_entity": max(source_org_entity - input_count, 0),
            "coverage_vs_source_org_npi_pct": round((100.0 * input_count / source_org_npi), 2) if source_org_npi else 0.0,
            "coverage_vs_source_org_entity_pct": round((100.0 * input_count / source_org_entity), 2) if source_org_entity else 0.0,
        }

    rows = []
    overall_source_fields = _source_gap_fields(
        summary.get("input_provider_count_in_scope", ""),
        summary.get("source_provider_count_org_npi", 0),
        summary.get("source_provider_count_org_entity", 0),
    )
    rows.append(
        {
            "dataset": summary["dataset"],
            "scope_level": "overall",
            "hospital_name": "",
            "clinic_name": "",
            "test_provider_count": summary["test_provider_count"],
            "input_provider_count_in_scope": summary.get("input_provider_count_in_scope", ""),
            "retrieved_input_provider_count": summary.get("retrieved_input_provider_count", ""),
            "retrieval_reached_input_provider_count": summary.get("retrieved_input_provider_count", ""),
            "candidate_input_provider_count": summary.get("candidate_input_provider_count", ""),
            "retrieved_input_provider_count_name_strong": summary.get("retrieved_input_provider_count_name_strong", ""),
            "retrieved_input_provider_count_name_plus_field": summary.get("retrieved_input_provider_count_name_plus_field", ""),
            "retrieved_input_provider_count_field_led": summary.get("retrieved_input_provider_count_field_led", ""),
            "candidate_input_provider_count_exact": summary.get("candidate_input_provider_count_exact", ""),
            "candidate_input_provider_count_relaxed": summary.get("candidate_input_provider_count_relaxed", ""),
            "candidate_input_provider_count_fallback_weak": summary.get("candidate_input_provider_count_fallback_weak", ""),
            "unique_matched_input_provider_count": summary.get("unique_matched_input_provider_count", ""),
            "source_provider_count_org_npi": summary.get("source_provider_count_org_npi", 0),
            "source_provider_count_org_entity": summary.get("source_provider_count_org_entity", 0),
            **overall_source_fields,
            "matched_provider_count": summary["matched_provider_count"],
            "matched_provider_count_name_strong": summary.get("matched_provider_count_name_strong", 0),
            "matched_provider_count_name_plus_field": summary.get("matched_provider_count_name_plus_field", 0),
            "matched_provider_count_field_led": summary.get("matched_provider_count_field_led", 0),
            "matched_provider_count_exact": summary.get("matched_provider_count_exact", 0),
            "matched_provider_count_relaxed": summary.get("matched_provider_count_relaxed", 0),
            "matched_provider_count_fallback_weak": summary.get("matched_provider_count_fallback_weak", 0),
            "reused_input_match_count": summary.get("reused_input_match_count", 0),
            "reused_input_match_pct_of_matched": summary.get("reused_input_match_pct_of_matched", 0.0),
            "unmatched_after_exact_relaxed_count": summary.get("unmatched_after_exact_relaxed_count", 0),
            "rescued_by_fallback_count": summary.get("rescued_by_fallback_count", 0),
            "matched_provider_pct_of_test": summary["matched_provider_pct_of_test"],
            "fully_matched_count": summary["fully_matched_count"],
            "partially_matched_count": summary["partially_matched_count"],
            "only_in_test_count": summary["only_in_test_count"],
            "only_in_input_count": summary["only_in_input_count"],
            "union_provider_count": summary["union_provider_count"],
            "fully_matched_pct": summary["fully_matched_pct_of_union"],
            "partially_matched_pct": summary["partially_matched_pct_of_union"],
            "only_in_test_pct": summary["only_in_test_pct_of_union"],
            "only_in_input_pct": summary["only_in_input_pct_of_union"],
            "hospital_match_pct": summary["hospital_match_pct_of_matched"],
            "clinic_match_pct": summary["clinic_match_pct_of_matched"],
            "affiliation_match_pct": summary["affiliation_match_pct_of_matched"],
            "specialty_match_pct_strict": summary["specialty_match_pct_strict_of_matched"],
            "specialty_match_pct_loose": summary["specialty_match_pct_loose_of_matched"],
            "phone_match_pct": summary["phone_match_pct_of_matched"],
            "address_match_pct": summary["address_match_pct_of_matched"],
            "address_match_zip_state_pct": summary["address_match_zip_state_pct_of_matched"],
            "exact_match_count": summary["exact_match_count"],
            "exact_match_pct": summary["exact_match_pct_of_matched"],
            "relaxed_match_count": summary["relaxed_match_count"],
            "relaxed_match_pct": summary["relaxed_match_pct_of_matched"],
        }
    )

    for item in summary["per_hospital"]:
        source_fields = _source_gap_fields(
            item.get("input_provider_count_in_scope", ""),
            item.get("source_provider_count_org_npi", 0),
            item.get("source_provider_count_org_entity", 0),
        )
        rows.append(
            {
                "dataset": summary["dataset"],
                "scope_level": "hospital",
                "hospital_name": item.get("hospital_name", ""),
                "clinic_name": "",
                "test_provider_count": item["test_provider_count"],
                "input_provider_count_in_scope": item.get("input_provider_count_in_scope", ""),
                "retrieved_input_provider_count": "",
                "retrieval_reached_input_provider_count": item.get("retrieval_reached_input_provider_count", ""),
                "candidate_input_provider_count": item.get("candidate_input_provider_count", ""),
                "retrieved_input_provider_count_name_strong": "",
                "retrieved_input_provider_count_name_plus_field": "",
                "retrieved_input_provider_count_field_led": "",
                "candidate_input_provider_count_exact": "",
                "candidate_input_provider_count_relaxed": "",
                "candidate_input_provider_count_fallback_weak": "",
                "unique_matched_input_provider_count": item.get("unique_matched_input_provider_count", ""),
                "source_provider_count_org_npi": item.get("source_provider_count_org_npi", 0),
                "source_provider_count_org_entity": item.get("source_provider_count_org_entity", 0),
                **source_fields,
                "matched_provider_count": item["matched_provider_count"],
                "matched_provider_count_name_strong": "",
                "matched_provider_count_name_plus_field": "",
                "matched_provider_count_field_led": "",
                "matched_provider_count_exact": "",
                "matched_provider_count_relaxed": "",
                "matched_provider_count_fallback_weak": "",
                "reused_input_match_count": "",
                "reused_input_match_pct_of_matched": "",
                "unmatched_after_exact_relaxed_count": "",
                "rescued_by_fallback_count": "",
                "matched_provider_pct_of_test": item["matched_provider_pct_of_test"],
                "fully_matched_count": item["fully_matched_count"],
                "partially_matched_count": item["partially_matched_count"],
                "only_in_test_count": item["only_in_test_count"],
                "only_in_input_count": item["only_in_input_count"],
                "union_provider_count": item["union_provider_count"],
                "fully_matched_pct": item["fully_matched_pct"],
                "partially_matched_pct": item["partially_matched_pct"],
                "only_in_test_pct": item["only_in_test_pct"],
                "only_in_input_pct": item["only_in_input_pct"],
                "hospital_match_pct": item["hospital_match_pct"],
                "clinic_match_pct": item["clinic_match_pct"],
                "affiliation_match_pct": item["affiliation_match_pct"],
                "specialty_match_pct_strict": item["specialty_match_pct_strict"],
                "specialty_match_pct_loose": item["specialty_match_pct_loose"],
                "phone_match_pct": item["phone_match_pct"],
                "address_match_pct": item["address_match_pct"],
                "address_match_zip_state_pct": item["address_match_zip_state_pct"],
                "exact_match_count": "",
                "exact_match_pct": "",
                "relaxed_match_count": "",
                "relaxed_match_pct": "",
            }
        )

    for item in summary["per_clinic"]:
        source_fields = _source_gap_fields("", "", "")
        rows.append(
            {
                "dataset": summary["dataset"],
                "scope_level": "clinic",
                "hospital_name": item.get("hospital_name", ""),
                "clinic_name": item.get("clinic_name", ""),
                "test_provider_count": item["test_provider_count"],
                "input_provider_count_in_scope": item.get("input_provider_count_in_scope", ""),
                "retrieved_input_provider_count": "",
                "retrieval_reached_input_provider_count": item.get("retrieval_reached_input_provider_count", ""),
                "candidate_input_provider_count": item.get("candidate_input_provider_count", ""),
                "retrieved_input_provider_count_name_strong": "",
                "retrieved_input_provider_count_name_plus_field": "",
                "retrieved_input_provider_count_field_led": "",
                "candidate_input_provider_count_exact": "",
                "candidate_input_provider_count_relaxed": "",
                "candidate_input_provider_count_fallback_weak": "",
                "unique_matched_input_provider_count": item.get("unique_matched_input_provider_count", ""),
                "source_provider_count_org_npi": item.get("source_provider_count_org_npi", ""),
                "source_provider_count_org_entity": item.get("source_provider_count_org_entity", ""),
                **source_fields,
                "matched_provider_count": item["matched_provider_count"],
                "matched_provider_count_name_strong": "",
                "matched_provider_count_name_plus_field": "",
                "matched_provider_count_field_led": "",
                "matched_provider_count_exact": "",
                "matched_provider_count_relaxed": "",
                "matched_provider_count_fallback_weak": "",
                "reused_input_match_count": "",
                "reused_input_match_pct_of_matched": "",
                "unmatched_after_exact_relaxed_count": "",
                "rescued_by_fallback_count": "",
                "matched_provider_pct_of_test": item["matched_provider_pct_of_test"],
                "fully_matched_count": item["fully_matched_count"],
                "partially_matched_count": item["partially_matched_count"],
                "only_in_test_count": item["only_in_test_count"],
                "only_in_input_count": item["only_in_input_count"],
                "union_provider_count": item["union_provider_count"],
                "fully_matched_pct": item["fully_matched_pct"],
                "partially_matched_pct": item["partially_matched_pct"],
                "only_in_test_pct": item["only_in_test_pct"],
                "only_in_input_pct": item["only_in_input_pct"],
                "hospital_match_pct": item["hospital_match_pct"],
                "clinic_match_pct": item["clinic_match_pct"],
                "affiliation_match_pct": item["affiliation_match_pct"],
                "specialty_match_pct_strict": item["specialty_match_pct_strict"],
                "specialty_match_pct_loose": item["specialty_match_pct_loose"],
                "phone_match_pct": item["phone_match_pct"],
                "address_match_pct": item["address_match_pct"],
                "address_match_zip_state_pct": item["address_match_zip_state_pct"],
                "exact_match_count": "",
                "exact_match_pct": "",
                "relaxed_match_count": "",
                "relaxed_match_pct": "",
            }
        )

    return rows


def _print_overall_summary_table(summaries: list[dict[str, Any]]):
    _log("Overall metrics summary")
    headers = [
        "dataset",
        "test_count",
        "matched_pct",
        "full",
        "partial",
        "only_test",
        "only_input",
        "full_pct",
        "partial_pct",
        "only_test_pct",
        "only_input_pct",
        "hospital_pct",
        "clinic_pct",
        "specialty_strict",
        "specialty_loose",
        "phone_pct",
        "address_pct",
        "address_zip_state",
        "exact_pct",
        "relaxed_pct",
        "fallback_pct",
        "rescued",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary["dataset"],
                f"{summary['test_provider_count']:,}",
                f"{summary['matched_provider_pct_of_test']:.2f}",
                f"{summary['fully_matched_count']:,}",
                f"{summary['partially_matched_count']:,}",
                f"{summary['only_in_test_count']:,}",
                f"{summary['only_in_input_count']:,}",
                f"{summary['fully_matched_pct_of_union']:.2f}",
                f"{summary['partially_matched_pct_of_union']:.2f}",
                f"{summary['only_in_test_pct_of_union']:.2f}",
                f"{summary['only_in_input_pct_of_union']:.2f}",
                f"{summary['hospital_match_pct_of_matched']:.2f}",
                f"{summary['clinic_match_pct_of_matched']:.2f}",
                f"{summary['specialty_match_pct_strict_of_matched']:.2f}",
                f"{summary['specialty_match_pct_loose_of_matched']:.2f}",
                f"{summary['phone_match_pct_of_matched']:.2f}",
                f"{summary['address_match_pct_of_matched']:.2f}",
                f"{summary['address_match_zip_state_pct_of_matched']:.2f}",
                f"{summary['exact_match_pct_of_matched']:.2f}",
                f"{summary['relaxed_match_pct_of_matched']:.2f}",
                f"{summary.get('fallback_weak_match_pct_of_matched', 0.0):.2f}",
                f"{summary.get('rescued_by_fallback_count', 0):,}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line, flush=True)
    print(separator_line, flush=True)
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)), flush=True)


PARQUET_CANDIDATE_COLUMNS = [
    "PROVIDER_FULL_NAME",
    "provider_name",
    "name",
    "full_name",
    "Provider First Name",
    "FIRST_NAME",
    "first_name",
    "Provider Middle Name",
    "MDL_NAME",
    "middle_name",
    "Provider Last Name (Legal Name)",
    "LAST_NAME",
    "last_name",
    "mapped_facility_name",
    "mapped_facility_address",
    "mapped_org_name",
    "mapped_org_address",
    "mapped_clinic_name",
    "Provider Organization Name (Legal Business Name)",
    "organization_name",
    "hospital_name",
    "clinic_name",
    "practice_location_name",
    "location_name",
    "practice_location_address",
    "clinic_address",
    "address",
    "Address",
    "primary_specialty",
    "specialty",
    "primary_specialty_description",
    "Healthcare Provider Primary Taxonomy Description",
    "Provider First Line Business Practice Location Address",
    "Provider Second Line Business Practice Location Address",
    "Provider Business Practice Location Address City Name",
    "Provider Business Practice Location Address State Name",
    "Provider Business Practice Location Address Postal Code",
    "Provider Business Practice Location Address Telephone Number",
    "Provider Business Mailing Address Telephone Number",
    "phone",
    "Phone",
    "PHONE_NUM",
    "NPI",
    "provider_id",
    "ENRLMT_ID",
    "city",
    "state",
    "zip",
    "CITY_NAME",
    "STATE_CD",
    "ZIP_CD",
] + [f"Healthcare Provider Taxonomy Code_{i}" for i in range(1, 16)]

ORGANIZATION_PARQUET_COLUMNS = [
    "NPI",
    "org_entity_id",
    "provider_count",
    "provider_count_entity",
    "mapped_facility_name",
    "mapped_org_name",
    "Provider Organization Name (Legal Business Name)",
    "organization_name",
    "hospital_name",
]


def load_prepared_test_datasets(test_data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    non_optum_path = _find_test_file(test_data_dir, "non_optum_clinic_open_scheduling_database_")
    optum_path = _find_test_file(test_data_dir, "optum_clinic_open_scheduling_database_")
    _log(f"Using non_optum file: {non_optum_path}")
    _log(f"Using optum_clinic file: {optum_path}")

    non_optum_df = _prepare_test_df(_read_dataset(non_optum_path), "non_optum")
    optum_df = _prepare_test_df(_read_dataset(optum_path), "optum_clinic")
    _log("Combining both test datasets")
    combined_df = pd.concat([non_optum_df, optum_df], ignore_index=True)
    _log(f"Combined test dataset size: {len(combined_df):,} providers")
    return non_optum_df, optum_df, combined_df


def load_prepared_input_dataset(input_path: Path, taxonomy_lookup: dict[str, str]) -> pd.DataFrame:
    if input_path.suffix.lower() == ".parquet":
        raw_input_df = _read_parquet_projected(input_path, PARQUET_CANDIDATE_COLUMNS)
    else:
        raw_input_df = _read_dataset(input_path)

    input_identity_df = _prepare_input_identity_df(raw_input_df)
    input_df = _enrich_input_scope_df(input_identity_df)
    return input_df


def _resolve_organizations_path(input_path: Path) -> Path:
    candidate = input_path.with_name(input_path.name.replace("npi_individuals_processed_", "npi_organizations_processed_"))
    if candidate.exists():
        return candidate
    matches = sorted(input_path.parent.glob("npi_organizations_processed_*.parquet"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"Could not find organizations parquet alongside {input_path}")


def load_prepared_organization_dataset(organizations_path: Path) -> pd.DataFrame:
    if organizations_path.suffix.lower() == ".parquet":
        raw_org_df = _read_parquet_projected(organizations_path, ORGANIZATION_PARQUET_COLUMNS)
    else:
        raw_org_df = _read_dataset(organizations_path)
    out = _enrich_input_scope_df(raw_org_df)
    out["org_npi"] = _coalesce_columns(out, ["NPI"]).astype("string").str.strip()
    out["org_entity_id"] = _coalesce_columns(out, ["org_entity_id"]).astype("string").str.strip()
    if "provider_count" not in out.columns:
        out["provider_count"] = 0
    if "provider_count_entity" not in out.columns:
        out["provider_count_entity"] = 0
    out["provider_count"] = pd.to_numeric(out["provider_count"], errors="coerce").fillna(0).astype("int64")
    out["provider_count_entity"] = pd.to_numeric(out["provider_count_entity"], errors="coerce").fillna(0).astype("int64")
    return out


def _aggregate_source_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"source_provider_count_org_npi": 0, "source_provider_count_org_entity": 0}

    org_npi_df = df[df["org_npi"].fillna("").astype("string").str.strip() != ""].copy()
    org_npi_df = org_npi_df.drop_duplicates(subset=["org_npi"], keep="first")
    org_npi_count = int(org_npi_df["provider_count"].sum()) if not org_npi_df.empty else 0

    entity_df = df[df["org_entity_id"].fillna("").astype("string").str.strip() != ""].copy()
    entity_count = 0
    if not entity_df.empty:
        entity_df = entity_df.sort_values(
            by=["provider_count_entity", "org_entity_id"],
            ascending=[False, True],
            kind="mergesort",
        ).drop_duplicates(subset=["org_entity_id"], keep="first")
        entity_count = int(entity_df["provider_count_entity"].sum())

    return {
        "source_provider_count_org_npi": org_npi_count,
        "source_provider_count_org_entity": entity_count,
    }


def _enrich_summary_with_source_counts(
    summary: dict[str, Any],
    test_df: pd.DataFrame,
    organization_df: pd.DataFrame,
    dataset_label: str,
) -> dict[str, Any]:
    org_test_df = test_df[["hospital_name", "hospital_name_canonical"]].drop_duplicates().copy()
    org_crosswalk, org_scope_diagnostics, _ = _build_hospital_crosswalk(org_test_df, organization_df, f"{dataset_label}_orgs")

    overall_mapped_canonicals = {item for values in org_crosswalk.values() for item in values}
    overall_source_df = organization_df[organization_df["hospital_name_canonical"].isin(overall_mapped_canonicals)].copy()
    overall_counts = _aggregate_source_counts(overall_source_df)
    summary["source_provider_count_org_npi"] = overall_counts["source_provider_count_org_npi"]
    summary["source_provider_count_org_entity"] = overall_counts["source_provider_count_org_entity"]

    for row in summary.get("per_hospital", []):
        hospital_canonical = _canonical_hospital_name(row.get("hospital_name", ""))
        mapped_input_canonicals = org_crosswalk.get(hospital_canonical, {hospital_canonical})
        hospital_source_df = organization_df[organization_df["hospital_name_canonical"].isin(mapped_input_canonicals)].copy()
        counts = _aggregate_source_counts(hospital_source_df)
        row["source_provider_count_org_npi"] = counts["source_provider_count_org_npi"]
        row["source_provider_count_org_entity"] = counts["source_provider_count_org_entity"]
        row["input_provider_count_in_scope"] = int(
            summary.get("input_scope_counts_by_hospital", {}).get(row.get("hospital_name", ""), 0)
        )
        row["candidate_input_provider_count"] = ""
        row["retrieval_reached_input_provider_count"] = int(
            summary.get("retrieval_reached_input_counts_by_hospital", {}).get(row.get("hospital_name", ""), 0)
        )
        row["unique_matched_input_provider_count"] = int(
            summary.get("unique_matched_input_counts_by_hospital", {}).get(row.get("hospital_name", ""), 0)
        )

    for row in summary.get("per_clinic", []):
        row["source_provider_count_org_npi"] = ""
        row["source_provider_count_org_entity"] = ""
        row["input_provider_count_in_scope"] = ""
        row["retrieved_input_provider_count"] = ""
        row["retrieval_reached_input_provider_count"] = ""
        row["candidate_input_provider_count"] = ""
        row["unique_matched_input_provider_count"] = ""

    summary.setdefault("source_scope_diagnostics", org_scope_diagnostics)
    return summary


def compare_all_datasets(
    non_optum_df: pd.DataFrame,
    optum_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    input_df: pd.DataFrame,
    organization_df: pd.DataFrame,
    taxonomy_lookup: dict[str, str],
) -> list[dict[str, Any]]:
    summaries = []
    for label, test_df in [
        ("non_optum", non_optum_df),
        ("optum_clinic", optum_df),
        ("combined", combined_df),
    ]:
        scoped_test_df, scoped_input_df, hospital_crosswalk, scope_diagnostics = _scope_dataset_to_hospitals(
            test_df,
            input_df,
            label,
        )
        _log(f"[{label}] Preparing stage-aware candidate search inside hospital-scoped input")
        enriched_scoped_input_df = _enrich_input_candidate_df(scoped_input_df, taxonomy_lookup)
        summary = _compare_single_dataset(
            scoped_test_df,
            enriched_scoped_input_df,
            label,
            hospital_crosswalk=hospital_crosswalk,
            scope_diagnostics=scope_diagnostics,
        )
        summary = _enrich_summary_with_source_counts(summary, scoped_test_df, organization_df, label)
        summaries.append(summary)
    return summaries


def build_report_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    csv_rows = []
    for summary in summaries:
        csv_rows.extend(_flatten_main_summary(summary))
    return pd.DataFrame(csv_rows, columns=_main_report_columns())


def build_report_diagnostics_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    csv_rows = []
    for summary in summaries:
        csv_rows.extend(_flatten_diagnostics_summary(summary))
    return pd.DataFrame(csv_rows, columns=_diagnostics_columns())


def build_hospital_crosswalk_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(summary.get("scope_diagnostics", {}).get("hospital_crosswalk_rows", []))
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "test_hospital_name",
                "test_hospital_canonical",
                "input_hospital_name",
                "input_hospital_canonical",
                "crosswalk_score",
                "match_confidence",
                "shared_core_tokens",
                "shared_distinctive_core_tokens",
                "shared_all_tokens",
                "shared_token_count",
                "shared_distinctive_token_count",
                "single_token_match",
                "distinctive_token_match",
                "sequence_similarity",
                "candidate_rank_for_test_hospital",
                "kept_for_scope",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        by=["dataset", "test_hospital_name", "kept_for_scope", "crosswalk_score", "input_hospital_name"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )


def build_retrieval_audit_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(summary.get("retrieval_audit_rows", []))
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "test_provider_name",
                "test_hospital_name",
                "test_specialty",
                "test_clinic_address",
                "retrieval_bucket",
                "match_strategy",
                "input_provider_name",
                "input_hospital_name",
                "input_specialty",
                "input_phone",
                "input_address",
                "name_similarity",
                "specialty_match_loose",
                "address_match_zip_state",
                "phone_match",
                "candidate_score",
                "second_best_candidate_score",
                "candidate_score_gap",
                "candidate_count",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        by=["dataset", "test_hospital_name", "retrieval_bucket", "match_strategy", "candidate_score"],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )


def build_fallback_match_audit_df(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    return build_retrieval_audit_df(summaries)


def run_comparison_pipeline(
    input_path: Path,
    test_data_dir: Path,
    repo_root: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if not test_data_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")

    _log(f"Using input dataset: {input_path}")
    _log(f"Using test data directory: {test_data_dir}")
    organizations_path = _resolve_organizations_path(input_path)
    _log(f"Using organizations dataset: {organizations_path}")
    taxonomy_lookup = _load_taxonomy_lookup(repo_root)
    _log(f"Loaded taxonomy lookup entries: {len(taxonomy_lookup):,}")
    non_optum_df, optum_df, combined_df = load_prepared_test_datasets(test_data_dir)
    input_df = load_prepared_input_dataset(input_path, taxonomy_lookup)
    organization_df = load_prepared_organization_dataset(organizations_path)
    summaries = compare_all_datasets(non_optum_df, optum_df, combined_df, input_df, organization_df, taxonomy_lookup)
    report_df = build_report_df(summaries)
    diagnostics_df = build_report_diagnostics_df(summaries)
    return summaries, report_df, diagnostics_df


def main():
    parser = argparse.ArgumentParser(description="Compare a provider dataset against curated test data.")
    parser.add_argument("input_path", help="Path to provider dataset to compare, e.g. data/processed_data/npi_individuals_processed_20251206.parquet")
    parser.add_argument(
        "--test-data-dir",
        default="data/test_data",
        help="Directory containing non_optum and optum_clinic CSVs",
    )
    parser.add_argument(
        "--output-csv",
        default="data/test_data/provider_compare_report.csv",
        help="Path to save the comparison results as a combined CSV table",
    )
    parser.add_argument(
        "--output-diagnostics-csv",
        default="data/test_data/provider_compare_report_diagnostics.csv",
        help="Path to save the expanded diagnostics CSV",
    )
    parser.add_argument(
        "--output-hospital-crosswalk-csv",
        default="data/test_data/provider_compare_hospital_crosswalk.csv",
        help="Path to save the hospital crosswalk audit CSV",
    )
    parser.add_argument(
        "--output-fallback-audit-csv",
        default="data/test_data/provider_compare_fallback_audit.csv",
        help="Path to save the provider retrieval audit CSV (legacy option name)",
    )
    parser.add_argument(
        "--output-retrieval-audit-csv",
        default="",
        help="Path to save the provider retrieval audit CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = (repo_root / args.input_path).resolve() if not Path(args.input_path).is_absolute() else Path(args.input_path)
    test_data_dir = (repo_root / args.test_data_dir).resolve() if not Path(args.test_data_dir).is_absolute() else Path(args.test_data_dir)

    summaries, report_df, diagnostics_df = run_comparison_pipeline(
        input_path=input_path,
        test_data_dir=test_data_dir,
        repo_root=repo_root,
    )
    _print_overall_summary_table(summaries)
    crosswalk_df = build_hospital_crosswalk_df(summaries)
    retrieval_audit_df = build_retrieval_audit_df(summaries)

    output_path = Path(args.output_csv)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    _log(f"Wrote combined CSV report: {output_path}")

    diagnostics_output_path = Path(args.output_diagnostics_csv)
    if not diagnostics_output_path.is_absolute():
        diagnostics_output_path = repo_root / diagnostics_output_path
    diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_df.to_csv(diagnostics_output_path, index=False)
    _log(f"Wrote report diagnostics CSV: {diagnostics_output_path}")

    crosswalk_output_path = Path(args.output_hospital_crosswalk_csv)
    if not crosswalk_output_path.is_absolute():
        crosswalk_output_path = repo_root / crosswalk_output_path
    crosswalk_output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk_df.to_csv(crosswalk_output_path, index=False)
    _log(f"Wrote hospital crosswalk audit CSV: {crosswalk_output_path}")

    retrieval_output_arg = args.output_retrieval_audit_csv or args.output_fallback_audit_csv
    retrieval_output_path = Path(retrieval_output_arg)
    if not retrieval_output_path.is_absolute():
        retrieval_output_path = repo_root / retrieval_output_path
    retrieval_output_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_audit_df.to_csv(retrieval_output_path, index=False)
    _log(f"Wrote provider retrieval audit CSV: {retrieval_output_path}")


if __name__ == "__main__":
    main()
