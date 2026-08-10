#!/usr/bin/env python3
"""Create aggregate quality summaries for the CMS provider baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROVIDER_COLUMNS = [
    "baseline_view",
    "provider_id",
    "npi",
    "specialty_normalized",
    "specialty_group",
    "provider_type_group",
    "provider_practice_address_1",
    "provider_practice_city",
    "provider_practice_state",
    "provider_practice_zip5",
    "provider_practice_phone",
    "mapped_clinic_id",
    "mapped_clinic_name",
    "mapped_system_id",
    "mapped_system_name",
    "mapping_confidence_tier",
    "primary_is_tied",
    "primary_is_ambiguous",
    "cms_current_active_flag",
    "cms_currentness_tier",
    "ppef_snapshot_date",
    "pecos_snapshot_date",
    "npi_registry_snapshot_date",
]


def latest(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} under {path}")
    return matches[-1]


def available_columns(path: Path, requested: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    columns = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in requested if column in columns]


def text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[column].astype("string").fillna("").str.strip()


def boolean(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype("string").fillna("").str.lower().isin({"1", "true", "t", "yes", "y"})


def populated(df: pd.DataFrame, columns: list[str], require_all: bool = False) -> pd.Series:
    masks = [text(df, column).ne("") for column in columns]
    if not masks:
        return pd.Series(False, index=df.index, dtype="bool")
    result = masks[0]
    for mask in masks[1:]:
        result = result & mask if require_all else result | mask
    return result


def pct(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator) * 100, 2) if denominator else 0.0


def count_by(
    df: pd.DataFrame,
    group_columns: list[str],
    label: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    work = df.copy()
    for column in group_columns:
        work[column] = text(work, column).replace("", "Unknown")
    work["_has_phone"] = populated(work, ["provider_practice_phone"])
    work["_has_address"] = populated(
        work,
        [
            "provider_practice_address_1",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
        ],
        require_all=True,
    )
    grouped = (
        work.groupby(group_columns, dropna=False)
        .agg(
            provider_count=("provider_id", "nunique"),
            npi_count=("npi", "nunique"),
            phone_complete_count=("_has_phone", "sum"),
            address_complete_count=("_has_address", "sum"),
        )
        .reset_index()
        .sort_values(["provider_count", *group_columns], ascending=[False, *([True] * len(group_columns))])
    )
    grouped["phone_complete_pct"] = grouped.apply(
        lambda row: pct(row["phone_complete_count"], row["provider_count"]), axis=1
    )
    grouped["address_complete_pct"] = grouped.apply(
        lambda row: pct(row["address_complete_count"], row["provider_count"]), axis=1
    )
    grouped.insert(0, "summary_level", label)
    return grouped.head(top_n).copy() if top_n else grouped


def distribution(df: pd.DataFrame, column: str, provider_total: int) -> list[dict]:
    values = text(df, column).replace("", "Unknown")
    counts = values.value_counts(dropna=False)
    return [
        {column: str(key), "provider_count": int(value), "provider_pct": pct(value, provider_total)}
        for key, value in counts.items()
    ]


def organization_quality(path: Path, entity_type: str) -> dict:
    count_column = "provider_count" if entity_type == "clinic" else "provider_count_entity"
    requested = [
        f"{entity_type}_id",
        count_column,
        f"{entity_type}_generic_name_flag",
        "duplicate_address_group_size",
        "rollup_quality_warning",
        "is_hospital",
    ]
    frame = pd.read_parquet(path, columns=available_columns(path, requested))
    counts = pd.to_numeric(frame.get(count_column, 0), errors="coerce").fillna(0)
    duplicate_size = pd.to_numeric(
        frame.get("duplicate_address_group_size", 0), errors="coerce"
    ).fillna(0)
    return {
        "entity_type": entity_type,
        "entity_count": int(len(frame)),
        "entities_with_providers": int(counts.gt(0).sum()),
        "entities_with_providers_pct": pct(counts.gt(0).sum(), len(frame)),
        "generic_name_count": int(boolean(frame, f"{entity_type}_generic_name_flag").sum()),
        "quality_warning_count": int(text(frame, "rollup_quality_warning").ne("").sum()),
        "duplicate_address_cluster_count": int(duplicate_size.gt(1).sum()),
        "hospital_flag_count": int(boolean(frame, "is_hospital").sum()),
    }


def render_markdown(summary: dict) -> str:
    coverage = summary["field_completeness"]
    lines = [
        "# CMS Provider Baseline Quality Summary",
        "",
        f"Generated: {summary['generated_at']}",
        f"Baseline view: `{summary['baseline_view']}`",
        "",
        "## Provider Universe",
        "",
        f"- Providers: {summary['provider_count']:,}",
        f"- Unique NPIs: {summary['npi_count']:,}",
        f"- Current-active providers: {summary['current_active_provider_count']:,}",
        "",
        "## Field Completeness",
        "",
        "| Field | Complete providers | Completeness |",
        "|---|---:|---:|",
    ]
    for field_name, values in coverage.items():
        lines.append(
            f"| {field_name.replace('_', ' ').title()} | {values['provider_count']:,} | {values['provider_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Mapping Quality",
            "",
            f"- Tied primary affiliation: {summary['tied_primary_count']:,}",
            f"- Ambiguous primary affiliation: {summary['ambiguous_primary_count']:,}",
            "",
            "## Organization Rollups",
            "",
            "| Entity | Count | With providers | Quality warnings |",
            "|---|---:|---:|---:|",
        ]
    )
    for values in summary["organization_quality"]:
        lines.append(
            f"| {values['entity_type'].title()} | {values['entity_count']:,} | {values['entities_with_providers']:,} | {values['quality_warning_count']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    processed_dir = args.processed_dir
    baseline_path = args.baseline_path or latest(
        processed_dir, "ppef_individuals_comparison_baseline_*.parquet"
    )
    clinic_path = args.clinic_path or latest(processed_dir, "pecos_clinic_rollup_*.parquet")
    system_path = args.system_path or latest(processed_dir, "pecos_system_rollup_*.parquet")
    columns = available_columns(baseline_path, PROVIDER_COLUMNS)
    frame = pd.read_parquet(
        baseline_path,
        columns=columns,
        filters=[("baseline_view", "==", args.baseline_view)],
    )
    if frame.empty:
        raise RuntimeError(f"Baseline view {args.baseline_view!r} contains no rows")
    frame["provider_id"] = text(frame, "provider_id")
    frame["npi"] = text(frame, "npi")
    provider_total = int(frame["provider_id"].replace("", pd.NA).nunique())
    npi_total = int(frame["npi"].replace("", pd.NA).nunique())
    current_active = int(boolean(frame, "cms_current_active_flag").sum())

    field_specs = {
        "npi": ["npi"],
        "specialty": ["specialty_normalized", "specialty_group"],
        "phone": ["provider_practice_phone"],
        "address": [
            "provider_practice_address_1",
            "provider_practice_city",
            "provider_practice_state",
            "provider_practice_zip5",
        ],
        "clinic_mapping": ["mapped_clinic_id", "mapped_clinic_name"],
        "system_mapping": ["mapped_system_id", "mapped_system_name"],
    }
    field_completeness = {}
    for name, fields in field_specs.items():
        mask = populated(frame, fields, require_all=True)
        count = int(frame.loc[mask, "provider_id"].replace("", pd.NA).nunique())
        field_completeness[name] = {
            "provider_count": count,
            "provider_pct": pct(count, provider_total),
        }

    snapshot_columns = [
        "ppef_snapshot_date",
        "pecos_snapshot_date",
        "npi_registry_snapshot_date",
    ]
    snapshots = {
        column: next((value for value in text(frame, column).unique() if value), "")
        for column in snapshot_columns
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_path": str(baseline_path),
        "baseline_view": args.baseline_view,
        "provider_count": provider_total,
        "npi_count": npi_total,
        "current_active_provider_count": current_active,
        "current_active_provider_pct": pct(current_active, provider_total),
        "tied_primary_count": int(boolean(frame, "primary_is_tied").sum()),
        "ambiguous_primary_count": int(boolean(frame, "primary_is_ambiguous").sum()),
        "field_completeness": field_completeness,
        "mapping_confidence_distribution": distribution(
            frame, "mapping_confidence_tier", provider_total
        ),
        "currentness_distribution": distribution(frame, "cms_currentness_tier", provider_total),
        "snapshots": snapshots,
        "organization_quality": [
            organization_quality(clinic_path, "clinic"),
            organization_quality(system_path, "system"),
        ],
    }

    state = count_by(frame, ["provider_practice_state"], "state")
    specialty = count_by(
        frame,
        ["specialty_group", "specialty_normalized", "provider_type_group"],
        "specialty",
        top_n=args.top_n,
    )
    system = count_by(
        frame,
        ["mapped_system_id", "mapped_system_name"],
        "system",
        top_n=args.top_n,
    )

    date_match = "".join(character for character in baseline_path.stem if character.isdigit())[-8:]
    suffix = date_match or datetime.now().strftime("%Y%m%d")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_json": args.output_dir / f"cms_baseline_summary_{suffix}.json",
        "summary_markdown": args.output_dir / f"cms_baseline_summary_{suffix}.md",
        "by_state": args.output_dir / f"cms_baseline_by_state_{suffix}.csv",
        "by_specialty": args.output_dir / f"cms_baseline_by_specialty_{suffix}.csv",
        "by_system": args.output_dir / f"cms_baseline_by_system_{suffix}.csv",
    }
    outputs["summary_json"].write_text(json.dumps(summary, indent=2))
    outputs["summary_markdown"].write_text(render_markdown(summary))
    state.to_csv(outputs["by_state"], index=False)
    specialty.to_csv(outputs["by_specialty"], index=False)
    system.to_csv(outputs["by_system"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_data"))
    parser.add_argument("--baseline-path", type=Path)
    parser.add_argument("--clinic-path", type=Path)
    parser.add_argument("--system-path", type=Path)
    parser.add_argument("--baseline-view", default="current_active_primary")
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis"))
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    outputs = analyze(parse_args())
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
