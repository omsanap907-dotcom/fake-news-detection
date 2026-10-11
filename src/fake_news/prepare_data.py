"""Combine two labeled news CSVs into a normalized, deduplicated hybrid dataset.

Labels from datasets are not assumed to have identical semantics. This tool requires
explicit source-column names and lets the user map each source's fake/real labels.
Rows with contradictory labels for the same normalized text are excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _read_source(path: str, text_col: str, label_col: str, fake_label: str, real_label: str, source_name: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{source_name} dataset not found: {file_path}")
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read {source_name} CSV: {exc}") from exc
    missing = {text_col, label_col} - set(frame.columns)
    if missing:
        raise ValueError(f"{source_name} dataset is missing column(s): {', '.join(sorted(missing))}")
    frame = frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "source_label"})
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["source_label"] = frame["source_label"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0].copy()
    fake_norm, real_norm = str(fake_label).strip().casefold(), str(real_label).strip().casefold()
    if not fake_norm or not real_norm or fake_norm == real_norm:
        raise ValueError(f"{source_name} fake/real label mappings must be non-empty and different.")
    normalized = frame["source_label"].str.casefold()
    unknown = sorted(frame.loc[~normalized.isin([fake_norm, real_norm]), "source_label"].unique().tolist())
    if unknown:
        preview = ", ".join(repr(value) for value in unknown[:10])
        raise ValueError(f"{source_name} has unmapped labels: {preview}. Specify the correct fake/real label values.")
    frame["label"] = normalized.map({fake_norm: "FAKE", real_norm: "REAL"})
    frame["source"] = source_name
    frame["normalized_text"] = frame["text"].str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    return frame[["text", "label", "source", "normalized_text"]]


def combine_datasets(
    first_path: str, second_path: str, output_path: str,
    first_text_col: str = "text", first_label_col: str = "label",
    second_text_col: str = "text", second_label_col: str = "label",
    first_fake_label: str = "FAKE", first_real_label: str = "REAL",
    second_fake_label: str = "FAKE", second_real_label: str = "REAL",
) -> dict[str, object]:
    first = _read_source(first_path, first_text_col, first_label_col, first_fake_label, first_real_label, "dataset_1")
    second = _read_source(second_path, second_text_col, second_label_col, second_fake_label, second_real_label, "dataset_2")
    combined = pd.concat([first, second], ignore_index=True)
    # Exclude duplicate text that has conflicting labels; keeping it would contaminate evaluation.
    conflicts = combined.groupby("normalized_text")["label"].nunique()
    conflicting_texts = set(conflicts[conflicts > 1].index)
    conflict_rows = int(combined["normalized_text"].isin(conflicting_texts).sum())
    combined = combined[~combined["normalized_text"].isin(conflicting_texts)].copy()
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["normalized_text"], keep="first")
    duplicate_rows_removed = before_dedup - len(combined)
    result = combined[["text", "label"]].reset_index(drop=True)
    if result.empty or result["label"].nunique() != 2:
        raise ValueError("The hybrid dataset must contain non-empty examples for both FAKE and REAL labels after deduplication.")
    counts = {str(label): int(count) for label, count in result["label"].value_counts().items()}
    if min(counts.values()) < 2:
        raise ValueError("The hybrid dataset needs at least two examples for each label.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return {
        "output": str(destination),
        "input_rows": int(len(first) + len(second)),
        "output_rows": int(len(result)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "conflicting_rows_removed": conflict_rows,
        "label_counts": counts,
        "warning": "Combining datasets does not guarantee reliable labels or generalization. Record dataset sources/licenses and evaluate for source/topic leakage.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="First source CSV")
    parser.add_argument("--second", required=True, help="Second source CSV")
    parser.add_argument("--output", default="data/hybrid_news.csv")
    parser.add_argument("--first-text-col", default="text")
    parser.add_argument("--first-label-col", default="label")
    parser.add_argument("--second-text-col", default="text")
    parser.add_argument("--second-label-col", default="label")
    parser.add_argument("--first-fake-label", default="FAKE")
    parser.add_argument("--first-real-label", default="REAL")
    parser.add_argument("--second-fake-label", default="FAKE")
    parser.add_argument("--second-real-label", default="REAL")
    args = parser.parse_args()
    try:
        report = combine_datasets(
            args.first, args.second, args.output,
            args.first_text_col, args.first_label_col, args.second_text_col, args.second_label_col,
            args.first_fake_label, args.first_real_label, args.second_fake_label, args.second_real_label,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(pd.Series(report).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
