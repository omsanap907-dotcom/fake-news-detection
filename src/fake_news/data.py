"""Input validation for labeled text data."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

VALID_LABELS = {"FAKE", "REAL"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV containing non-empty `text` and `label` columns."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset CSV: {exc}") from exc

    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError("Dataset is missing required column(s): " + ", ".join(sorted(missing)))

    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label"] = frame["label"].fillna("").astype(str).str.strip().str.upper()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")].reset_index(drop=True)

    invalid_labels = sorted(set(frame["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(
            "Labels must be FAKE or REAL; found: " + ", ".join(invalid_labels)
        )
    if frame.empty:
        raise ValueError("Dataset has no non-empty labeled examples.")
    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain examples for both FAKE and REAL labels.")
    if frame["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples for a train/test split.")

    return frame
