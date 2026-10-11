from pathlib import Path

import pandas as pd
import pytest

from fake_news.data import load_dataset


def test_load_dataset_normalizes_labels(tmp_path: Path) -> None:
    file = tmp_path / "news.csv"
    pd.DataFrame(
        {"text": ["Claim one", "Claim two", "Claim three", "Claim four"],
         "label": ["fake", "REAL", "FAKE", "real"]}
    ).to_csv(file, index=False)
    result = load_dataset(file)
    assert result["label"].tolist() == ["FAKE", "REAL", "FAKE", "REAL"]


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    file = tmp_path / "news.csv"
    pd.DataFrame({"headline": ["Example"], "label": ["FAKE"]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="missing required"):
        load_dataset(file)


def test_empty_text_rows_are_dropped(tmp_path: Path) -> None:
    file = tmp_path / "news.csv"
    pd.DataFrame(
        {"text": [" ", "Fake A", "Real A", "Fake B", "Real B"],
         "label": ["FAKE", "FAKE", "REAL", "FAKE", "REAL"]}
    ).to_csv(file, index=False)
    result = load_dataset(file)
    assert len(result) == 4
