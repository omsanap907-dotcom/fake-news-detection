from pathlib import Path

import pandas as pd
import pytest

from fake_news.prepare_data import combine_datasets


def test_combine_maps_labels_and_deduplicates(tmp_path: Path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "hybrid.csv"
    pd.DataFrame({
        "headline": ["Claim A is false", "Official report is released", "Duplicate story", "City publishes transport schedule"],
        "truth": ["false", "true", "false", "true"],
    }).to_csv(first, index=False)
    pd.DataFrame({
        "article": ["claim a is false", "Another fabricated claim", "Conflicting duplicate"],
        "class": ["FAKE", "FAKE", "REAL"],
    }).to_csv(second, index=False)
    # Add a real row that shares text with the first dataset's fake row, to check conflict removal.
    second_df = pd.read_csv(second)
    second_df.loc[2, "article"] = "Duplicate story"
    second_df.loc[2, "class"] = "REAL"
    second_df.to_csv(second, index=False)

    report = combine_datasets(
        str(first), str(second), str(output),
        first_text_col="headline", first_label_col="truth",
        second_text_col="article", second_label_col="class",
        first_fake_label="false", first_real_label="true",
        second_fake_label="FAKE", second_real_label="REAL",
    )
    result = pd.read_csv(output)
    assert set(result["label"]) == {"FAKE", "REAL"}
    assert not result["text"].str.casefold().eq("duplicate story").any()
    assert report["conflicting_rows_removed"] == 2
    assert report["duplicate_rows_removed"] == 1


def test_rejects_unknown_source_labels(tmp_path: Path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"text": ["A fake story", "A real story"], "label": ["FAKE", "REAL"]}).to_csv(first, index=False)
    pd.DataFrame({"text": ["story one", "story two"], "label": ["mostly true", "FAKE"]}).to_csv(second, index=False)
    with pytest.raises(ValueError, match="unmapped labels"):
        combine_datasets(str(first), str(second), str(tmp_path / "out.csv"))
