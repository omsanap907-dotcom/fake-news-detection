"""Train and evaluate the baseline model using a labeled CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Support `python -m fake_news.train` after installing the package or adding src to PYTHONPATH.
from fake_news.data import load_dataset
from fake_news.model import build_model


def train(dataset_path: str, model_path: str, report_path: str, test_size: float = 0.2) -> dict:
    frame = load_dataset(dataset_path)
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")

    # Stratification keeps both labels represented, when sample counts allow it.
    train_text, test_text, train_labels, test_labels = train_test_split(
        frame["text"],
        frame["label"],
        test_size=test_size,
        random_state=42,
        stratify=frame["label"],
    )
    model = build_model()
    model.fit(train_text, train_labels)
    predictions = model.predict(test_text)

    report = {
        "dataset": str(Path(dataset_path)),
        "rows": int(len(frame)),
        "train_rows": int(len(train_text)),
        "test_rows": int(len(test_text)),
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "classification_report": classification_report(
            test_labels, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": list(model.named_steps["classifier"].classes_),
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=model.named_steps["classifier"].classes_,
        ).tolist(),
        "caveat": (
            "Results measure agreement with this dataset's labels, not real-world truth. "
            "Performance may be inflated by duplicates or dataset/source leakage."
        ),
    }

    model_file = Path(model_path)
    report_file = Path(report_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/news.csv", help="Path to labeled CSV")
    parser.add_argument("--model", default="artifacts/fake_news_model.joblib")
    parser.add_argument("--report", default="artifacts/evaluation.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    try:
        result = train(args.data, args.model, args.report, args.test_size)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Dataset rows: {result['rows']}")
    print(f"Test accuracy: {result['accuracy']:.3f} (not a measure of factual truth)")
    print(f"Saved model: {args.model}")
    print(f"Saved report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
