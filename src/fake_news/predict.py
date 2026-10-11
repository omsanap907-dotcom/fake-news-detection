"""Predict the dataset label for one piece of text."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
from fake_news.model import predict_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Headline or article text to classify")
    parser.add_argument("--model", default="artifacts/fake_news_model.joblib")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(
            f"Model not found: {model_path}. Train a model first; see README.md.",
            file=sys.stderr,
        )
        return 2
    try:
        result = predict_text(joblib.load(model_path), args.text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Predicted dataset label: {result.label}")
    print("Model scores:")
    for label, score in sorted(result.scores.items()):
        print(f"  {label}: {score:.3f}")
    print(f"Note: {result.disclaimer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
