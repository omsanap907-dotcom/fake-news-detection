"""Baseline TF-IDF + logistic regression classifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class Prediction:
    label: str
    scores: dict[str, float]
    disclaimer: str = (
        "This is a dataset-based text classification prediction, not a fact-check. "
        "Verify the claim using independent, trustworthy sources."
    )


def build_model(random_state: int = 42) -> Pipeline:
    """Create a reproducible baseline model; it must be trained before prediction."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def predict_text(model: Any, text: str) -> Prediction:
    """Predict a label for non-empty text and return per-class model scores."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a non-empty headline or article text.")
    predicted = str(model.predict([text.strip()])[0])
    scores_raw = model.predict_proba([text.strip()])[0]
    classes = model.named_steps["classifier"].classes_
    scores = {str(label): float(score) for label, score in zip(classes, scores_raw)}
    return Prediction(label=predicted, scores=scores)
