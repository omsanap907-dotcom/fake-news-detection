# Fake News Detection — Learning Project

A small, explainable starter project for experimenting with text classification and misinformation signals. This is an educational prototype, not a fact-checking authority.

## What it does

- Validates a labeled CSV with `text` and `label` columns.
- Trains a baseline TF-IDF + logistic regression classifier.
- Evaluates on a held-out split with accuracy, per-class precision/recall/F1, and a confusion matrix.
- Classifies a headline/article text locally and prints model scores.
- Includes input validation tests and a GitHub Actions test workflow.

## Important limitation

**A model prediction is not proof that a story is true or false.** This baseline learns patterns correlated with the labels in its training dataset. It may mistake writing style, publisher, topic, or dataset artifacts for truth. Verify claims with independent, trustworthy sources. Do not use this prototype for consequential decisions about people.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/omsanap907-dotcom/fake-news-detection.git
cd fake-news-detection
python -m venv .venv
```

Activate the environment:

- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install the project and test dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Dataset

Put a CSV at `data/news.csv` with these columns:

- `text`: headline or article text
- `label`: `FAKE` or `REAL` (case-insensitive)

See [`data/README.md`](data/README.md) for the schema. Record the dataset's source and license before using it. The project intentionally does not invent a training dataset. Don't commit a dataset unless its license permits it and you're comfortable making it public.

## Train and evaluate

```bash
python -m fake_news.train --data data/news.csv
```

This writes the model to `artifacts/fake_news_model.joblib` and the evaluation report to `artifacts/evaluation.json`. The report measures agreement with the dataset labels, not real-world factual truth. Inspect class-level metrics and possible duplicates/leakage rather than relying on accuracy alone.

To choose different paths:

```bash
python -m fake_news.train --data data/news.csv --model artifacts/model.joblib --report artifacts/report.json --test-size 0.2
```

## Predict

After training:

```bash
python -m fake_news.predict "Paste a headline or article excerpt here"
```

The output is a model prediction and class scores—not a verified fact-check. The scores are not guaranteed to be calibrated probabilities.

## Run tests

```bash
pytest -q
```

## Roadmap

1. Establish a reproducible baseline and understand its failure cases.
2. Add a local user interface after the training pipeline is reliable.
3. Compare models and test for duplicate/source/topic leakage.
4. Explore evidence retrieval and source/context signals with transparent citations and human review.
