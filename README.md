# Fake News Detection — Learning Project

An educational prototype for experimenting with news-text classification and misinformation signals. It is not a fact-checking authority.

## Current capabilities

- Validates labeled CSV data, trains a TF-IDF + logistic-regression baseline, and reports held-out accuracy, per-class precision/recall/F1 and a confusion matrix.
- Provides a Flask API for text analysis and selectable-text PDF uploads.
- Flags selected sensational or urgency-oriented phrases as writing-style signals, never as proof that a claim is false.
- Runs without a trained model for API/UI development; predictions appear once a model artifact is available.
- Includes automated tests.

## Important limitation

A model prediction is not proof that a story is true or false. The model learns patterns correlated with the training labels and may learn topic, publisher, style, or dataset artifacts. Suspicious wording is not proof either. Verify claims with independent, reliable sources. Do not use this prototype for consequential decisions about people.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/omsanap907-dotcom/fake-news-detection.git
cd fake-news-detection
python -m venv .venv
```

Activate with PowerShell on Windows: `.venv\Scripts\Activate.ps1`; on macOS/Linux: `source .venv/bin/activate`.

```bash
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Dataset and training

Place a CSV at `data/news.csv` with columns `text` and `label`; labels must be `FAKE` or `REAL` (case-insensitive). See `data/README.md`. Record dataset source and license; don't commit data unless redistribution is permitted.

Train/evaluate with:

```bash
python -m fake_news.train --data data/news.csv
```

This writes `artifacts/fake_news_model.joblib` and `artifacts/evaluation.json`. Scores measure agreement with dataset labels, not real-world factual truth. Inspect class-wise metrics, duplicates, and source/topic leakage.

## Start API

```bash
python -m fake_news.api
```

Health: `GET http://127.0.0.1:5000/api/health`

Text analysis: `POST http://127.0.0.1:5000/api/analyze`, JSON body:

```json
{"text": "Paste an article or claim with at least 20 characters here."}
```

PDF analysis: `POST http://127.0.0.1:5000/api/analyze-pdf` as `multipart/form-data`, field name `file`. Limits: 8 MB, 40 pages, and 200,000 extracted characters. Password-protected files and scanned/image-only PDFs are not supported yet; OCR is a later step. Uploaded files are processed in memory rather than saved to disk.

The API can run before the model is trained. Once a model exists at `artifacts/fake_news_model.joblib`, restart the API to load it. Model class scores are not guaranteed to be calibrated probabilities.

## Tests

```bash
pytest -q
```

## Roadmap

1. Add a React UI.
2. Optionally add OCR for scanned PDFs with resource limits.
3. Add optional Gemini-generated summaries while keeping the API key server-side.
4. Add evidence retrieval with source citations and human review. Never conflate generated commentary with verified evidence.
