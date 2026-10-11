# Fake News Detection — TruthLens Learning Project

An educational prototype for experimenting with news-text classification and misinformation signals. It is not a fact-checking authority.

## Project structure

- `src/fake_news/`: Python data validation, TF-IDF + logistic regression model, training, prediction, PDF extraction, and Flask API.
- `frontend/`: React + Vite interface for text and PDF analysis.
- `tests/`: Python tests for data validation, model, API, and PDF handling.
- `.github/workflows/tests.yml`: CI for Python tests and frontend production build.

## What works in this branch

- Validates labeled CSV data and trains/evaluates a baseline classifier.
- Serves text-analysis and PDF-upload endpoints from Flask.
- Extracts selectable PDF text in memory, with upload/page/character limits.
- Flags a small set of rhetorical phrases as style signals.
- Provides a responsive React UI, result cards, optional local-browser history, and a button for Gemini contextual analysis with optional Google Search grounding links.
- Gemini is optional and its API key is read only by the backend from an environment variable.
- Works without a trained model for UI development, but cannot produce ML predictions until a model artifact exists.

## Important limitations

A model prediction is not proof that a story is true or false. The classifier learns correlations in the training dataset and may learn source, topic, style, duplicate, or other dataset artifacts. The phrase scanner is deliberately simple and cannot establish truth. Gemini output can also be wrong. Verify claims against independent, reliable sources; do not use this prototype for consequential decisions about people.

## Python setup

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
```

If PowerShell blocks environment activation, use `.venv\Scripts\python.exe -m pip install -e ".[test]"` and run tools with `.venv\Scripts\python.exe`.

## Dataset and training

Place a CSV at `data/news.csv` with `text` and `label` columns. Labels must be `FAKE` or `REAL` (case-insensitive). See `data/README.md`. Check the dataset source and license before downloading, using, or redistributing it. The repository does not ship a real dataset.

```powershell
python -m fake_news.train --data data/news.csv
```

This writes `artifacts/fake_news_model.joblib` and `artifacts/evaluation.json`. The metrics measure agreement with dataset labels, not real-world truth. Review per-class metrics and check for duplicates and source/topic leakage.

## Start the API

From the repository root, with the virtual environment active:

```powershell
python -m fake_news.api
```

Health endpoint: `GET http://127.0.0.1:5000/api/health`

Text endpoint: `POST http://127.0.0.1:5000/api/analyze`, JSON body:

```json
{"text":"A news article or claim of at least 20 characters"}
```

PDF endpoint: `POST http://127.0.0.1:5000/api/analyze-pdf` using `multipart/form-data`, field name `file`. Current limits: 8 MB, 40 pages, and 200,000 extracted characters. Password-protected PDFs and scanned/image-only PDFs are not supported yet; OCR is future work. The API returns extracted PDF text to the client so the optional Gemini action can use it; be mindful of sensitive documents.

## Optional Gemini analysis

Create a server-side environment variable before starting Flask. Never place the key in React or commit it to GitHub.

PowerShell for the current terminal:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
$env:GEMINI_MODEL="gemini-2.5-flash"
python -m fake_news.api
```

The model name can be changed using `GEMINI_MODEL`. Without `GEMINI_API_KEY`, the Gemini endpoint returns a clear configuration error; the rest of the app continues to work. Requests require internet access and are subject to the provider's availability, quotas, and terms.

## Run React

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The frontend uses `http://127.0.0.1:5000` by default. To override it, create `frontend/.env.local` with:

```text
VITE_API_BASE_URL=http://127.0.0.1:5000
```

Build the frontend for production with `npm run build`.

## Tests

From the repository root:

```powershell
pytest -q
```

GitHub Actions also builds the React frontend and runs Python tests on pushes and pull requests.

## Roadmap

- OCR for scanned PDFs with resource limits.
- Evidence retrieval with source citations and claim-level comparisons.
- Better dataset provenance, duplicate checks, leakage-aware evaluation, and robustness testing.
- Deployment configuration, authentication/rate limits if hosted publicly, and privacy review.
