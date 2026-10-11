"""Local API for the fake-news learning project.

The classifier only reports learned text patterns. It must not be presented as
a definitive fact-check; independent sources and human review remain necessary.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

from fake_news.pdf_text import PdfTextError, extract_pdf_text

DEFAULT_MODEL = Path("artifacts/fake_news_model.joblib")
MAX_REQUEST_BYTES = 8 * 1024 * 1024


def suspicious_language(text: str) -> list[dict[str, str]]:
    """Flag sensational phrasing as a writing-style signal, never as proof of fakery."""
    patterns = {
        "urgency": ["share immediately", "before it gets deleted", "act now"],
        "sensationalism": ["shocking truth", "wake up", "you won't believe", "secret they don't want you to know"],
        "unverified_authority": ["scientists confirm", "experts say", "officials admit"],
        "certainty": ["100% proven", "undeniable proof", "everyone knows"],
    }
    lowered = text.casefold()
    flags: list[dict[str, str]] = []
    for category, phrases in patterns.items():
        for phrase in phrases:
            if phrase in lowered:
                flags.append({
                    "category": category,
                    "phrase": phrase,
                    "explanation": "This phrasing can be sensational or unsupported; verify the underlying claim independently.",
                })
    return flags


def _load_model(model_path: Path) -> Any | None:
    if not model_path.is_file():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def create_app(model_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    CORS(app, resources={r"/api/*": {"origins": os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")}})
    selected_path = Path(model_path) if model_path else Path(os.environ.get("FAKE_NEWS_MODEL", DEFAULT_MODEL))
    classifier = _load_model(selected_path)

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": classifier is not None,
            "model_path": str(selected_path),
            "warning": "Predictions are not verified facts.",
        })

    def analyze_text(text: str):
        clean_text = text.strip()
        if len(clean_text) < 20:
            return jsonify({"error": "Please provide at least 20 characters of article text."}), 400
        if len(clean_text) > 200_000:
            return jsonify({"error": "Text exceeds the 200,000-character limit."}), 413
        prediction = None
        scores = None
        if classifier is not None:
            try:
                prediction = str(classifier.predict([clean_text])[0])
                if hasattr(classifier, "predict_proba"):
                    raw_scores = classifier.predict_proba([clean_text])[0]
                    classes = getattr(classifier, "classes_", [])
                    if hasattr(classifier, "named_steps"):
                        final_estimator = list(classifier.named_steps.values())[-1]
                        classes = getattr(final_estimator, "classes_", classes)
                    scores = {str(label): float(score) for label, score in zip(classes, raw_scores)}
            except Exception:
                app.logger.exception("Model prediction failed")
                return jsonify({"error": "The trained model could not analyze this text."}), 500

        return jsonify({
            "prediction": prediction,
            "scores": scores,
            "model_available": classifier is not None,
            "suspicious_language": suspicious_language(clean_text),
            "text_length": len(clean_text),
            "caveat": (
                "This result detects patterns associated with dataset labels; it does not establish factual truth. "
                "Check independent, reliable sources before drawing conclusions."
            ),
        })

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object with a 'text' field."}), 400
        text = payload.get("text")
        if not isinstance(text, str):
            return jsonify({"error": "The 'text' field must be a string."}), 400
        result = analyze_text(text)
        return result

    @app.post("/api/analyze-pdf")
    def analyze_pdf():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Choose a PDF file to upload."}), 400
        if not uploaded.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF uploads are supported."}), 415
        try:
            extracted = extract_pdf_text(uploaded.read())
        except PdfTextError as exc:
            return jsonify({"error": str(exc)}), 400
        result = analyze_text(str(extracted["text"]))
        # Flask response bodies are JSON; add the extraction metadata if analysis succeeded.
        response, status = result if isinstance(result, tuple) else (result, 200)
        if status != 200:
            return response, status
        body = response.get_json()
        body["pdf"] = {
            "filename": Path(uploaded.filename).name,
            "page_count": extracted["page_count"],
            "character_count": extracted["character_count"],
        }
        return jsonify(body)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Upload too large. Maximum request size is 8 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
