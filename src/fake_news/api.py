"""Local Flask API for the educational fake-news text classifier."""
from __future__ import annotations
import os
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
from fake_news.pdf_text import PdfTextError, extract_pdf_text

def suspicious_language(text: str) -> list[dict[str, str]]:
    """Surface selected rhetorical patterns; these do not determine truth."""
    patterns = {
        "urgency": ["share immediately", "before it gets deleted", "act now"],
        "sensationalism": ["shocking truth", "wake up", "you won't believe", "secret they don't want you to know"],
        "unverified_authority": ["scientists confirm", "experts say", "officials admit"],
        "certainty": ["100% proven", "undeniable proof", "everyone knows"],
    }
    lowered, flags = text.casefold(), []
    for category, phrases in patterns.items():
        for phrase in phrases:
            if phrase in lowered:
                flags.append({"category": category, "phrase": phrase,
                              "explanation": "This wording can be sensational or unsupported; verify the underlying claim independently."})
    return flags

def create_app(model_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    CORS(app, resources={r"/api/*": {"origins": os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")}})
    selected_path = Path(model_path) if model_path else Path(os.environ.get("FAKE_NEWS_MODEL", "artifacts/fake_news_model.joblib"))
    try:
        classifier = joblib.load(selected_path) if selected_path.is_file() else None
    except Exception:
        app.logger.exception("Unable to load model artifact")
        classifier = None

    def analyze_text(text: str):
        cleaned = text.strip()
        if len(cleaned) < 20:
            return jsonify({"error": "Please provide at least 20 characters of article text."}), 400
        if len(cleaned) > 200_000:
            return jsonify({"error": "Text exceeds the 200,000-character limit."}), 413
        prediction, scores = None, None
        if classifier is not None:
            try:
                prediction = str(classifier.predict([cleaned])[0])
                if hasattr(classifier, "predict_proba"):
                    probabilities = classifier.predict_proba([cleaned])[0]
                    estimator = list(classifier.named_steps.values())[-1] if hasattr(classifier, "named_steps") else classifier
                    classes = getattr(estimator, "classes_", [])
                    scores = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
            except Exception:
                app.logger.exception("Prediction failed")
                return jsonify({"error": "The trained model could not analyze this text."}), 500
        return jsonify({
            "prediction": prediction, "scores": scores, "model_available": classifier is not None,
            "suspicious_language": suspicious_language(cleaned), "text_length": len(cleaned),
            "caveat": "A model prediction and writing-style flags do not establish factual truth. Verify claims against independent, reliable sources."
        })

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": classifier is not None,
                        "warning": "Predictions are not verified facts."})

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return jsonify({"error": "Send a JSON object containing a string 'text' field."}), 400
        return analyze_text(payload["text"])

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
        response, status = result if isinstance(result, tuple) else (result, 200)
        if status != 200:
            return response, status
        body = response.get_json()
        body["pdf"] = {"filename": Path(uploaded.filename).name,
                       "page_count": extracted["page_count"],
                       "character_count": extracted["character_count"]}
        return jsonify(body)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "Upload too large; the limit is 8 MB."}), 413
    return app

app = create_app()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
