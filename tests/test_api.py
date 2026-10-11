from io import BytesIO

from fake_news.api import create_app, suspicious_language


def test_health_without_model(tmp_path):
    client = create_app(tmp_path / "missing.joblib").test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is False


def test_requires_json_text():
    client = create_app("missing-model.joblib").test_client()
    response = client.post("/api/analyze", json={"text": "short"})
    assert response.status_code == 400


def test_analyze_without_model_still_returns_language_flags():
    client = create_app("missing-model.joblib").test_client()
    response = client.post("/api/analyze", json={
        "text": "Shocking truth! Share immediately before it gets deleted. This claim needs checking."
    })
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_available"] is False
    assert body["prediction"] is None
    assert len(body["suspicious_language"]) >= 2
    assert "does not establish factual truth" in body["caveat"]


def test_rejects_wrong_pdf_extension():
    client = create_app("missing-model.joblib").test_client()
    response = client.post(
        "/api/analyze-pdf",
        data={"file": (BytesIO(b"not a pdf"), "article.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415


def test_suspicious_language_is_style_signal():
    flags = suspicious_language("Scientists confirm a shocking truth. Wake up!")
    assert {flag["category"] for flag in flags} >= {"sensationalism", "unverified_authority"}
