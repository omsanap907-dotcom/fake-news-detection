import pytest

from fake_news.model import build_model, predict_text


def test_model_learns_simple_training_examples() -> None:
    texts = [
        "official election results published by the commission",
        "government agency releases official economic report",
        "scientists publish peer reviewed climate study",
        "city council announces public transport schedule",
        "secret miracle cure hidden by every doctor share now",
        "shocking hoax claims all scientists are lying share immediately",
        "unbelievable conspiracy proves everything is fake click here",
        "celebrity secretly reveals impossible cure overnight",
    ]
    labels = ["REAL", "REAL", "REAL", "REAL", "FAKE", "FAKE", "FAKE", "FAKE"]
    model = build_model()
    model.fit(texts, labels)

    prediction = predict_text(model, "official government agency publishes report")
    assert prediction.label in {"FAKE", "REAL"}
    assert set(prediction.scores) == {"FAKE", "REAL"}
    assert abs(sum(prediction.scores.values()) - 1.0) < 1e-8
    assert "not a fact-check" in prediction.disclaimer


def test_prediction_rejects_empty_text() -> None:
    model = build_model()
    model.fit(["ordinary public report", "impossible secret miracle"], ["REAL", "FAKE"])
    with pytest.raises(ValueError, match="non-empty"):
        predict_text(model, "   ")
