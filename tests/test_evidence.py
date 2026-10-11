from fake_news.evidence import EvidenceItem, summarize_evidence


def test_summary_counts_review_annotations_without_verdict() -> None:
    items = [
        EvidenceItem("Report A", "Source A", "https://example.com/a", True),
        EvidenceItem("Report B", "Source B", "https://example.com/b", False),
        EvidenceItem("Report C", "Source C", "https://example.com/c", None),
    ]
    summary = summarize_evidence(items)
    assert summary["evidence_count"] == 3
    assert summary["support_count"] == 1
    assert summary["contradict_count"] == 1
    assert summary["unassessed_count"] == 1
    assert summary["status"] == "evidence_requires_human_review"


def test_empty_evidence_stays_unverified() -> None:
    summary = summarize_evidence([])
    assert summary["evidence_count"] == 0
    assert summary["status"] == "evidence_requires_human_review"
