"""Small helpers for a future evidence-aware verification workflow.

These helpers do not decide whether a claim is true. They preserve the distinction
between a model's text-label prediction and external evidence about the claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvidenceItem:
    title: str
    source_name: str
    url: str
    supports_claim: bool | None = None
    note: str = ""


def summarize_evidence(items: Iterable[EvidenceItem]) -> dict[str, object]:
    """Summarize supplied evidence annotations without deciding truth.

    `supports_claim=True` means the reviewer marked this item as supportive;
    `False` means contradictory; `None` means not assessed. This is a bookkeeping
    function only. It does not fetch, validate, rank, or independently verify URLs.
    """
    materialized = list(items)
    assessed = [item for item in materialized if item.supports_claim is not None]
    support_count = sum(item.supports_claim is True for item in assessed)
    contradict_count = sum(item.supports_claim is False for item in assessed)
    return {
        "evidence_count": len(materialized),
        "assessed_count": len(assessed),
        "support_count": support_count,
        "contradict_count": contradict_count,
        "unassessed_count": len(materialized) - len(assessed),
        "status": "evidence_requires_human_review",
        "caveat": (
            "Counts summarize reviewer annotations only. Source quality, context, "
            "independence, and claim scope must be assessed separately."
        ),
    }
