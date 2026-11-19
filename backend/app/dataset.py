"""Load the public GAAP / SEC sample audit dataset."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from app.schemas import AuditCategory, AuditItem, ContextChunk, RetrievedChunk
from app.textutil import normalize_ws

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_eval_dataset.json"
_DEFAULT_ROUTING_CONFIDENCE = 0.91

_SCENARIOS: dict[str, str] = {
    "Q-101": "Year-end life reserve memo: unpaid ASC 944 claim benefits",
    "Q-102": "Three-year SaaS revenue memo: constrain usage-based fees",
    "Q-103": "Integrated audit: PCAOB sample sizes for revenue cutoff",
    "Q-104": "Warehouse lease amendment: discount-rate reassessment",
    "Q-105": "Form 10-K earnings release: one-time severance add-backs",
    "Q-106": "Annual goodwill file: qualitative assessment before the quantitative test",
    "Q-107": "CECL allowance memo: forecasts and reversion for held-to-maturity loans",
    "Q-108": "Bank capital memo: Basel III operational-risk threshold",
}


class SampleEvalRecord(BaseModel):
    """One public financial-reporting case as stored in the JSON dataset."""

    query_id: str = Field(min_length=1)
    category: AuditCategory
    original_query: str = Field(min_length=1)
    modified_query: str = Field(min_length=1)
    expected_route: str = Field(min_length=1)
    actual_route: str = Field(min_length=1)
    retrieved_chunks: list[str] = Field(min_length=2, max_length=3)
    reference_chunks: list[str] = Field(min_length=1)
    assembled_prompt: str = Field(min_length=1)
    ground_truth_answer: str = Field(min_length=1)
    model_generated_answer: str


def load_dataset() -> list[AuditItem]:
    """Read the JSON audit log and map it onto the evaluation schema."""
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    records = TypeAdapter(list[SampleEvalRecord]).validate_python(payload)
    items = [_to_audit_item(record) for record in records]
    query_ids = [item.query_id for item in items]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Audit dataset contains duplicate query_id values")
    return items


def _to_audit_item(record: SampleEvalRecord) -> AuditItem:
    gold = [
        ContextChunk(
            chunk_id=f"{record.query_id}-ref-{index}",
            text=text,
            source=f"{record.category} reference {index}",
        )
        for index, text in enumerate(record.reference_chunks, start=1)
    ]
    gold_by_text = {normalize_ws(chunk.text): chunk for chunk in gold}
    retrieved: list[RetrievedChunk] = []
    for index, text in enumerate(record.retrieved_chunks, start=1):
        match = gold_by_text.get(normalize_ws(text))
        if match is None:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=f"{record.query_id}-ret-{index}",
                    text=text,
                    source=f"{record.category} retrieved {index}",
                    score=0.62,
                    relevant=False,
                )
            )
            continue
        retrieved.append(
            RetrievedChunk(
                chunk_id=match.chunk_id,
                text=text,
                source=match.source,
                score=0.9,
                relevant=True,
            )
        )
    return AuditItem(
        query_id=record.query_id,
        category=record.category,
        scenario=_SCENARIOS.get(record.query_id, record.category),
        user_query=record.original_query,
        rewritten_query=record.modified_query,
        expected_route=record.expected_route,
        actual_route=record.actual_route,
        routing_confidence=_DEFAULT_ROUTING_CONFIDENCE,
        retrieved_chunks=retrieved,
        gold_chunks=gold,
        assembled_prompt=record.assembled_prompt,
        generated_response=record.model_generated_answer,
        ground_truth_answer=record.ground_truth_answer,
    )
