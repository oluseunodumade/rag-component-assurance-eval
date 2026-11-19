"""Evaluators for query rewriting and index routing."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.criteria import (
    PRESERVATION_LABELS,
    PRESERVATION_PASS_MIN,
    ROUTING_CONFIDENCE_MIN,
    modification_passed,
    routing_passed,
)
from app.llm import LLMClient
from app.schemas import AuditItem, QueryModificationResult, QueryRoutingResult
from app.textutil import ambiguity_level, constraint_tokens, content_tokens, preservation_score, preview

logger = logging.getLogger(__name__)

_MODIFICATION_SYSTEM = """You are an auditor for the query-rewriting stage of a retrieval-augmented generation system.
Compare the original user query with the rewritten query.
Score intent preservation as an integer from 1 to 5:
1 = intent lost, 2 = major constraints dropped, 3 = partial preservation, 4 = minor wording changes, 5 = intent fully preserved.
Set ambiguity_level of the REWRITTEN query to exactly one of: low, medium, high.
High means the rewrite is vague, underspecified, or missing the entity the user asked about.
Return a JSON object with keys preservation_score, ambiguity_level, and rationale.
The rationale must cite which constraints were kept or dropped."""


class _ModificationJudgement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    preservation_score: int = Field(ge=1, le=5)
    ambiguity_level: Literal["low", "medium", "high"]
    rationale: str = Field(min_length=1)

    @field_validator("preservation_score", mode="before")
    @classmethod
    def _round_score(cls, value: object) -> int:
        return int(round(float(value)))  # type: ignore[arg-type]

    @field_validator("ambiguity_level", mode="before")
    @classmethod
    def _normalize_level(cls, value: object) -> str:
        text = str(value).strip().lower()
        aliases = {"med": "medium", "moderate": "medium", "hi": "high", "lo": "low"}
        text = aliases.get(text, text)
        if text not in {"low", "medium", "high"}:
            raise ValueError("ambiguity_level must be low, medium, or high")
        return text


class QueryModificationEvaluator:
    """Score whether a rewrite preserves intent and stays specific."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def evaluate(self, item: AuditItem, *, mock_mode: bool) -> QueryModificationResult:
        if mock_mode:
            return self._heuristic(item)
        judgement = self._client_or_new().complete_structured(
            system=_MODIFICATION_SYSTEM,
            user=_modification_user_prompt(item),
            schema=_ModificationJudgement,
        )
        passed = modification_passed(judgement.preservation_score, judgement.ambiguity_level)
        gate = _modification_gate_sentence(passed)
        return QueryModificationResult(
            preservation_score=judgement.preservation_score,
            ambiguity_level=judgement.ambiguity_level,  # validated as low, medium, or high
            passed=passed,
            rationale=f"{judgement.rationale.strip()} {gate}",
        )

    def _heuristic(self, item: AuditItem) -> QueryModificationResult:
        score = preservation_score(item.user_query, item.rewritten_query)
        level = ambiguity_level(item.rewritten_query)
        passed = modification_passed(score, level)
        kept = set(content_tokens(item.user_query)) & set(content_tokens(item.rewritten_query))
        dropped = set(content_tokens(item.user_query)) - set(content_tokens(item.rewritten_query))
        missing_constraints = constraint_tokens(item.user_query) - constraint_tokens(item.rewritten_query)
        rationale = (
            f"Preservation score {score}/5 ({PRESERVATION_LABELS[score]}). "
            f"Ambiguity of the rewrite is {level}. "
            f"Kept terms: {preview(kept)}. Dropped terms: {preview(dropped)}. "
            f"Missing constraints: {preview(missing_constraints)}. "
            f"{_modification_gate_sentence(passed)}"
        )
        return QueryModificationResult(
            preservation_score=score,
            ambiguity_level=level,
            passed=passed,
            rationale=rationale,
        )

    def _client_or_new(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client


class QueryRoutingEvaluator:
    """Compare the expected route with the route the system actually took.

    The match itself is deterministic in live and mock mode. An LLM is not asked
    to overturn a logged route, because routing accuracy has to stay reproducible.
    """

    def evaluate(self, item: AuditItem, *, mock_mode: bool) -> QueryRoutingResult:
        logger.debug(
            "Routing check query_id=%s mock_mode=%s",
            item.query_id,
            mock_mode,
        )
        expected = item.expected_route.strip()
        actual = item.actual_route.strip()
        routes_match = expected.casefold() == actual.casefold()
        confidence = item.routing_confidence
        passed = routing_passed(routes_match, confidence)
        if routes_match and passed:
            detail = (
                f"Actual route {actual} matches expected route {expected} "
                f"with confidence {confidence:.2f}."
            )
        elif routes_match:
            detail = (
                f"Actual route {actual} matches expected route {expected}, but confidence "
                f"{confidence:.2f} is below {ROUTING_CONFIDENCE_MIN:.2f}."
            )
        else:
            detail = (
                f"Actual route {actual} does not match expected route {expected} "
                f"(confidence {confidence:.2f})."
            )
        gate = (
            "Pass gate met."
            if passed
            else (
                "Pass gate not met (requires an exact route match and confidence "
                f">= {ROUTING_CONFIDENCE_MIN:.2f})."
            )
        )
        return QueryRoutingResult(
            expected_route=item.expected_route,
            actual_route=item.actual_route,
            routing_confidence=confidence,
            passed=passed,
            rationale=f"{detail} {gate}",
        )


def _modification_user_prompt(item: AuditItem) -> str:
    return (
        f"Original user query:\n{item.user_query}\n\n"
        f"Rewritten query:\n{item.rewritten_query}\n"
    )


def _modification_gate_sentence(passed: bool) -> str:
    if passed:
        return "Pass gate met."
    return (
        "Pass gate not met (requires preservation "
        f">= {PRESERVATION_PASS_MIN} and ambiguity other than high)."
    )
