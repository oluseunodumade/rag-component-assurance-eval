"""Retrieval overlap, embedding drift, and prompt-assembly checks."""

from __future__ import annotations

from dataclasses import dataclass

from app.criteria import (
    RETRIEVAL_F1_MIN,
    RETRIEVAL_PRECISION_MIN,
    RETRIEVAL_RECALL_MIN,
    retrieval_passed,
)
from app.llm import LLMClient
from app.schemas import AuditItem, RetrievalResult
from app.textutil import clamp01, contains_text, cosine_similarity, lexical_cosine


@dataclass(frozen=True)
class RetrievalMeasurement:
    """Overlap metrics and embedding similarity before the prompt check is applied."""

    context_precision: float
    context_recall: float
    f1_score: float
    embedding_cosine_similarity: float
    rationale: str


@dataclass(frozen=True)
class PromptAssemblyAssessment:
    """Deterministic check that the final prompt still holds the query and chunks."""

    prompt_assembly_valid: bool
    missing_query: bool
    truncated_chunk_ids: tuple[str, ...]
    rationale: str


class RetrievalQualityEvaluator:
    """Precision, recall, F1, and cosine similarity between retrieved and gold context."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def evaluate(self, item: AuditItem, *, mock_mode: bool) -> RetrievalMeasurement:
        retrieved_count = len(item.retrieved_chunks)
        relevant_count = sum(1 for chunk in item.retrieved_chunks if chunk.relevant)
        precision = relevant_count / retrieved_count if retrieved_count else 0.0

        gold_ids = {chunk.chunk_id for chunk in item.gold_chunks}
        retrieved_ids = {chunk.chunk_id for chunk in item.retrieved_chunks}
        hits = len(gold_ids & retrieved_ids)
        recall = hits / len(gold_ids) if gold_ids else 1.0
        f1 = _f1(precision, recall)

        retrieved_text = "\n".join(chunk.text for chunk in item.retrieved_chunks)
        gold_text = "\n".join(chunk.text for chunk in item.gold_chunks)
        similarity, method = self._similarity(retrieved_text, gold_text, mock_mode=mock_mode)
        drift = clamp01(1.0 - similarity)
        rationale = (
            f"Context precision is {precision:.2f} ({relevant_count}/{retrieved_count} retrieved chunks relevant). "
            f"Context recall is {recall:.2f} ({hits}/{len(gold_ids)} gold chunks retrieved). "
            f"F1 is {f1:.2f}. "
            f"Cosine similarity between retrieved context and gold context is {similarity:.2f} "
            f"(drift {drift:.2f}) using {method}."
        )
        return RetrievalMeasurement(
            context_precision=clamp01(precision),
            context_recall=clamp01(recall),
            f1_score=clamp01(f1),
            embedding_cosine_similarity=similarity,
            rationale=rationale,
        )

    def _similarity(self, retrieved_text: str, gold_text: str, *, mock_mode: bool) -> tuple[float, str]:
        if mock_mode or not retrieved_text.strip() or not gold_text.strip():
            return lexical_cosine(retrieved_text, gold_text), "a deterministic lexical cosine"
        client = self._client or LLMClient()
        self._client = client
        vectors = client.embed([retrieved_text, gold_text])
        if len(vectors) != 2:
            return 0.0, client.embedding_model
        return cosine_similarity(vectors[0], vectors[1]), client.embedding_model


class PromptAugmentationEvaluator:
    """Verify the assembled prompt contains the user query and each chunk in full."""

    def evaluate(self, item: AuditItem) -> PromptAssemblyAssessment:
        missing_query = not contains_text(item.assembled_prompt, item.user_query)
        truncated = tuple(
            chunk.chunk_id
            for chunk in item.retrieved_chunks
            if not contains_text(item.assembled_prompt, chunk.text)
        )
        valid = not missing_query and not truncated
        problems: list[str] = []
        if missing_query:
            problems.append("the user query is absent from the assembled prompt")
        if truncated:
            problems.append("truncated or missing chunks: " + ", ".join(truncated))
        if valid:
            detail = "Assembled prompt contains the full user query and every retrieved chunk without truncation."
        else:
            detail = "Prompt assembly failed: " + "; ".join(problems) + "."
        return PromptAssemblyAssessment(
            prompt_assembly_valid=valid,
            missing_query=missing_query,
            truncated_chunk_ids=truncated,
            rationale=detail,
        )


def run_retrieval_stage(
    item: AuditItem,
    *,
    mock_mode: bool,
    client: LLMClient | None = None,
    quality: RetrievalQualityEvaluator | None = None,
    prompt_evaluator: PromptAugmentationEvaluator | None = None,
) -> RetrievalResult:
    """Run retrieval metrics and the prompt check, then apply the shared pass gate."""
    measurement = (quality or RetrievalQualityEvaluator(client)).evaluate(item, mock_mode=mock_mode)
    assembly = (prompt_evaluator or PromptAugmentationEvaluator()).evaluate(item)
    passed = retrieval_passed(
        measurement.context_precision,
        measurement.context_recall,
        measurement.f1_score,
        assembly.prompt_assembly_valid,
    )
    gate = (
        "Pass gate met."
        if passed
        else (
            "Pass gate not met (requires precision, recall, and F1 of at least "
            f"{RETRIEVAL_PRECISION_MIN:.2f} / {RETRIEVAL_RECALL_MIN:.2f} / {RETRIEVAL_F1_MIN:.2f} "
            "and a valid prompt assembly)."
        )
    )
    return RetrievalResult(
        context_precision=measurement.context_precision,
        context_recall=measurement.context_recall,
        f1_score=measurement.f1_score,
        embedding_cosine_similarity=measurement.embedding_cosine_similarity,
        prompt_assembly_valid=assembly.prompt_assembly_valid,
        passed=passed,
        rationale=f"{measurement.rationale} {assembly.rationale} {gate}",
    )


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
