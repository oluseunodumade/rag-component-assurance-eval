"""Faithfulness, relevancy, and ground-truth alignment for generated answers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.criteria import (
    ALIGNMENT_TIER_MIN,
    ANSWER_RELEVANCY_MIN,
    FAITHFULNESS_MIN,
    response_passed,
)
from app.llm import LLMClient
from app.schemas import ALIGNMENT_TIER_LABELS, AuditItem, GroundingSpan, ResponseGenerationResult
from app.textutil import clamp01, sentences, token_recall

_SUPPORT_THRESHOLD = 0.55
_EXTRA_THRESHOLD = 0.45

_RESPONSE_SYSTEM = """You are an LLM-as-a-Judge auditing the answer stage of a retrieval-augmented generation system.
Score only the generated answer. Use the retrieved context for faithfulness and the ground-truth answer for alignment.
Return a JSON object with keys faithfulness_score, answer_relevancy_score, ground_truth_alignment_tier, and judge_rationale.
faithfulness_score and answer_relevancy_score are floats from 0 to 1.
faithfulness_score is the share of the answer that is supported by the retrieved context alone.
answer_relevancy_score is how completely the answer addresses the user query.
ground_truth_alignment_tier is an integer:
3 = Complete Alignment
2 = Partial Alignment
1 = Complete Alignment + Ungrounded Extras
0 = Partial Alignment + Hallucinated Extras
Use tier 1 when the reference answer is fully covered but the response adds claims that are not in the ground truth.
Use tier 0 when coverage is only partial and the response also adds unsupported claims.
judge_rationale must name the supported claims and any extra claims."""


class _ResponseJudgement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    answer_relevancy_score: float = Field(ge=0.0, le=1.0)
    ground_truth_alignment_tier: int = Field(ge=0, le=3)
    judge_rationale: str = Field(min_length=1)

    @field_validator("faithfulness_score", "answer_relevancy_score", mode="before")
    @classmethod
    def _clamp_score(cls, value: object) -> float:
        return clamp01(float(value))  # type: ignore[arg-type]

    @field_validator("ground_truth_alignment_tier", mode="before")
    @classmethod
    def _round_tier(cls, value: object) -> int:
        return int(round(float(value)))  # type: ignore[arg-type]


class ResponseQualityJudge:
    """Judge a generated answer with a live model or a deterministic heuristic."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    def evaluate(self, item: AuditItem, *, mock_mode: bool) -> ResponseGenerationResult:
        if mock_mode:
            return self._heuristic(item)
        judgement = self._client_or_new().complete_structured(
            system=_RESPONSE_SYSTEM,
            user=_response_user_prompt(item),
            schema=_ResponseJudgement,
        )
        passed = response_passed(
            judgement.faithfulness_score,
            judgement.answer_relevancy_score,
            judgement.ground_truth_alignment_tier,
        )
        label = ALIGNMENT_TIER_LABELS[judgement.ground_truth_alignment_tier]
        return ResponseGenerationResult(
            faithfulness_score=clamp01(judgement.faithfulness_score),
            answer_relevancy_score=clamp01(judgement.answer_relevancy_score),
            ground_truth_alignment_tier=judgement.ground_truth_alignment_tier,
            passed=passed,
            judge_rationale=(
                f"{judgement.judge_rationale.strip()} "
                f"Alignment tier {judgement.ground_truth_alignment_tier} ({label}). "
                f"{_response_gate_sentence(passed)}"
            ),
            grounding_spans=grounding_spans(item),
        )

    def _heuristic(self, item: AuditItem) -> ResponseGenerationResult:
        retrieved_text = "\n".join(chunk.text for chunk in item.retrieved_chunks)
        gold_text = "\n".join(chunk.text for chunk in item.gold_chunks)
        support_text = f"{gold_text}\n{item.ground_truth_answer}"
        response_sentences = sentences(item.generated_response)
        reference_sentences = sentences(item.ground_truth_answer)

        if response_sentences:
            faithfulness = sum(token_recall(sentence, retrieved_text) for sentence in response_sentences)
            faithfulness /= len(response_sentences)
        else:
            faithfulness = 0.0
        relevancy = token_recall(item.user_query, item.generated_response)

        covered = sum(
            1 for sentence in reference_sentences if token_recall(sentence, item.generated_response) >= _SUPPORT_THRESHOLD
        )
        unsupported = [
            sentence
            for sentence in response_sentences
            if token_recall(sentence, support_text) < _EXTRA_THRESHOLD
        ]
        extras = len(unsupported)
        coverage = covered / len(reference_sentences) if reference_sentences else 0.0
        extra_rate = extras / len(response_sentences) if response_sentences else 1.0
        tier = _alignment_tier(coverage, extra_rate)
        faithfulness_score = clamp01(faithfulness)
        relevancy_score = clamp01(relevancy)
        passed = response_passed(faithfulness_score, relevancy_score, tier)
        label = ALIGNMENT_TIER_LABELS[tier]
        extra_note = ""
        if unsupported:
            quoted = " | ".join(sentence.rstrip(".") for sentence in unsupported)
            extra_note = f" Ungrounded extra claims not in the reference answer or gold context: {quoted}."
        rationale = (
            f"Faithfulness {faithfulness_score:.2f} against retrieved context. "
            f"Answer relevancy {relevancy_score:.2f} against the user query. "
            f"Ground-truth alignment tier {tier} ({label}): "
            f"{covered}/{len(reference_sentences)} reference claims covered and "
            f"{extras}/{len(response_sentences)} response sentences unsupported."
            f"{extra_note} "
            f"{_response_gate_sentence(passed)}"
        )
        return ResponseGenerationResult(
            faithfulness_score=faithfulness_score,
            answer_relevancy_score=relevancy_score,
            ground_truth_alignment_tier=tier,
            passed=passed,
            judge_rationale=rationale,
            grounding_spans=grounding_spans(item),
        )

    def _client_or_new(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient()
        return self._client


def grounding_spans(item: AuditItem) -> list[GroundingSpan]:
    """Mark each answer sentence as grounded in the retrieved context or not."""
    retrieved_text = "\n".join(chunk.text for chunk in item.retrieved_chunks)
    support_text = "\n".join(chunk.text for chunk in item.gold_chunks) + "\n" + item.ground_truth_answer
    spans: list[GroundingSpan] = []
    for sentence in sentences(item.generated_response):
        if token_recall(sentence, retrieved_text) >= _SUPPORT_THRESHOLD:
            spans.append(
                GroundingSpan(
                    text=sentence,
                    label="grounded",
                    reason="This sentence is supported by the retrieved context.",
                )
            )
            continue
        if token_recall(sentence, support_text) < _EXTRA_THRESHOLD:
            spans.append(
                GroundingSpan(
                    text=sentence,
                    label="ungrounded",
                    reason="This sentence adds a claim that is not in the retrieved context or the reference answer.",
                )
            )
            continue
        spans.append(
            GroundingSpan(
                text=sentence,
                label="ungrounded",
                reason="This sentence is only partly supported by the retrieved context.",
            )
        )
    return spans


def _alignment_tier(coverage: float, extra_rate: float) -> int:
    has_extras = extra_rate > 0.0
    if coverage >= 0.80 and not has_extras:
        return 3
    if coverage >= 0.40 and not has_extras:
        return 2
    if coverage >= 0.80 and has_extras:
        return 1
    return 0


def _response_user_prompt(item: AuditItem) -> str:
    context = "\n\n".join(
        f"[{chunk.chunk_id}] {chunk.text}" for chunk in item.retrieved_chunks
    ) or "(no retrieved context)"
    return (
        f"User query:\n{item.user_query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Generated answer:\n{item.generated_response or '(empty)'}\n\n"
        f"Ground-truth answer:\n{item.ground_truth_answer}\n"
    )


def _response_gate_sentence(passed: bool) -> str:
    if passed:
        return "Pass gate met."
    return (
        "Pass gate not met (requires faithfulness "
        f">= {FAITHFULNESS_MIN:.2f}, answer relevancy >= {ANSWER_RELEVANCY_MIN:.2f}, "
        f"and alignment tier >= {ALIGNMENT_TIER_MIN})."
    )
