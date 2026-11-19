"""Pass gates shared by the component evaluators and the batch report."""

from __future__ import annotations

PRESERVATION_PASS_MIN = 4
ROUTING_CONFIDENCE_MIN = 0.70
RETRIEVAL_PRECISION_MIN = 0.50
RETRIEVAL_RECALL_MIN = 0.50
RETRIEVAL_F1_MIN = 0.50
FAITHFULNESS_MIN = 0.70
ANSWER_RELEVANCY_MIN = 0.50
ALIGNMENT_TIER_MIN = 2
UNGROUNDED_EXTRA_TIERS = frozenset({0, 1})

PRESERVATION_LABELS: dict[int, str] = {
    1: "intent lost",
    2: "major constraints dropped",
    3: "partial preservation",
    4: "minor wording changes",
    5: "intent fully preserved",
}


def modification_passed(preservation_score: int, ambiguity_level: str) -> bool:
    """A rewrite passes when intent is kept and the rewrite is not highly ambiguous."""
    return preservation_score >= PRESERVATION_PASS_MIN and ambiguity_level != "high"


def routing_passed(routes_match: bool, routing_confidence: float) -> bool:
    """A route passes when the index matches and confidence clears the gate."""
    return routes_match and routing_confidence >= ROUTING_CONFIDENCE_MIN


def retrieval_passed(
    context_precision: float,
    context_recall: float,
    f1_score: float,
    prompt_assembly_valid: bool,
) -> bool:
    """Retrieval passes when overlap metrics clear the gate and the prompt is intact."""
    return (
        context_precision >= RETRIEVAL_PRECISION_MIN
        and context_recall >= RETRIEVAL_RECALL_MIN
        and f1_score >= RETRIEVAL_F1_MIN
        and prompt_assembly_valid
    )


def response_passed(faithfulness: float, relevancy: float, alignment_tier: int) -> bool:
    """Generation passes when the answer is grounded, relevant, and free of extras."""
    return (
        faithfulness >= FAITHFULNESS_MIN
        and relevancy >= ANSWER_RELEVANCY_MIN
        and alignment_tier >= ALIGNMENT_TIER_MIN
    )
