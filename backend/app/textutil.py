"""Token, containment, and similarity helpers used by the deterministic evaluators."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Literal

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONSTRAINT = re.compile(r"\b[a-z0-9]*\d[a-z0-9]*\b", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
_WHITESPACE = re.compile(r"\s+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "on",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "how",
        "what",
        "when",
        "where",
        "who",
        "whom",
        "which",
        "that",
        "this",
        "these",
        "those",
        "please",
        "tell",
        "me",
        "about",
        "from",
        "at",
        "as",
        "into",
        "under",
        "our",
        "your",
        "their",
        "it",
        "its",
        "can",
        "you",
        "i",
        "we",
        "they",
        "must",
        "should",
        "will",
        "just",
        "than",
        "then",
        "also",
        "using",
        "only",
        "below",
        "above",
        "within",
        "without",
    }
)

_VAGUE_TERMS: frozenset[str] = frozenset(
    {
        "something",
        "stuff",
        "thing",
        "things",
        "info",
        "information",
        "details",
        "whatever",
        "anything",
    }
)


def clamp01(value: float) -> float:
    """Clamp a metric to the unit interval and round it for stable JSON."""
    if math.isnan(value) or value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(float(value), 4)


def content_tokens(text: str) -> list[str]:
    """Lowercase content tokens with stopwords removed."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


def constraint_tokens(text: str) -> set[str]:
    """Numbers, mixed alphanumeric ids, and acronyms that a rewrite must keep."""
    found = {match.lower() for match in _CONSTRAINT.findall(text)}
    found.update(match.lower() for match in _ACRONYM.findall(text))
    return found


def token_recall(hypothesis: str, reference: str) -> float:
    """Share of hypothesis content tokens that also appear in the reference."""
    hypothesis_tokens = set(content_tokens(hypothesis))
    if not hypothesis_tokens:
        return 0.0
    reference_tokens = set(content_tokens(reference))
    return len(hypothesis_tokens & reference_tokens) / len(hypothesis_tokens)


def sentences(text: str) -> list[str]:
    """Split text into sentences, falling back to the stripped text."""
    stripped = text.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(stripped) if part.strip()]
    return parts or [stripped]


def normalize_ws(text: str) -> str:
    """Casefold text and collapse whitespace so containment checks are stable."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def contains_text(haystack: str, needle: str) -> bool:
    """Return whether the full needle survives inside haystack after normalization."""
    normalized_needle = normalize_ws(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalize_ws(haystack)


def ambiguity_level(query: str) -> Literal["low", "medium", "high"]:
    """Classify a rewritten query as low, medium, or high ambiguity."""
    tokens = content_tokens(query)
    vague_count = sum(1 for token in tokens if token in _VAGUE_TERMS)
    if len(tokens) <= 2 or (len(tokens) <= 4 and vague_count >= 1):
        return "high"
    if vague_count >= 2 or (vague_count >= 1 and len(tokens) <= 6):
        return "medium"
    return "low"


def preservation_score(original: str, rewritten: str) -> int:
    """Map constraint recall onto the 1-5 intent-preservation scale."""
    original_tokens = set(content_tokens(original))
    rewritten_tokens = set(content_tokens(rewritten))
    if not original_tokens or not rewritten_tokens:
        return 1
    recall = len(original_tokens & rewritten_tokens) / len(original_tokens)
    missing_constraints = constraint_tokens(original) - constraint_tokens(rewritten)
    if missing_constraints:
        recall *= 0.5
    if recall >= 0.90:
        return 5
    if recall >= 0.75:
        return 4
    if recall >= 0.55:
        return 3
    if recall >= 0.35:
        return 2
    return 1


def lexical_cosine(left: str, right: str) -> float:
    """Bag-of-words cosine used as the deterministic stand-in for embeddings."""
    left_counts = Counter(content_tokens(left))
    right_counts = Counter(content_tokens(right))
    if not left_counts or not right_counts:
        return 0.0
    shared = set(left_counts) | set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp01(dot / (left_norm * right_norm))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Dense cosine similarity for embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return clamp01(dot / (left_norm * right_norm))


def preview(tokens: list[str] | set[str], limit: int = 8) -> str:
    """Render a short comma-separated token list for rationales."""
    ordered = sorted(tokens)
    if not ordered:
        return "none"
    head = ordered[:limit]
    text = ", ".join(head)
    remaining = len(ordered) - len(head)
    if remaining:
        text += f" (+{remaining} more)"
    return text
