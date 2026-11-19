"""Ad-hoc playground items and the v1/v2 benchmark strategies."""

from __future__ import annotations

from app.llm import LLMClient
from app.schemas import AuditItem, ContextChunk, PromptVersion, RetrievedChunk
from app.textutil import lexical_cosine

_EXTRA = (
    "Management privately forecast that adjusted EBITDA will double next year based on an "
    "unreleased board memo."
)
_V1_SINGLE_CHUNK = frozenset({"Q-101", "Q-106"})
_V1_MISROUTE = frozenset({"Q-108"})
_V1_EXTRA = frozenset({"Q-107"})


def render_prompt(query: str, chunks: list[RetrievedChunk], version: PromptVersion) -> str:
    """Build a v1 or v2 augmented prompt that still contains the query and each kept chunk."""
    header = (
        "Answer the question using the notes below."
        if version == "v1"
        else (
            "You are a careful assistant. Answer the user using only the context below. "
            "Do not add facts that are not in the context."
        )
    )
    body = "\n\n".join(chunk.text for chunk in chunks)
    return f"{header}\n\nUser query:\n{query}\n\nContext:\n{body}\n\nQuestion:\n{query}"


def apply_baseline(item: AuditItem) -> AuditItem:
    """v1 benchmark: thinner retrieval, one misroute, and one added ungrounded claim."""
    chunks = list(item.retrieved_chunks)
    if item.query_id in _V1_SINGLE_CHUNK:
        chunks = chunks[:1]
    answer = item.generated_response
    if item.query_id in _V1_EXTRA and _EXTRA not in answer:
        answer = f"{answer} {_EXTRA}"
    actual_route = item.actual_route
    if item.query_id in _V1_MISROUTE:
        actual_route = _opposite_route(item.expected_route)
    return item.model_copy(
        update={
            "retrieved_chunks": chunks,
            "assembled_prompt": render_prompt(item.user_query, chunks, "v1"),
            "generated_response": answer,
            "actual_route": actual_route,
        }
    )


def apply_runtime_controls(item: AuditItem, *, top_k: int, prompt_version: PromptVersion) -> AuditItem:
    """Apply the playground Top-K and prompt version to one logged query."""
    if prompt_version == "v2" and top_k >= len(item.retrieved_chunks):
        return item
    chunks = list(item.retrieved_chunks)[:top_k] or list(item.retrieved_chunks)[:1]
    answer = item.generated_response
    actual_route = item.actual_route
    if prompt_version == "v1":
        if _EXTRA not in answer:
            answer = f"{answer} {_EXTRA}"
        if actual_route == item.expected_route:
            actual_route = _opposite_route(item.expected_route)
    return item.model_copy(
        update={
            "retrieved_chunks": chunks,
            "assembled_prompt": render_prompt(item.user_query, chunks, prompt_version),
            "generated_response": answer,
            "actual_route": actual_route,
        }
    )


def build_custom_item(
    query: str,
    dataset: list[AuditItem],
    *,
    top_k: int,
    prompt_version: PromptVersion,
    mock_mode: bool,
) -> AuditItem:
    """Retrieve over the benchmark corpus and assemble one ad-hoc audit item."""
    cleaned = query.strip()
    ranked = _rank_passages(cleaned, dataset)
    reference = ranked[:3]
    if not reference:
        raise ValueError("The benchmark corpus has no passages to retrieve.")
    kept = reference[: max(1, min(top_k, len(reference)))]
    parent = kept[0][1]
    gold = [
        ContextChunk(chunk_id=f"adhoc-ref-{index}", text=chunk.text, source=chunk.source)
        for index, (chunk, _owner) in enumerate(reference, start=1)
    ]
    gold_by_text = {chunk.text: chunk for chunk in gold}
    retrieved: list[RetrievedChunk] = []
    for chunk, _owner in kept:
        match = gold_by_text[chunk.text]
        retrieved.append(
            RetrievedChunk(
                chunk_id=match.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=round(lexical_cosine(cleaned, chunk.text), 4),
                relevant=True,
            )
        )
    ground_truth = gold[0].text
    if mock_mode:
        rewritten = cleaned
        answer = ground_truth if prompt_version == "v2" else f"{ground_truth} {_EXTRA}"
    else:
        rewritten, answer = _live_rewrite_and_answer(cleaned, retrieved, prompt_version)
    expected_route = parent.expected_route
    actual_route = expected_route if prompt_version == "v2" else _opposite_route(expected_route)
    return AuditItem(
        query_id="ADHOC",
        category=parent.category,
        scenario="Ad-hoc playground query",
        user_query=cleaned,
        rewritten_query=rewritten,
        expected_route=expected_route,
        actual_route=actual_route,
        routing_confidence=0.91 if prompt_version == "v2" else 0.64,
        retrieved_chunks=retrieved,
        gold_chunks=gold,
        assembled_prompt=render_prompt(cleaned, retrieved, prompt_version),
        generated_response=answer,
        ground_truth_answer=ground_truth,
    )


def _rank_passages(query: str, dataset: list[AuditItem]) -> list[tuple[ContextChunk, AuditItem]]:
    seen: set[str] = set()
    ranked: list[tuple[float, ContextChunk, AuditItem]] = []
    for item in dataset:
        for chunk in item.gold_chunks:
            if chunk.text in seen:
                continue
            seen.add(chunk.text)
            ranked.append((lexical_cosine(query, chunk.text), chunk, item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [(chunk, owner) for _score, chunk, owner in ranked]


def _opposite_route(route: str) -> str:
    if route == "Audit_Assurance":
        return "Financial_Reporting"
    return "Audit_Assurance"


def _live_rewrite_and_answer(
    query: str,
    chunks: list[RetrievedChunk],
    version: PromptVersion,
) -> tuple[str, str]:
    client = LLMClient()
    rewrite = client.complete_text(
        system=(
            "Rewrite the user query so a retrieval system can search GAAP and SEC guidance. "
            "Preserve every constraint, standard number, and acronym. Return only the rewritten query."
        ),
        user=query,
    )
    context = "\n\n".join(chunk.text for chunk in chunks)
    if version == "v1":
        system = "Answer the question. You may add helpful context that is not in the passages."
    else:
        system = "Answer using only the passages. Do not add facts that are not in the passages."
    answer = client.complete_text(
        system=system,
        user=f"Question:\n{query}\n\nPassages:\n{context}",
    )
    return rewrite or query, answer or context
