"""Unit and integration gates for the component assurance pipeline."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dataset import load_dataset
from app.evaluators.query_transformation import QueryModificationEvaluator, QueryRoutingEvaluator
from app.evaluators.response_generation import ResponseQualityJudge
from app.evaluators.retrieval_quality import (
    PromptAugmentationEvaluator,
    RetrievalQualityEvaluator,
    run_retrieval_stage,
)
from app.main import app
from app.pipeline import evaluation_graph, run_evaluation
from app.schemas import AuditItem, ContextChunk, RetrievedChunk

_QUERY = "Under ASC 606, when is revenue recognized for a contract with a customer?"
_REWRITE = "Under ASC 606, when is revenue recognized for a contract with a customer using the five-step model?"
_CONTEXT = (
    "Under ASC 606, revenue is recognized for a contract with a customer when the entity "
    "satisfies a performance obligation by transferring control of a promised good or service."
)
_SECOND = (
    "The five-step model identifies the contract, identifies the performance obligations, "
    "determines the transaction price, allocates that price, and recognizes revenue as each "
    "performance obligation is satisfied."
)
_ANNUAL_TEST = (
    "Goodwill is tested for impairment at least annually and also when a triggering event "
    "indicates that fair value may be below carrying amount."
)
_MEASUREMENT = (
    "An impairment loss is recognized when the carrying amount of a reporting unit exceeds "
    "its fair value, and the loss is limited to the goodwill balance."
)
_DISTRACTOR = "Revenue is recognized when control of a promised good or service transfers to a customer."
_EBITDA_QUERY = (
    "How does Regulation G require a company to reconcile non-GAAP EBITDA to the most "
    "directly comparable GAAP measure?"
)
_EBITDA_FACT = (
    "Regulation G requires a company to reconcile non-GAAP EBITDA to the most directly "
    "comparable GAAP measure, and that measure is net income."
)
_EBITDA_EXTRA = (
    "Management privately forecast that adjusted EBITDA will double next year based on an "
    "unreleased board memo."
)


def _prompt(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)
    return (
        "You are a careful assistant. Answer the user using only the context below.\n\n"
        f"User query:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )


def _retrieved(chunk_id: str, text: str, *, relevant: bool, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=f"unit-{chunk_id}",
        score=score,
        relevant=relevant,
    )


def _gold(chunk_id: str, text: str) -> ContextChunk:
    return ContextChunk(chunk_id=chunk_id, text=text, source=f"unit-{chunk_id}")


def _item(
    *,
    query_id: str = "T-1",
    user_query: str = _QUERY,
    rewritten_query: str = _REWRITE,
    expected_route: str = "Financial_Reporting",
    actual_route: str = "Financial_Reporting",
    routing_confidence: float = 0.91,
    retrieved: list[RetrievedChunk] | None = None,
    gold: list[ContextChunk] | None = None,
    prompt: str | None = None,
    answer: str = _CONTEXT,
    ground_truth: str = _CONTEXT,
) -> AuditItem:
    retrieved_chunks = retrieved or [_retrieved("ctx-1", _CONTEXT, relevant=True)]
    gold_chunks = gold or [_gold("ctx-1", _CONTEXT)]
    return AuditItem(
        query_id=query_id,
        category="Financial_Reporting",
        scenario="unit fixture",
        user_query=user_query,
        rewritten_query=rewritten_query,
        expected_route=expected_route,
        actual_route=actual_route,
        routing_confidence=routing_confidence,
        retrieved_chunks=retrieved_chunks,
        gold_chunks=gold_chunks,
        assembled_prompt=prompt if prompt is not None else _prompt(user_query, [chunk.text for chunk in retrieved_chunks]),
        generated_response=answer,
        ground_truth_answer=ground_truth,
    )


def test_query_modification_preserves_a_specific_rewrite() -> None:
    result = QueryModificationEvaluator().evaluate(_item(), mock_mode=True)

    assert result.preservation_score == 5
    assert result.ambiguity_level == "low"
    assert result.passed is True
    assert "Pass gate met" in result.rationale


def test_query_modification_fails_when_the_rewrite_drops_scope() -> None:
    item = _item(rewritten_query="Explain revenue")
    result = QueryModificationEvaluator().evaluate(item, mock_mode=True)

    assert result.preservation_score == 1
    assert result.ambiguity_level == "high"
    assert result.passed is False


def test_query_routing_passes_on_an_exact_confident_match() -> None:
    result = QueryRoutingEvaluator().evaluate(_item(), mock_mode=True)

    assert result.expected_route == "Financial_Reporting"
    assert result.actual_route == "Financial_Reporting"
    assert result.passed is True


def test_query_routing_fails_when_the_index_is_wrong() -> None:
    item = _item(actual_route="Audit_Assurance")
    result = QueryRoutingEvaluator().evaluate(item, mock_mode=True)

    assert result.passed is False
    assert "does not match" in result.rationale


def test_query_routing_fails_when_confidence_is_below_the_gate() -> None:
    item = _item(routing_confidence=0.42)
    result = QueryRoutingEvaluator().evaluate(item, mock_mode=True)

    assert result.expected_route == result.actual_route
    assert result.passed is False


def test_retrieval_metrics_are_perfect_when_gold_context_is_returned() -> None:
    item = _item()
    measurement = RetrievalQualityEvaluator().evaluate(item, mock_mode=True)

    assert measurement.context_precision == 1.0
    assert measurement.context_recall == 1.0
    assert measurement.f1_score == 1.0
    assert measurement.embedding_cosine_similarity == 1.0


def test_retrieval_recall_drops_when_a_reference_chunk_is_missing() -> None:
    retrieved = [
        _retrieved("unit-1", _CONTEXT, relevant=True),
        _retrieved("off-topic", _DISTRACTOR, relevant=False, score=0.55),
    ]
    gold = [_gold("unit-1", _CONTEXT), _gold("annual", _ANNUAL_TEST), _gold("measure", _MEASUREMENT)]
    item = _item(retrieved=retrieved, gold=gold, answer=_CONTEXT, ground_truth=_CONTEXT)
    measurement = RetrievalQualityEvaluator().evaluate(item, mock_mode=True)

    assert measurement.context_precision == 0.5
    assert measurement.context_recall < 0.5
    assert measurement.f1_score < 0.5
    stage = run_retrieval_stage(item, mock_mode=True)
    assert stage.prompt_assembly_valid is True
    assert stage.passed is False


def test_prompt_assembly_accepts_an_intact_prompt() -> None:
    item = _item()
    assessment = PromptAugmentationEvaluator().evaluate(item)

    assert assessment.prompt_assembly_valid is True
    assert assessment.missing_query is False
    assert assessment.truncated_chunk_ids == ()


def test_prompt_assembly_fails_when_a_chunk_is_truncated() -> None:
    full = _item()
    truncated = _prompt(full.user_query, [_CONTEXT[:40]])
    item = _item(prompt=truncated)
    assessment = PromptAugmentationEvaluator().evaluate(item)
    stage = run_retrieval_stage(item, mock_mode=True)

    assert assessment.prompt_assembly_valid is False
    assert assessment.truncated_chunk_ids == ("ctx-1",)
    assert stage.context_precision == 1.0
    assert stage.passed is False


def test_response_judge_scores_complete_alignment_at_tier_3() -> None:
    result = ResponseQualityJudge().evaluate(_item(), mock_mode=True)

    assert result.faithfulness_score == 1.0
    assert result.answer_relevancy_score >= 0.5
    assert result.ground_truth_alignment_tier == 3
    assert result.passed is True


def test_response_judge_flags_ungrounded_extras_at_tier_1() -> None:
    chunk = _retrieved("ebitda", _EBITDA_FACT, relevant=True)
    item = _item(
        user_query=_EBITDA_QUERY,
        rewritten_query=_EBITDA_QUERY,
        retrieved=[chunk],
        gold=[_gold("ebitda", _EBITDA_FACT)],
        answer=f"{_EBITDA_FACT} {_EBITDA_EXTRA}",
        ground_truth=_EBITDA_FACT,
    )
    result = ResponseQualityJudge().evaluate(item, mock_mode=True)

    assert result.ground_truth_alignment_tier == 1
    assert result.passed is False
    assert "Ungrounded extra claims" in result.judge_rationale
    assert "unreleased board memo" in result.judge_rationale


def test_langgraph_runs_the_four_stages_in_order() -> None:
    edges = {(edge.source, edge.target) for edge in evaluation_graph.get_graph().edges}

    assert ("__start__", "query_transformation") in edges
    assert ("query_transformation", "retrieval_quality") in edges
    assert ("retrieval_quality", "response_generation") in edges
    assert ("response_generation", "agent_control") in edges
    assert ("agent_control", "aggregate_report") in edges
    assert ("aggregate_report", "__end__") in edges


def test_mock_pipeline_is_deterministic() -> None:
    items = load_dataset()
    first = run_evaluation(items, mock_mode=True)
    second = run_evaluation(items, mock_mode=True)

    assert first.model_dump() == second.model_dump()


def test_sample_audit_isolates_each_component_failure() -> None:
    report = run_evaluation(load_dataset(), mock_mode=True)
    by_id = {item.query_id: item for item in report.items}

    assert report.total_items == 8
    assert report.overall_pass_rate == 0.5
    assert report.routing_accuracy == pytest.approx(0.875)
    assert report.ungrounded_extras_rate == pytest.approx(0.125)
    assert by_id["Q-101"].passed is True
    assert by_id["Q-101"].response_generation.ground_truth_alignment_tier == 3

    modification = by_id["Q-105"]
    assert modification.query_modification.passed is False
    assert modification.query_modification.preservation_score == 2
    assert modification.query_routing.passed is True
    assert modification.retrieval.passed is True
    assert modification.response_generation.passed is True

    routing = by_id["Q-103"]
    assert routing.query_routing.passed is False
    assert routing.query_routing.actual_route == "Financial_Reporting"
    assert routing.query_routing.expected_route == "Audit_Assurance"
    assert routing.query_modification.passed is True
    assert routing.retrieval.passed is False
    assert routing.retrieval.context_recall == 0.0

    retrieval = by_id["Q-104"]
    assert retrieval.retrieval.passed is False
    assert retrieval.retrieval.context_recall < 0.5
    assert retrieval.retrieval.prompt_assembly_valid is True
    assert retrieval.query_modification.passed is True
    assert retrieval.query_routing.passed is True
    assert retrieval.response_generation.ground_truth_alignment_tier == 2
    assert retrieval.response_generation.passed is True

    extras = by_id["Q-102"]
    assert extras.response_generation.ground_truth_alignment_tier == 1
    assert extras.response_generation.passed is False
    assert "15% of variable fees" in extras.response_generation.judge_rationale
    assert extras.query_modification.passed is True
    assert extras.query_routing.passed is True
    assert extras.retrieval.passed is True


def test_agent_retries_a_weak_retrieval_and_accepts_a_clean_trace() -> None:
    report = run_evaluation(load_dataset(), mock_mode=True)
    by_id = {item.query_id: item for item in report.items}

    clean = by_id["Q-101"].agent
    assert clean.hops == 1
    assert clean.retried is False
    assert clean.critic_verdict == "accept"
    assert clean.passed is True
    assert clean.tool_calls[0].tool == "search_reporting_index"

    retry = by_id["Q-104"].agent
    assert retry.retried is True
    assert retry.hops == 2
    assert retry.critic_verdict == "revise"
    assert retry.passed is False
    assert retry.tool_calls[1].tool == "search_reporting_index"
    assert "second hop" in retry.stop_reason

    critic = by_id["Q-102"].agent
    assert critic.retried is False
    assert critic.critic_verdict == "revise"
    assert "ungrounded" in critic.stop_reason

    misroute = by_id["Q-103"].agent
    assert misroute.tool_calls[0].tool == "search_reporting_index"
    assert misroute.tool_calls[1].tool == "search_assurance_index"
    assert misroute.critic_verdict == "revise"

    actions = {action.component: action for action in by_id["Q-101"].agent.actions}
    assert set(actions) == {"preprocess", "storage", "transform", "retrieval", "generation", "agent"}
    assert actions["preprocess"].agent == "Corpus Agent"
    assert actions["storage"].agent == "Storage Agent"
    assert actions["transform"].agent == "Query Agent"
    assert actions["retrieval"].agent == "Retrieval Agent"
    assert [tool.name for tool in actions["retrieval"].tools] == ["search_reporting_index", "assemble_prompt"]
    assert actions["generation"].agent == "Generation Agent"
    assert actions["agent"].agent == "Orchestrator"
    clean_pairs = [(handoff.source, handoff.target) for handoff in by_id["Q-101"].agent.handoffs]
    assert clean_pairs == [
        ("Orchestrator", "Corpus Agent"),
        ("Corpus Agent", "Storage Agent"),
        ("Storage Agent", "Query Agent"),
        ("Query Agent", "Retrieval Agent"),
        ("Retrieval Agent", "Generation Agent"),
        ("Generation Agent", "Orchestrator"),
    ]
    retry_pairs = [(handoff.source, handoff.target) for handoff in by_id["Q-104"].agent.handoffs]
    assert ("Retrieval Agent", "Query Agent") in retry_pairs
    assert retry_pairs.count(("Query Agent", "Retrieval Agent")) == 2
    retry_tools = [tool.name for tool in by_id["Q-104"].agent.actions[3].tools]
    assert retry_tools == ["search_reporting_index", "search_reporting_index", "assemble_prompt"]
    switched = [tool.name for tool in by_id["Q-103"].agent.actions[3].tools]
    assert switched[:2] == ["search_reporting_index", "search_assurance_index"]


def test_ci_prompt_assembly_integrity_is_complete() -> None:
    """Release gate: every shipped prompt must contain the query and the full chunks."""
    report = run_evaluation(load_dataset(), mock_mode=True)
    intact = [item.query_id for item in report.items if item.retrieval.prompt_assembly_valid]
    integrity = len(intact) / report.total_items

    assert integrity == 1.0, f"Prompt assembly integrity is {integrity:.0%}, below the 100% gate."


def test_custom_query_runs_in_mock_mode() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/evaluate",
        json={
            "mock_mode": True,
            "custom_query": "Under ASC 606, when is revenue recognized for a contract with a customer?",
            "top_k": 2,
            "prompt_version": "v2",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_items"] == 1
    item = body["items"][0]
    assert item["query_id"] == "ADHOC"
    assert [timing["stage"] for timing in item["timings"]] == [
        "input",
        "rewriter",
        "retriever",
        "assembler",
        "judge",
        "agent",
    ]
    assert item["timings"][1]["latency_ms"] == 142
    assert item["response_generation"]["grounding_spans"]
    assert item["response_generation"]["grounding_spans"][0]["label"] == "grounded"


def test_custom_query_v1_surfaces_an_ungrounded_span() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/evaluate",
        json={
            "mock_mode": True,
            "custom_query": "Under ASC 842, how does a lessee classify a lease as finance or operating?",
            "top_k": 1,
            "prompt_version": "v1",
        },
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    labels = [span["label"] for span in item["response_generation"]["grounding_spans"]]
    assert "ungrounded" in labels
    assert item["query_routing"]["passed"] is False


def test_experiment_candidate_beats_v1_baseline() -> None:
    client = TestClient(app)
    baseline = client.post("/api/evaluate", json={"mock_mode": True, "experiment": "baseline"}).json()
    candidate = client.post("/api/evaluate", json={"mock_mode": True, "experiment": "candidate"}).json()

    assert candidate["routing_accuracy"] - baseline["routing_accuracy"] == pytest.approx(0.125)
    f1_delta = candidate["avg_retrieval_f1"] - baseline["avg_retrieval_f1"]
    extras_delta = baseline["ungrounded_extras_rate"] - candidate["ungrounded_extras_rate"]
    assert f1_delta == pytest.approx(0.125, abs=0.02)
    assert extras_delta > 0
    assert baseline["total_items"] == candidate["total_items"] == 8


def test_evaluate_api_returns_the_mock_audit() -> None:
    client = TestClient(app)
    response = client.post("/api/evaluate", json={"mock_mode": True})

    assert response.status_code == 200
    body = response.json()
    assert body["total_items"] == 8
    assert all(item["retrieval"]["prompt_assembly_valid"] for item in body["items"])
    assert client.get("/api/health").status_code == 200
    assert len(client.get("/api/dataset").json()) == 8
