"""LangGraph orchestration for the component audit and the agent loop."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent import build_agent_result
from app.config import get_settings
from app.criteria import UNGROUNDED_EXTRA_TIERS
from app.evaluators.query_transformation import QueryModificationEvaluator, QueryRoutingEvaluator
from app.evaluators.response_generation import ResponseQualityJudge
from app.evaluators.retrieval_quality import PromptAugmentationEvaluator, RetrievalQualityEvaluator, run_retrieval_stage
from app.llm import LLMNotConfiguredError
from app.schemas import (
    AuditItem,
    AuditItemResult,
    BatchAuditReport,
    AgentResult,
    QueryModificationResult,
    QueryRoutingResult,
    ResponseGenerationResult,
    RetrievalResult,
    StageTiming,
)

logger = logging.getLogger(__name__)


class ItemTrace(BaseModel):
    """Working state for one item as it moves through the graph."""

    item: AuditItem
    query_modification: QueryModificationResult | None = None
    query_routing: QueryRoutingResult | None = None
    retrieval: RetrievalResult | None = None
    response_generation: ResponseGenerationResult | None = None
    agent: AgentResult | None = None


class PipelineState(TypedDict):
    """Shared state for a batch audit run."""

    mock_mode: bool
    traces: list[ItemTrace]
    report: BatchAuditReport | None


def query_transformation(state: PipelineState) -> dict[str, object]:
    """Stage 1: score the rewrite and the index route."""
    mock_mode = state["mock_mode"]
    modification = QueryModificationEvaluator()
    routing = QueryRoutingEvaluator()
    traces = [
        trace.model_copy(
            update={
                "query_modification": modification.evaluate(trace.item, mock_mode=mock_mode),
                "query_routing": routing.evaluate(trace.item, mock_mode=mock_mode),
            }
        )
        for trace in state["traces"]
    ]
    return {"traces": traces}


def retrieval_quality(state: PipelineState) -> dict[str, object]:
    """Stage 2: score retrieval overlap, embedding drift, and prompt assembly."""
    mock_mode = state["mock_mode"]
    quality = RetrievalQualityEvaluator()
    prompt_evaluator = PromptAugmentationEvaluator()
    traces = [
        trace.model_copy(
            update={
                "retrieval": run_retrieval_stage(
                    trace.item,
                    mock_mode=mock_mode,
                    quality=quality,
                    prompt_evaluator=prompt_evaluator,
                ),
            }
        )
        for trace in state["traces"]
    ]
    return {"traces": traces}


def response_generation(state: PipelineState) -> dict[str, object]:
    """Stage 3: judge the generated answer."""
    mock_mode = state["mock_mode"]
    judge = ResponseQualityJudge()
    traces = [
        trace.model_copy(
            update={
                "response_generation": judge.evaluate(trace.item, mock_mode=mock_mode),
            }
        )
        for trace in state["traces"]
    ]
    return {"traces": traces}


def agent_control(state: PipelineState) -> dict[str, object]:
    """Stage 4: plan, grade the tool call, retry once, and accept or revise the draft."""
    traces = []
    for trace in state["traces"]:
        traces.append(
            trace.model_copy(
                update={
                    "agent": build_agent_result(
                        trace.item,
                        modification=_require(trace.query_modification, "query modification"),
                        routing=_require(trace.query_routing, "query routing"),
                        retrieval=_require(trace.retrieval, "retrieval"),
                        response=_require(trace.response_generation, "response generation"),
                    )
                }
            )
        )
    return {"traces": traces}


def aggregate_report(state: PipelineState) -> dict[str, object]:
    """Stage 4: roll component traces into dataset KPIs."""
    results = [_to_item_result(trace) for trace in state["traces"]]
    total = len(results)
    report = BatchAuditReport(
        total_items=total,
        overall_pass_rate=_mean([1.0 if result.passed else 0.0 for result in results]),
        routing_accuracy=_mean(
            [
                1.0
                if result.query_routing.expected_route.casefold()
                == result.query_routing.actual_route.casefold()
                else 0.0
                for result in results
            ]
        ),
        avg_retrieval_f1=_mean([result.retrieval.f1_score for result in results]),
        avg_faithfulness=_mean([result.response_generation.faithfulness_score for result in results]),
        ungrounded_extras_rate=_mean(
            [
                1.0 if result.response_generation.ground_truth_alignment_tier in UNGROUNDED_EXTRA_TIERS else 0.0
                for result in results
            ]
        ),
        items=results,
    )
    return {"report": report}


def build_evaluation_graph() -> object:
    """Compile the sequential component graph."""
    graph = StateGraph(PipelineState)
    graph.add_node("query_transformation", query_transformation)
    graph.add_node("retrieval_quality", retrieval_quality)
    graph.add_node("response_generation", response_generation)
    graph.add_node("agent_control", agent_control)
    graph.add_node("aggregate_report", aggregate_report)
    graph.add_edge(START, "query_transformation")
    graph.add_edge("query_transformation", "retrieval_quality")
    graph.add_edge("retrieval_quality", "response_generation")
    graph.add_edge("response_generation", "agent_control")
    graph.add_edge("agent_control", "aggregate_report")
    graph.add_edge("aggregate_report", END)
    return graph.compile()


evaluation_graph = build_evaluation_graph()


def run_evaluation(items: Sequence[AuditItem], *, mock_mode: bool) -> BatchAuditReport:
    """Run every item through the four stages and return the batch report."""
    if not mock_mode and not get_settings().llm_configured:
        raise LLMNotConfiguredError(
            "OPENAI_API_KEY is not set. Call the API with mock_mode=true or add a key to backend/.env."
        )
    logger.info("Running component audit on %s item(s), mock_mode=%s", len(items), mock_mode)
    if not items:
        return BatchAuditReport(
            total_items=0,
            overall_pass_rate=0.0,
            routing_accuracy=0.0,
            avg_retrieval_f1=0.0,
            avg_faithfulness=0.0,
            ungrounded_extras_rate=0.0,
            items=[],
        )
    final_state = evaluation_graph.invoke(
        {
            "mock_mode": mock_mode,
            "traces": [ItemTrace(item=item) for item in items],
            "report": None,
        }
    )
    report = final_state["report"]
    if not isinstance(report, BatchAuditReport):
        raise RuntimeError("Evaluation graph did not produce a report")
    return report


def _to_item_result(trace: ItemTrace) -> AuditItemResult:
    modification = _require(trace.query_modification, "query modification")
    routing = _require(trace.query_routing, "query routing")
    retrieval = _require(trace.retrieval, "retrieval")
    response = _require(trace.response_generation, "response generation")
    agent = _require(trace.agent, "agent control")
    return AuditItemResult(
        query_id=trace.item.query_id,
        category=trace.item.category,
        scenario=trace.item.scenario,
        user_query=trace.item.user_query,
        rewritten_query=trace.item.rewritten_query,
        retrieved_chunks=list(trace.item.retrieved_chunks),
        gold_chunks=list(trace.item.gold_chunks),
        assembled_prompt=trace.item.assembled_prompt,
        generated_response=trace.item.generated_response,
        ground_truth_answer=trace.item.ground_truth_answer,
        query_modification=modification,
        query_routing=routing,
        retrieval=retrieval,
        response_generation=response,
        agent=agent,
        passed=modification.passed and routing.passed and retrieval.passed and response.passed,
        timings=_stage_timings(trace.item.query_id, trace.item.user_query),
    )


_T = TypeVar("_T")


def _require(value: _T | None, stage: str) -> _T:
    if value is None:
        raise RuntimeError(f"Pipeline reached aggregate without {stage}")
    return value


def _stage_timings(query_id: str, user_query: str) -> list[StageTiming]:
    """Stable per-stage badges so the playground can animate a trace."""
    profile = (
        ("input", 48, 40),
        ("rewriter", 142, 160),
        ("retriever", 89, 380),
        ("assembler", 64, 520),
        ("judge", 610, 220),
        ("agent", 240, 180),
    )
    extra_tokens = max(0, len(user_query) // 12)
    return [
        StageTiming(
            stage=stage,
            latency_ms=latency,
            tokens=tokens + extra_tokens,
            cost_usd=round((tokens + extra_tokens) * 0.0000025, 6),
        )
        for stage, latency, tokens in profile
    ]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
