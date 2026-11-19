"""Input records and component-evaluation outputs for the audit API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PromptVersion = Literal["v1", "v2"]
ExperimentName = Literal["baseline", "candidate"]

AmbiguityLevel = Literal["low", "medium", "high"]
AuditCategory = Literal["Financial_Reporting", "Audit_Assurance"]

ALIGNMENT_TIER_LABELS: dict[int, str] = {
    3: "Complete Alignment",
    2: "Partial Alignment",
    1: "Complete Alignment + Ungrounded Extras",
    0: "Partial Alignment + Hallucinated Extras",
}


class ContextChunk(BaseModel):
    """A source passage used as gold context or as a retrieval candidate."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)


class RetrievedChunk(ContextChunk):
    """A chunk returned by the retriever, with its score and relevance label."""

    score: float = Field(ge=0.0, le=1.0)
    relevant: bool


class AuditItem(BaseModel):
    """One logged RAG transaction and the reference data needed to audit it."""

    query_id: str = Field(min_length=1)
    category: AuditCategory
    scenario: str = Field(min_length=1)
    user_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)
    expected_route: str = Field(min_length=1)
    actual_route: str = Field(min_length=1)
    routing_confidence: float = Field(ge=0.0, le=1.0)
    retrieved_chunks: list[RetrievedChunk]
    gold_chunks: list[ContextChunk] = Field(min_length=1)
    assembled_prompt: str
    generated_response: str
    ground_truth_answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_chunk_ids(self) -> AuditItem:
        _require_unique_ids([chunk.chunk_id for chunk in self.retrieved_chunks], "retrieved_chunks")
        _require_unique_ids([chunk.chunk_id for chunk in self.gold_chunks], "gold_chunks")
        return self


class EvaluateRequest(BaseModel):
    """Selects the judge, an optional ad-hoc query, and playground controls."""

    mock_mode: bool = True
    custom_query: str | None = Field(default=None, max_length=2000)
    query_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)
    prompt_version: PromptVersion = "v2"
    experiment: ExperimentName | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"mock_mode": True},
                {
                    "mock_mode": True,
                    "custom_query": "Under ASC 606, when is revenue recognized for a contract with a customer?",
                    "top_k": 2,
                    "prompt_version": "v2",
                },
            ]
        }
    )


class QueryModificationResult(BaseModel):
    """Intent preservation and ambiguity of the query rewrite."""

    preservation_score: int = Field(ge=1, le=5, description="1 = intent lost, 5 = intent fully preserved.")
    ambiguity_level: AmbiguityLevel
    passed: bool
    rationale: str


class QueryRoutingResult(BaseModel):
    """Comparison of the expected index route and the route the system took."""

    expected_route: str
    actual_route: str
    routing_confidence: float = Field(ge=0.0, le=1.0)
    passed: bool
    rationale: str


class RetrievalResult(BaseModel):
    """Retrieval overlap, embedding drift against gold context, and prompt integrity."""

    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    embedding_cosine_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity between retrieved context and gold context. Drift is 1 minus this value.",
    )
    prompt_assembly_valid: bool
    passed: bool
    rationale: str


class GroundingSpan(BaseModel):
    """One sentence of the model answer marked grounded or ungrounded."""

    text: str
    label: Literal["grounded", "ungrounded"]
    reason: str


class StageTiming(BaseModel):
    """Latency, token, and cost badge for one playground stage."""

    stage: str
    latency_ms: int = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0)


class AgentToolCall(BaseModel):
    """One retrieval-tool call inside the agent loop."""

    hop: int = Field(ge=1)
    tool: str = Field(min_length=1)
    query: str = Field(min_length=1)
    chunk_ids: list[str] = Field(default_factory=list)
    sufficient: bool
    observation: str


ComponentId = Literal["preprocess", "storage", "transform", "retrieval", "generation", "agent"]


class AgentToolUse(BaseModel):
    """One tool the component agent called, with the call and what came back."""

    name: str = Field(min_length=1)
    sub_id: str = Field(min_length=1)
    purpose: str
    call: str
    result: str


class AgentAction(BaseModel):
    """What one specialist agent planned and which tools it used."""

    component: ComponentId
    agent: str = Field(min_length=1)
    plan: str
    tools: list[AgentToolUse] = Field(min_length=1)


class AgentHandoff(BaseModel):
    """One dispatch from the orchestrator to a specialist agent."""

    step: int = Field(ge=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    component: ComponentId
    instruction: str
    tools: list[str] = Field(min_length=1)
    observation: str


class AgentResult(BaseModel):
    """Plan, tool calls, retry decision, and critic verdict for one query."""

    plan: list[str] = Field(min_length=1)
    tool_calls: list[AgentToolCall] = Field(min_length=1)
    hops: int = Field(ge=1)
    retried: bool
    critic_verdict: Literal["accept", "revise"]
    stop_reason: str
    passed: bool
    rationale: str
    actions: list[AgentAction] = Field(min_length=6)
    handoffs: list[AgentHandoff] = Field(min_length=1)


class ResponseGenerationResult(BaseModel):
    """LLM-as-a-Judge scores for the generated answer."""

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    answer_relevancy_score: float = Field(ge=0.0, le=1.0)
    ground_truth_alignment_tier: int = Field(
        ge=0,
        le=3,
        description=(
            "3 = Complete Alignment, 2 = Partial Alignment, "
            "1 = Complete Alignment + Ungrounded Extras, "
            "0 = Partial Alignment + Hallucinated Extras."
        ),
    )
    passed: bool
    judge_rationale: str
    grounding_spans: list[GroundingSpan] = Field(default_factory=list)


class AuditItemResult(BaseModel):
    """Component traces for a single audited query."""

    query_id: str
    category: AuditCategory
    scenario: str
    user_query: str
    rewritten_query: str = ""
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    gold_chunks: list[ContextChunk] = Field(default_factory=list)
    assembled_prompt: str = ""
    generated_response: str = ""
    ground_truth_answer: str = ""
    query_modification: QueryModificationResult
    query_routing: QueryRoutingResult
    retrieval: RetrievalResult
    response_generation: ResponseGenerationResult
    agent: AgentResult
    passed: bool = Field(description="True when every component stage passed its gate.")
    timings: list[StageTiming] = Field(default_factory=list)


class BatchAuditReport(BaseModel):
    """Dataset-level KPIs plus the per-item component traces."""

    total_items: int = Field(ge=0)
    overall_pass_rate: float = Field(ge=0.0, le=1.0)
    routing_accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of items whose actual route matches the expected route.",
    )
    avg_retrieval_f1: float = Field(ge=0.0, le=1.0)
    avg_faithfulness: float = Field(ge=0.0, le=1.0)
    ungrounded_extras_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Share of items in alignment tier 1 or 0 (ungrounded or hallucinated extras).",
    )
    items: list[AuditItemResult]


class HealthResponse(BaseModel):
    """Liveness payload for the evaluation service."""

    status: Literal["ok"]
    llm_configured: bool
    dataset_size: int = Field(ge=0)
    mock_mode_default: bool = True


def _require_unique_ids(chunk_ids: list[str], field_name: str) -> None:
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"{field_name} contains duplicate chunk_id values")
