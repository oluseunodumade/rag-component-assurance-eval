"""Deterministic agent loop over a logged RAG trace.

The controller plans from the user question, calls a route-specific retrieval tool,
grades the context, and retries once when the first hop is weak or the tool was wrong.
The critic then accepts the draft or stops the loop for a revision.
"""

from __future__ import annotations

from app.schemas import (
    AgentAction,
    AgentHandoff,
    ComponentId,
    AgentResult,
    AgentToolCall,
    AgentToolUse,
    AuditItem,
    QueryModificationResult,
    QueryRoutingResult,
    ResponseGenerationResult,
    RetrievalResult,
)
from app.textutil import sentences


def build_agent_result(
    item: AuditItem,
    *,
    modification: QueryModificationResult,
    routing: QueryRoutingResult,
    retrieval: RetrievalResult,
    response: ResponseGenerationResult,
) -> AgentResult:
    """Decide the plan, the tool calls, and whether the critic accepts the draft."""
    plan = _plan(item.user_query)
    retrieved_ids = [chunk.chunk_id for chunk in item.retrieved_chunks]
    first = AgentToolCall(
        hop=1,
        tool=_tool_name(item.actual_route),
        query=item.rewritten_query,
        chunk_ids=retrieved_ids,
        sufficient=retrieval.passed and routing.passed,
        observation=retrieval.rationale,
    )
    calls = [first]
    retried = not retrieval.passed or not routing.passed
    if retried:
        missed = [chunk for chunk in item.gold_chunks if chunk.chunk_id not in set(retrieved_ids)]
        recovered = missed or list(item.gold_chunks)
        calls.append(
            AgentToolCall(
                hop=2,
                tool=_tool_name(item.expected_route),
                query=_refined_query(item.user_query, recovered[0].text if recovered else item.user_query),
                chunk_ids=[chunk.chunk_id for chunk in recovered],
                sufficient=bool(recovered),
                observation=_retry_observation(recovered),
            )
        )
    accepted = modification.passed and routing.passed and retrieval.passed and response.passed
    stop_reason = _stop_reason(
        accepted=accepted,
        modification=modification,
        routing=routing,
        retrieval=retrieval,
    )
    verdict = "accept" if accepted else "revise"
    return AgentResult(
        plan=plan,
        tool_calls=calls,
        hops=len(calls),
        retried=retried,
        critic_verdict=verdict,
        stop_reason=stop_reason,
        passed=accepted,
        rationale=stop_reason,
        actions=(
            specialist_actions := _actions(
                item,
                plan=plan,
                calls=calls,
                verdict=verdict,
                modification=modification,
                routing=routing,
                retrieval=retrieval,
                response=response,
            )
        ),
        handoffs=_handoffs(specialist_actions, stop_reason),
    )


def _plan(query: str) -> list[str]:
    parts = sentences(query)
    questions = [part for part in parts if part.endswith("?")]
    return questions or parts[:1] or [query.strip()]


def _tool_name(route: str) -> str:
    if "audit" in route.casefold():
        return "search_assurance_index"
    return "search_reporting_index"


def _refined_query(question: str, passage: str) -> str:
    lead = sentences(passage)[0] if sentences(passage) else passage
    opening = _plan(question)[0]
    return f"{opening} Focus on: {lead}"


def _retry_observation(recovered: list[object]) -> str:
    count = len(recovered)
    noun = "passage" if count == 1 else "passages"
    return (
        f"Graded the first context as insufficient. The second tool call recovered {count} reference {noun} "
        "that the first hop missed. The draft answer was not regenerated from that context."
    )


def _actions(
    item: AuditItem,
    *,
    plan: list[str],
    calls: list[AgentToolCall],
    verdict: str,
    modification: QueryModificationResult,
    routing: QueryRoutingResult,
    retrieval: RetrievalResult,
    response: ResponseGenerationResult,
) -> list[AgentAction]:
    """Explain the specialist agent, its plan, and its tool calls for every component."""
    gold_ids = ", ".join(chunk.chunk_id for chunk in item.gold_chunks) or "none"
    index_tool = _tool_name(item.expected_route)
    first = calls[0]
    kept_question = item.rewritten_query.strip() == item.user_query.strip()
    return [
        AgentAction(
            component="preprocess",
            agent="Corpus Agent",
            plan=(
                f"Identify the {len(item.gold_chunks)} reference passages for {item.scenario}, "
                "parse them as policy text, then chunk, embed, and index them for "
                f"{index_tool}."
            ),
            tools=[
                AgentToolUse(
                    name="identify_documents",
                    sub_id="1.1",
                    purpose="Choose the documents that can answer this case.",
                    call=f"category={item.category}; scenario={item.scenario}",
                    result=f"{len(item.gold_chunks)} reference passages: {gold_ids}.",
                ),
                AgentToolUse(
                    name="parse_passages",
                    sub_id="1.2",
                    purpose="Read the selected documents into text.",
                    call="parse policy text and keep section headers",
                    result="Parsed as text. This trace has no separate image or table parse.",
                ),
                AgentToolUse(
                    name="chunk_documents",
                    sub_id="1.3",
                    purpose="Split the parsed documents into retrieval chunks.",
                    call=f"chunk {len(item.gold_chunks)} passages",
                    result=gold_ids,
                ),
                AgentToolUse(
                    name="embed_passages",
                    sub_id="1.4",
                    purpose="Embed each chunk for the search index.",
                    call="embed the reference passages",
                    result=f"Cosine similarity with the retrieved set is {retrieval.embedding_cosine_similarity:.2f}.",
                ),
                AgentToolUse(
                    name="build_index",
                    sub_id="1.5",
                    purpose="Make the chunks searchable by the retrieval tool.",
                    call=f"index for {index_tool}",
                    result=f"{len(item.retrieved_chunks)} chunks are available to search.",
                ),
            ],
        ),
        AgentAction(
            component="storage",
            agent="Storage Agent",
            plan="Store the chunk embeddings and the passage metadata the retrieval tool will read.",
            tools=[
                AgentToolUse(
                    name="store_embeddings",
                    sub_id="2.1",
                    purpose="Persist the vectors for the indexed passages.",
                    call=f"write embeddings for {gold_ids}",
                    result="Stored." if item.retrieved_chunks else "Empty. No embeddings were stored.",
                ),
                AgentToolUse(
                    name="store_metadata",
                    sub_id="2.2",
                    purpose="Persist the chunk ids and sources beside the vectors.",
                    call="write chunk_id and source for each passage",
                    result=gold_ids,
                ),
            ],
        ),
        AgentAction(
            component="transform",
            agent="Query Agent",
            plan=(
                f"Keep the user question, embed it, and route it to {routing.expected_route}."
                if kept_question and modification.passed
                else (
                    f"Rewrite the question, embed the rewrite, and route it to {routing.expected_route}. "
                    f"Preservation scored {modification.preservation_score}/5."
                )
            ),
            tools=[
                AgentToolUse(
                    name="receive_query",
                    sub_id="3.1",
                    purpose="Take the analyst's question as the task.",
                    call=_clip(plan[0]),
                    result="Question received.",
                ),
                AgentToolUse(
                    name="rewrite_query",
                    sub_id="3.2",
                    purpose="Produce the search query while keeping the asked constraint.",
                    call=_clip(item.user_query),
                    result=(
                        f"Kept the question. Preservation {modification.preservation_score}/5."
                        if kept_question
                        else f"Rewrote the question. Preservation {modification.preservation_score}/5. Search query: {_clip(item.rewritten_query)}"
                    ),
                ),
                AgentToolUse(
                    name="route_query",
                    sub_id="3.3",
                    purpose="Choose the index the retrieval tool should search.",
                    call=f"expected={routing.expected_route}",
                    result=(
                        f"Selected {routing.actual_route} at confidence {routing.routing_confidence:.2f}."
                    ),
                ),
                AgentToolUse(
                    name="embed_query",
                    sub_id="3.4",
                    purpose="Embed the search query for the chosen index.",
                    call=_clip(item.rewritten_query),
                    result="Query embedding ready.",
                ),
            ],
        ),
        AgentAction(
            component="retrieval",
            agent="Retrieval Agent",
            plan=(
                f"Search with {first.tool}. The first context is not sufficient, so call "
                f"{calls[-1].tool} again for the missed passages, then assemble the prompt from the first hop."
                if len(calls) > 1
                else f"Search once with {first.tool} and assemble the prompt from the passages it returns."
            ),
            tools=[
                *[
                    AgentToolUse(
                        name=call.tool,
                        sub_id="4.1",
                        purpose="Retrieve passages for the current search query." if call.hop == 1 else "Retry retrieval for the passages the first hop missed.",
                        call=_clip(call.query),
                        result=(
                            f"Hop {call.hop}: {', '.join(call.chunk_ids) or 'no passages'}. "
                            + ("Context sufficient." if call.sufficient else "Context insufficient.")
                        ),
                    )
                    for call in calls
                ],
                AgentToolUse(
                    name="assemble_prompt",
                    sub_id="4.2",
                    purpose="Place the question and the first-hop passages into the generation prompt.",
                    call=f"top passages {', '.join(first.chunk_ids) or 'none'}",
                    result=(
                        f"Prompt intact. Precision {retrieval.context_precision:.2f}, "
                        f"recall {retrieval.context_recall:.2f}, F1 {retrieval.f1_score:.2f}."
                        if retrieval.prompt_assembly_valid
                        else f"Prompt broken. F1 {retrieval.f1_score:.2f}."
                    ),
                ),
            ],
        ),
        AgentAction(
            component="generation",
            agent="Generation Agent",
            plan="Draft the answer from the assembled prompt, then check each sentence against the retrieved passages.",
            tools=[
                AgentToolUse(
                    name="draft_answer",
                    sub_id="5.1",
                    purpose="Write the answer using the assembled prompt.",
                    call=_clip(item.user_query),
                    result=(
                        f"Draft returned. Faithfulness {response.faithfulness_score:.2f}, "
                        f"relevancy {response.answer_relevancy_score:.2f}, "
                        f"alignment tier {response.ground_truth_alignment_tier}."
                    ),
                ),
                AgentToolUse(
                    name="grounding_judge",
                    sub_id="5.1",
                    purpose="Mark sentences that the retrieved passages support.",
                    call="compare the draft with the retrieved passages and the reference answer",
                    result=response.judge_rationale,
                ),
            ],
        ),
        AgentAction(
            component="agent",
            agent="Orchestrator",
            plan=(
                "Plan the question, choose the retrieval tool, grade the context, and "
                f"{'accept' if verdict == 'accept' else 'send back'} the draft."
            ),
            tools=[
                AgentToolUse(
                    name="plan_task",
                    sub_id="6.1",
                    purpose="Turn the question into the steps the other agents must finish.",
                    call=f"{len(plan)} question step(s)",
                    result=" ".join(plan),
                ),
                AgentToolUse(
                    name="route_tool",
                    sub_id="6.2",
                    purpose="Pick the retrieval tool for the first hop.",
                    call=f"route={routing.actual_route}",
                    result=f"Called {first.tool}.",
                ),
                AgentToolUse(
                    name="grade_context",
                    sub_id="6.3",
                    purpose="Decide whether the retrieved context is enough to answer.",
                    call=f"F1 {retrieval.f1_score:.2f}; route {'matched' if routing.passed else 'mismatched'}",
                    result=(
                        f"Retried with {calls[-1].tool}."
                        if len(calls) > 1
                        else "One hop was enough. No retry."
                    ),
                ),
                AgentToolUse(
                    name="critique_draft",
                    sub_id="6.4",
                    purpose="Accept the draft or stop the loop for a revision.",
                    call=f"faithfulness {response.faithfulness_score:.2f}; preservation {modification.preservation_score}/5",
                    result=f"Verdict: {verdict}.",
                ),
            ],
        ),
    ]


def _handoffs(actions: list[AgentAction], stop_reason: str) -> list[AgentHandoff]:
    """Record specialist agents handing work to one another, with one retrieval retry."""
    by_component = {action.component: action for action in actions}
    corpus = by_component["preprocess"]
    storage = by_component["storage"]
    query = by_component["transform"]
    retrieval = by_component["retrieval"]
    generation = by_component["generation"]
    searches = [tool for tool in retrieval.tools if tool.name.startswith("search_")]
    assemble = next((tool for tool in retrieval.tools if tool.name == "assemble_prompt"), None)
    messages: list[tuple[str, str, ComponentId, str, list[str], str]] = [
        (
            "Orchestrator",
            corpus.agent,
            "preprocess",
            corpus.plan,
            [tool.name for tool in corpus.tools],
            corpus.tools[-1].result,
        ),
        (
            corpus.agent,
            storage.agent,
            "storage",
            "Store the passages the Corpus Agent just prepared.",
            [tool.name for tool in storage.tools],
            storage.tools[-1].result,
        ),
        (
            storage.agent,
            query.agent,
            "transform",
            "The index is ready. Rewrite the question and choose the search tool.",
            [tool.name for tool in query.tools],
            query.tools[-1].result,
        ),
        (
            query.agent,
            retrieval.agent,
            "retrieval",
            searches[0].purpose,
            [searches[0].name],
            searches[0].result,
        ),
    ]
    if len(searches) > 1:
        messages.append(
            (
                retrieval.agent,
                query.agent,
                "transform",
                "The first context is not enough. Choose the index again.",
                ["route_query"],
                f"Retrieval asked the Query Agent to re-route after {searches[0].name} came back short.",
            )
        )
        second_tools = [searches[1].name, *([assemble.name] if assemble else [])]
        second_result = f"{searches[1].result} {assemble.result}" if assemble else searches[1].result
        messages.append(
            (
                query.agent,
                retrieval.agent,
                "retrieval",
                searches[1].purpose,
                second_tools,
                second_result,
            )
        )
    messages.extend(
        [
            (
                retrieval.agent,
                generation.agent,
                "generation",
                "Draft the answer from the passages I assembled.",
                [tool.name for tool in generation.tools],
                generation.tools[-1].result,
            ),
            (
                generation.agent,
                "Orchestrator",
                "agent",
                "Here is the draft. Accept it or send it back.",
                ["critique_draft"],
                stop_reason,
            ),
        ]
    )
    return [
        _dispatch(index + 1, source, target, component, instruction, tools, observation)
        for index, (source, target, component, instruction, tools, observation) in enumerate(messages)
    ]


def _dispatch(
    step: int,
    source: str,
    target: str,
    component: ComponentId,
    instruction: str,
    tools: list[str],
    observation: str,
) -> AgentHandoff:
    return AgentHandoff(
        step=step,
        source=source,
        target=target,
        component=component,
        instruction=instruction,
        tools=tools,
        observation=observation,
    )


def _clip(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _stop_reason(
    *,
    accepted: bool,
    modification: QueryModificationResult,
    routing: QueryRoutingResult,
    retrieval: RetrievalResult,
) -> str:
    if accepted:
        return "The agent graded the retrieved context as sufficient and the critic accepted the draft."
    if not modification.passed:
        return "The agent rejected the plan because the search query did not preserve the original constraint."
    if not routing.passed:
        return (
            "The first tool call used the wrong index. The agent called the expected index on the next hop, "
            "and the critic rejected the draft because it was not regenerated."
        )
    if not retrieval.passed:
        return (
            "The agent graded the first context as insufficient and retrieved the missed passages. "
            "The draft was not regenerated from that second hop, so the critic rejected it."
        )
    return "The agent graded the context as sufficient, then the critic rejected the draft for an ungrounded claim and stopped."
