import { TIER_LABELS, type AgentAction, type AgentHandoff, type AuditItemResult, type GroundingSpan } from "@/lib/types";
import { parentOf, type StageId } from "@/lib/stages";
import { formatRoute, score } from "@/lib/format";

function GroundedAnswer({ spans }: { spans: GroundingSpan[] }) {
  if (!spans.length) return <p className="text-sm text-zinc-500">No model output was returned for this trace.</p>;
  return (
    <p className="text-sm leading-6">
      {spans.map((span) => (
        <span key={span.text} className="group relative">
          <span
            className={
              span.label === "grounded"
                ? "rounded-sm bg-emerald-100/80 text-emerald-950"
                : "rounded-sm bg-amber-50 text-amber-950 underline decoration-rose-500 decoration-2 underline-offset-[3px]"
            }
          >
            {span.text}
          </span>{" "}
          <span className="pointer-events-none absolute left-0 top-full z-20 mt-1 hidden w-72 rounded-md border border-zinc-200 bg-white p-2 text-xs font-normal leading-5 text-zinc-700 shadow-card group-hover:block">
            {span.reason}
          </span>
        </span>
      ))}
    </p>
  );
}

function Bar({ label, value }: { label: string; value: number }) {
  const width = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div>
      <div className="flex items-baseline justify-between font-mono text-[11px] text-zinc-500">
        <span>{label}</span>
        <span className="text-zinc-950">{score(value)}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-zinc-100">
        <div className={`h-full rounded-full ${value >= 0.5 ? "bg-emerald-500" : "bg-rose-500"}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

function AgentFlow({ action, subFocus }: { action: AgentAction; subFocus: string | null }) {
  const focused = subFocus ? action.tools.filter((tool) => tool.sub_id === subFocus) : [];
  const tools = focused.length ? focused : action.tools;
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <span className="shrink-0 rounded-full bg-zinc-950 px-2.5 py-1 text-[11px] font-medium text-white">{action.agent}</span>
        <p className="truncate text-sm text-zinc-600" title={action.plan}>
          {action.plan}
        </p>
      </div>
      <div className="mt-3 flex items-stretch gap-1.5 overflow-x-auto pb-1">
        {tools.map((tool, index) => (
          <div key={`${tool.sub_id}-${tool.name}-${tool.call}`} className="flex items-stretch gap-1.5">
            {index > 0 && <span className="self-center font-mono text-zinc-300">→</span>}
            <div
              title={`${tool.purpose}\n${tool.call}`}
              className={`w-40 shrink-0 rounded-lg border bg-zinc-50 px-2.5 py-2 ${
                subFocus === tool.sub_id ? "border-zinc-900" : "border-zinc-200"
              }`}
            >
              <p className="font-mono text-[10px] text-zinc-400">{tool.sub_id}</p>
              <p className="truncate font-mono text-[12px] font-semibold text-zinc-950">{tool.name}</p>
              <p className="mt-1 line-clamp-3 text-[11px] leading-4 text-zinc-600">{tool.result}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function HandoffLog({ handoffs }: { handoffs: AgentHandoff[] }) {
  return (
    <ol className="relative ml-2 border-l border-zinc-200 pl-4">
      {handoffs.map((handoff) => {
        const closing = handoff.target === "Orchestrator";
        const askAgain = handoff.source === "Retrieval Agent" && handoff.target === "Query Agent";
        return (
          <li key={handoff.step} className="relative mb-4 last:mb-0">
            <span
              className={`absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full ring-4 ring-white ${
                askAgain ? "bg-amber-500" : closing ? "bg-zinc-950" : "bg-emerald-500"
              }`}
            />
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-semibold text-zinc-950">
                {handoff.source}
                <span className="mx-1.5 font-normal text-zinc-400">→</span>
                {handoff.target}
              </p>
              {handoff.tools.map((tool) => (
                <span key={tool} className="rounded-full bg-zinc-100 px-2 py-0.5 font-mono text-[10px] text-zinc-700">
                  {tool}
                </span>
              ))}
            </div>
            <p className="mt-1 line-clamp-2 text-sm leading-5 text-zinc-600" title={handoff.instruction}>
              {handoff.observation}
            </p>
          </li>
        );
      })}
    </ol>
  );
}

const COMPONENT_TITLES: Record<StageId, string> = {
  preprocess: "1. Data Preprocessing",
  storage: "2. Data Storage",
  transform: "3. User Query Transformation",
  retrieval: "4. Retrieval",
  generation: "5. Generation",
  agent: "6. Agent Control",
};

export function TraceInspector({ item, focus }: { item: AuditItemResult | null; focus: string }) {
  if (!item) {
    return (
      <section className="rounded-xl border border-dashed border-zinc-300 bg-white px-4 py-10 text-center text-sm text-zinc-500">
        Execute a trace to inspect inputs, outputs, and judge reasoning.
      </section>
    );
  }

  const stage = parentOf(focus);
  const subFocus = focus === stage ? null : focus;
  const action = item.agent.actions.find((row) => row.component === stage);
  const routeMatch = item.query_routing.expected_route === item.query_routing.actual_route;

  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-4 shadow-sm">
      <header className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold text-zinc-950">Deep trace inspector · {COMPONENT_TITLES[stage]}</h2>
        <p className="font-mono text-[11px] text-zinc-500">{item.query_id}</p>
      </header>
      {action && stage !== "agent" && <AgentFlow action={action} subFocus={subFocus} />}

      {stage === "preprocess" && (
        <div>
          <div className="mb-2 flex flex-wrap gap-2 font-mono text-[11px] text-zinc-500">
            <span className="rounded-full bg-zinc-100 px-2 py-1">Text parse</span>
            <span className="rounded-full bg-zinc-100 px-2 py-1">cos {score(item.retrieval.embedding_cosine_similarity)}</span>
            <span className="rounded-full bg-zinc-100 px-2 py-1">{item.retrieved_chunks.length} indexed</span>
          </div>
          <div className="grid gap-2 md:grid-cols-3">
            {item.gold_chunks.map((chunk) => (
              <article key={chunk.chunk_id} className="rounded-lg border border-zinc-200 border-l-emerald-400 border-l-4 bg-zinc-50 p-3">
                <p className="font-mono text-[11px] font-semibold text-zinc-950">{chunk.chunk_id}</p>
                <p className="mt-2 line-clamp-5 text-xs leading-5 text-zinc-700">{chunk.text}</p>
              </article>
            ))}
          </div>
        </div>
      )}

      {stage === "storage" && (
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-lg border border-zinc-200 p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Embeddings</p>
            <p className="mt-2 font-mono text-3xl font-semibold text-zinc-950">{item.retrieved_chunks.length}</p>
            <Bar label="Cosine to reference" value={item.retrieval.embedding_cosine_similarity} />
          </div>
          <div className="rounded-lg border border-zinc-200 p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Metadata</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {item.gold_chunks.map((chunk) => (
                <span key={chunk.chunk_id} className="rounded-full bg-zinc-100 px-2 py-1 font-mono text-[11px] text-zinc-800">
                  {chunk.chunk_id}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {stage === "transform" && (
        <div className="space-y-3">
          <div className="grid items-stretch gap-2 md:grid-cols-[1fr_auto_1fr]">
            <article className="rounded-lg border border-zinc-200 bg-zinc-50 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Asked</p>
              <p className="mt-2 text-sm leading-6 text-zinc-800">{item.user_query}</p>
            </article>
            <div className="hidden items-center justify-center font-mono text-zinc-300 md:flex">→</div>
            <article className="rounded-lg border border-zinc-200 bg-zinc-50 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Searched</p>
              <p className="mt-2 text-sm leading-6 text-zinc-800">{item.rewritten_query}</p>
            </article>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-full border border-zinc-200 px-2.5 py-1 font-mono text-[11px]">
              {item.query_modification.preservation_score}/5 kept
            </span>
            <span className="rounded-full border border-zinc-200 px-2.5 py-1 text-[11px] text-zinc-700">
              {formatRoute(item.query_routing.expected_route)}
            </span>
            <span className="font-mono text-zinc-300">→</span>
            <span
              className={`rounded-full px-2.5 py-1 text-[11px] ${
                routeMatch ? "bg-emerald-50 text-emerald-800" : "bg-rose-50 text-rose-800"
              }`}
            >
              {formatRoute(item.query_routing.actual_route)}
            </span>
          </div>
        </div>
      )}

      {stage === "retrieval" && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Bar label="Precision" value={item.retrieval.context_precision} />
            <Bar label="Recall" value={item.retrieval.context_recall} />
            <Bar label="F1" value={item.retrieval.f1_score} />
            <Bar label="Cosine" value={item.retrieval.embedding_cosine_similarity} />
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            <div className="space-y-2">
              {item.retrieved_chunks.map((chunk) => (
                <article
                  key={chunk.chunk_id}
                  className={`rounded-lg border border-l-4 bg-zinc-50 p-3 ${chunk.relevant ? "border-zinc-200 border-l-emerald-400" : "border-zinc-200 border-l-rose-400"}`}
                >
                  <p className="font-mono text-[11px] font-semibold text-zinc-950">
                    {chunk.chunk_id}
                    <span className="ml-2 font-normal text-zinc-500">{chunk.relevant ? "hit" : "miss"}</span>
                  </p>
                  <p className="mt-1 line-clamp-3 text-xs leading-5 text-zinc-700">{chunk.text}</p>
                </article>
              ))}
            </div>
            <div className="space-y-2">
              {item.gold_chunks.map((chunk) => {
                const retrieved = item.retrieved_chunks.some((row) => row.chunk_id === chunk.chunk_id);
                return (
                  <article
                    key={chunk.chunk_id}
                    className={`rounded-lg border border-l-4 bg-zinc-50 p-3 ${retrieved ? "border-zinc-200 border-l-emerald-400" : "border-zinc-200 border-l-amber-400"}`}
                  >
                    <p className="font-mono text-[11px] font-semibold text-zinc-950">
                      {chunk.chunk_id}
                      <span className="ml-2 font-normal text-zinc-500">{retrieved ? "in results" : "missing"}</span>
                    </p>
                    <p className="mt-1 line-clamp-3 text-xs leading-5 text-zinc-700">{chunk.text}</p>
                  </article>
                );
              })}
            </div>
          </div>
          <p className="font-mono text-[11px] text-zinc-500">
            Prompt {item.retrieval.prompt_assembly_valid ? "contains the question and every kept passage" : "is missing the question or a passage"}
          </p>
        </div>
      )}

      {stage === "generation" && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <Bar label="Faithfulness" value={item.response_generation.faithfulness_score} />
            <Bar label="Relevancy" value={item.response_generation.answer_relevancy_score} />
          </div>
          <div className="flex items-center gap-3 text-[11px] text-zinc-600">
            <span className="rounded-full bg-zinc-950 px-2 py-0.5 font-mono text-white">
              Tier {item.response_generation.ground_truth_alignment_tier} · {TIER_LABELS[item.response_generation.ground_truth_alignment_tier]}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm bg-emerald-300" /> grounded
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2 w-2 rounded-sm bg-amber-200" /> ungrounded
            </span>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            <article className="rounded-lg border border-zinc-200 bg-zinc-50 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Reference</p>
              <p className="mt-2 text-sm leading-6 text-zinc-800">{item.ground_truth_answer}</p>
            </article>
            <article className="rounded-lg border border-zinc-200 bg-zinc-50 p-3">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">Draft</p>
              <div className="mt-2">
                <GroundedAnswer spans={item.response_generation.grounding_spans} />
              </div>
            </article>
          </div>
        </div>
      )}

      {stage === "agent" && <HandoffLog handoffs={item.agent.handoffs} />}
    </section>
  );
}
