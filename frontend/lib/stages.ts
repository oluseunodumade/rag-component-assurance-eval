import type { AuditItemResult, StageTiming } from "@/lib/types";

export type StageId = "preprocess" | "storage" | "transform" | "retrieval" | "generation" | "agent";

export type StageStatus = "pending" | "running" | "pass" | "fail" | "warn";

export type SubView = {
  id: string;
  title: string;
  accelerator: string;
  status: StageStatus;
  score: string;
};

export type StageView = {
  id: StageId;
  index: string;
  title: string;
  status: StageStatus;
  latencyMs: number | null;
  tokens: number | null;
  costUsd: number | null;
  score: string;
  agentName: string;
  subs: SubView[];
  loopTarget: StageId | null;
};

type SubDef = { id: string; title: string; accelerator: string };

const TAXONOMY: { id: StageId; index: string; title: string; agentName: string; clocks: string[]; subs: SubDef[] }[] = [
  {
    id: "preprocess",
    index: "1",
    title: "Data Preprocessing",
    agentName: "Corpus Agent",
    clocks: ["input"],
    subs: [
      { id: "1.1", title: "Identify Relevant Document", accelerator: "Relevance and Sufficiency Analyzer" },
      { id: "1.2", title: "Parse images, text, and tables", accelerator: "Document Parsing Inspector" },
      { id: "1.3", title: "Chunk Documents", accelerator: "Document Chunking Inspector" },
      { id: "1.4", title: "Create Embeddings", accelerator: "Document Embedding Inspector" },
      { id: "1.5", title: "Create Index", accelerator: "Document Indexing Inspector" },
    ],
  },
  {
    id: "storage",
    index: "2",
    title: "Data Storage",
    agentName: "Storage Agent",
    clocks: [],
    subs: [
      { id: "2.1", title: "Store Embeddings", accelerator: "Embedding Storage Inspector" },
      { id: "2.2", title: "Store Metadata", accelerator: "None required" },
    ],
  },
  {
    id: "transform",
    index: "3",
    title: "User Query Transformation",
    agentName: "Query Agent",
    clocks: ["rewriter"],
    subs: [
      { id: "3.1", title: "Receive User Query", accelerator: "None required" },
      { id: "3.2", title: "Modify Query", accelerator: "Query Modification Analyzer" },
      { id: "3.3", title: "Routing", accelerator: "Query Routing Analyzer" },
      { id: "3.4", title: "Embed Query", accelerator: "Query Embedding Analyzer" },
    ],
  },
  {
    id: "retrieval",
    index: "4",
    title: "Retrieval",
    agentName: "Retrieval Agent",
    clocks: ["retriever", "assembler"],
    subs: [
      { id: "4.1", title: "Retrieve, Process, and Combine Chunks", accelerator: "Retrieval Quality Analyzer" },
      { id: "4.2", title: "Update Query with Retrieved Information", accelerator: "Query Quality Analyzer" },
    ],
  },
  {
    id: "generation",
    index: "5",
    title: "Generation",
    agentName: "Generation Agent",
    clocks: ["judge"],
    subs: [{ id: "5.1", title: "Generate Response", accelerator: "Response Quality Analyzer" }],
  },
  {
    id: "agent",
    index: "Supervisor",
    title: "Agent Control",
    agentName: "Orchestrator",
    clocks: ["agent"],
    subs: [],
  },
];

export const STAGE_ORDER: StageId[] = TAXONOMY.filter((component) => component.id !== "agent").map((component) => component.id);

export function focusDetail(id: string): { id: string; title: string; accelerator: string } | null {
  for (const component of TAXONOMY) {
    const sub = component.subs.find((row) => row.id === id);
    if (sub) return sub;
  }
  return null;
}

export function parentOf(id: string): StageId {
  const match = TAXONOMY.find((component) => component.id === id || component.subs.some((sub) => sub.id === id));
  return match?.id ?? "generation";
}

function timingFor(timings: StageTiming[] | undefined, stage: string): StageTiming | undefined {
  return timings?.find((row) => row.stage === stage);
}

function retrieverPasses(item: AuditItemResult): boolean {
  const retrieval = item.retrieval;
  return retrieval.context_precision >= 0.5 && retrieval.context_recall >= 0.5 && retrieval.f1_score >= 0.5;
}

function componentScore(item: AuditItemResult, id: StageId): string {
  if (id === "preprocess") return `${item.gold_chunks.length} chunks`;
  if (id === "storage") return item.retrieved_chunks.length ? "Stored" : "Empty";
  if (id === "transform") {
    return item.query_routing.passed ? `${item.query_modification.preservation_score}/5` : "Misroute";
  }
  if (id === "retrieval") return `F1 ${item.retrieval.f1_score.toFixed(2)}`;
  if (id === "generation") return `Faith ${item.response_generation.faithfulness_score.toFixed(2)}`;
  return item.agent.retried ? `${item.agent.hops} hops` : item.agent.critic_verdict === "accept" ? "Accept" : "Revise";
}

function worst(statuses: StageStatus[]): StageStatus {
  if (statuses.includes("fail")) return "fail";
  if (statuses.includes("warn")) return "warn";
  if (statuses.includes("running")) return "running";
  if (statuses.includes("pending")) return "pending";
  return "pass";
}

function subScore(item: AuditItemResult, id: string): { status: StageStatus; score: string } {
  const retrieval = item.retrieval;
  const response = item.response_generation;
  if (id === "1.1") return { status: item.gold_chunks.length ? "pass" : "fail", score: `${item.gold_chunks.length} docs` };
  if (id === "1.2") return { status: "pass", score: "Text" };
  if (id === "1.3") return { status: item.gold_chunks.length ? "pass" : "fail", score: `${item.gold_chunks.length} chunks` };
  if (id === "1.4") return { status: "pass", score: `cos ${retrieval.embedding_cosine_similarity.toFixed(2)}` };
  if (id === "1.5") return { status: item.retrieved_chunks.length ? "pass" : "fail", score: `${item.retrieved_chunks.length} hits` };
  if (id === "2.1") return { status: item.retrieved_chunks.length ? "pass" : "fail", score: "Stored" };
  if (id === "2.2") return { status: "pass", score: "None required" };
  if (id === "3.1") return { status: item.user_query.trim() ? "pass" : "fail", score: "Received" };
  if (id === "3.2") {
    return {
      status: item.query_modification.passed ? "pass" : "fail",
      score: `${item.query_modification.preservation_score}/5`,
    };
  }
  if (id === "3.3") return { status: item.query_routing.passed ? "pass" : "fail", score: item.query_routing.passed ? "Match" : "Misroute" };
  if (id === "3.4") return { status: "pass", score: "Embedded" };
  if (id === "4.1") return { status: retrieverPasses(item) ? "pass" : "fail", score: `F1 ${retrieval.f1_score.toFixed(2)}` };
  if (id === "4.2") return { status: retrieval.prompt_assembly_valid ? "pass" : "fail", score: retrieval.prompt_assembly_valid ? "Intact" : "Broken" };
  if (id === "5.1") {
    if (response.ground_truth_alignment_tier === 2 && response.passed) return { status: "warn", score: `Faith ${response.faithfulness_score.toFixed(2)}` };
    return { status: response.passed ? "pass" : "fail", score: `Faith ${response.faithfulness_score.toFixed(2)}` };
  }
  const agent = item.agent;
  if (id === "6.1") return { status: item.query_modification.passed ? "pass" : "fail", score: `${agent.plan.length} steps` };
  if (id === "6.2") {
    const tool = agent.tool_calls[0]?.tool ?? "";
    return { status: item.query_routing.passed ? "pass" : "fail", score: tool.includes("assurance") ? "Assurance" : "Reporting" };
  }
  if (id === "6.3") return { status: agent.retried ? "warn" : "pass", score: `${agent.hops} hop${agent.hops === 1 ? "" : "s"}` };
  return { status: agent.critic_verdict === "accept" ? "pass" : "fail", score: agent.critic_verdict === "accept" ? "Accept" : "Revise" };
}

export function buildStages(item: AuditItemResult | null, revealed: number): StageView[] {
  const workerCount = STAGE_ORDER.length;
  return TAXONOMY.map((component) => {
    const workerIndex = STAGE_ORDER.indexOf(component.id);
    const supervisor = component.id === "agent";
    const visible = supervisor ? Boolean(item) : Boolean(item) && workerIndex < revealed;
    const clocks = component.clocks
      .map((name) => timingFor(item?.timings, name))
      .filter((row): row is StageTiming => Boolean(row));
    const subs: SubView[] = component.subs.map((sub) => {
      if (!visible || !item) return { ...sub, status: workerIndex === revealed && item ? "running" : "pending", score: "—" };
      const scored = subScore(item, sub.id);
      return { ...sub, status: scored.status, score: scored.score };
    });
    let status: StageStatus = "pending";
    if (supervisor && item) status = revealed < workerCount ? "running" : item.agent.passed ? "pass" : "fail";
    else if (item && workerIndex === revealed) status = "running";
    else if (visible) status = worst(subs.map((sub) => sub.status));
    const done = supervisor && Boolean(item) && revealed >= workerCount;
    return {
      id: component.id,
      index: component.index,
      title: component.title,
      status,
      latencyMs: visible && clocks.length ? clocks.reduce((sum, row) => sum + row.latency_ms, 0) : null,
      tokens: visible && clocks.length ? clocks.reduce((sum, row) => sum + row.tokens, 0) : null,
      costUsd: visible && clocks.length ? clocks.reduce((sum, row) => sum + row.cost_usd, 0) : null,
      score: supervisor
        ? done
          ? item?.agent.critic_verdict === "accept"
            ? "Accept"
            : "Revise"
          : visible
            ? "Dispatching"
            : "—"
        : visible && item
          ? componentScore(item, component.id)
          : "—",
      agentName: component.agentName,
      subs,
      loopTarget: visible && item && component.id === "retrieval" && item.agent.retried ? "agent" : null,
    };
  });
}

export function revealedTotals(stages: StageView[]): { latencyMs: number; tokens: number; costUsd: number } {
  return stages.reduce(
    (totals, stage) => ({
      latencyMs: totals.latencyMs + (stage.latencyMs ?? 0),
      tokens: totals.tokens + (stage.tokens ?? 0),
      costUsd: totals.costUsd + (stage.costUsd ?? 0),
    }),
    { latencyMs: 0, tokens: 0, costUsd: 0 },
  );
}
