export type AuditCategory = "Financial_Reporting" | "Audit_Assurance";

export type ContextChunk = {
  chunk_id: string;
  text: string;
  source: string;
};

export type RetrievedChunk = ContextChunk & {
  score: number;
  relevant: boolean;
};

export type AuditDatasetItem = {
  query_id: string;
  category: AuditCategory;
  scenario: string;
  user_query: string;
  rewritten_query: string;
  expected_route: string;
  actual_route: string;
  routing_confidence: number;
  retrieved_chunks: RetrievedChunk[];
  gold_chunks: ContextChunk[];
  assembled_prompt: string;
  generated_response: string;
  ground_truth_answer: string;
};

export type QueryModificationResult = {
  preservation_score: number;
  ambiguity_level: "low" | "medium" | "high";
  passed: boolean;
  rationale: string;
};

export type QueryRoutingResult = {
  expected_route: string;
  actual_route: string;
  routing_confidence: number;
  passed: boolean;
  rationale: string;
};

export type RetrievalResult = {
  context_precision: number;
  context_recall: number;
  f1_score: number;
  embedding_cosine_similarity: number;
  prompt_assembly_valid: boolean;
  passed: boolean;
  rationale: string;
};

export type GroundingSpan = {
  text: string;
  label: "grounded" | "ungrounded";
  reason: string;
};

export type StageTiming = {
  stage: string;
  latency_ms: number;
  tokens: number;
  cost_usd: number;
};

export type AgentToolCall = {
  hop: number;
  tool: string;
  query: string;
  chunk_ids: string[];
  sufficient: boolean;
  observation: string;
};

export type AgentToolUse = {
  name: string;
  sub_id: string;
  purpose: string;
  call: string;
  result: string;
};

export type AgentAction = {
  component: "preprocess" | "storage" | "transform" | "retrieval" | "generation" | "agent";
  agent: string;
  plan: string;
  tools: AgentToolUse[];
};

export type AgentHandoff = {
  step: number;
  source: string;
  target: string;
  component: "preprocess" | "storage" | "transform" | "retrieval" | "generation" | "agent";
  instruction: string;
  tools: string[];
  observation: string;
};

export type AgentResult = {
  plan: string[];
  tool_calls: AgentToolCall[];
  hops: number;
  retried: boolean;
  critic_verdict: "accept" | "revise";
  stop_reason: string;
  passed: boolean;
  rationale: string;
  actions: AgentAction[];
  handoffs: AgentHandoff[];
};

export type ResponseGenerationResult = {
  faithfulness_score: number;
  answer_relevancy_score: number;
  ground_truth_alignment_tier: number;
  passed: boolean;
  judge_rationale: string;
  grounding_spans: GroundingSpan[];
};

export type AuditItemResult = {
  query_id: string;
  category: AuditCategory;
  scenario: string;
  user_query: string;
  rewritten_query: string;
  retrieved_chunks: RetrievedChunk[];
  gold_chunks: ContextChunk[];
  assembled_prompt: string;
  generated_response: string;
  ground_truth_answer: string;
  query_modification: QueryModificationResult;
  query_routing: QueryRoutingResult;
  retrieval: RetrievalResult;
  response_generation: ResponseGenerationResult;
  agent: AgentResult;
  passed: boolean;
  timings: StageTiming[];
};

export type BatchAuditReport = {
  total_items: number;
  overall_pass_rate: number;
  routing_accuracy: number;
  avg_retrieval_f1: number;
  avg_faithfulness: number;
  ungrounded_extras_rate: number;
  items: AuditItemResult[];
};

export type HealthResponse = {
  status: "ok";
  llm_configured: boolean;
  dataset_size: number;
  mock_mode_default: boolean;
};

export const TIER_LABELS: Record<number, string> = {
  3: "Complete",
  2: "Partial",
  1: "Complete + Extras",
  0: "Hallucinated",
};

export const TIER_DETAIL: Record<number, string> = {
  3: "Complete alignment. The response covers the reference answer and does not add unsupported claims.",
  2: "Partial alignment. The response stays grounded but leaves out part of the reference answer.",
  1: "Complete alignment with ungrounded extras. The reference answer is covered, and the response adds claims that are not in the evidence.",
  0: "Partial alignment with hallucinated extras. Coverage is incomplete and the response adds unsupported claims.",
};
