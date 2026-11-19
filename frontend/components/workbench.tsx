"use client";

import { Loader2, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ExperimentDiff } from "@/components/experiment-diff";
import { PipelineGraph } from "@/components/pipeline-graph";
import { TraceInspector } from "@/components/trace-inspector";
import { fetchDataset, fetchHealth, runAudit } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import { STAGE_ORDER, buildStages, parentOf, revealedTotals } from "@/lib/stages";
import type { AuditDatasetItem, AuditItemResult, BatchAuditReport, HealthResponse } from "@/lib/types";

type View = "trace" | "diff";

function caseLabel(queryId: string, question: string): string {
  const opening = question.split(/(?<=[.?!])\s+/)[0]?.trim() || question.trim();
  const preview = opening.length > 220 ? `${opening.slice(0, 217).trimEnd()}…` : opening;
  return `${queryId} · ${preview}`;
}

export function Workbench() {
  const [demoMode, setDemoMode] = useState(true);
  const [dataset, setDataset] = useState<AuditDatasetItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selection, setSelection] = useState("Q-101");
  const [customQuery, setCustomQuery] = useState("");
  const [topK, setTopK] = useState(3);
  const [promptVersion, setPromptVersion] = useState<"v1" | "v2">("v2");
  const [view, setView] = useState<View>("trace");
  const [item, setItem] = useState<AuditItemResult | null>(null);
  const [activeId, setActiveId] = useState("generation");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<BatchAuditReport | null>(null);
  const [candidate, setCandidate] = useState<BatchAuditReport | null>(null);
  const [diffLoading, setDiffLoading] = useState(false);
  const booted = useRef(false);
  const requestSeq = useRef(0);

  const execute = useCallback(async (request?: { queryId: string; custom: string | null }) => {
    const queryId = request?.queryId ?? selection;
    const custom = request ? request.custom : customQuery.trim() || null;
    const seq = requestSeq.current + 1;
    requestSeq.current = seq;
    setLoading(true);
    setError(null);
    setPlaying(false);
    setRevealed(0);
    try {
      const report = await runAudit({
        mock_mode: demoMode,
        query_id: custom ? null : queryId,
        custom_query: custom,
        top_k: topK,
        prompt_version: promptVersion,
      });
      if (seq !== requestSeq.current) return;
      setItem(report.items[0] ?? null);
      setActiveId("preprocess");
      setExpandedId(null);
      setRevealed(0);
      setPlaying(true);
    } catch (cause) {
      if (seq !== requestSeq.current) return;
      setError(cause instanceof Error ? cause.message : "The trace could not be executed.");
    } finally {
      if (seq === requestSeq.current) setLoading(false);
    }
  }, [customQuery, demoMode, promptVersion, selection, topK]);

  useEffect(() => {
    void fetchDataset().then(setDataset).catch((cause: unknown) => {
      setError(cause instanceof Error ? cause.message : "The dataset could not be loaded.");
    });
    void fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (!dataset.length || booted.current) return;
    booted.current = true;
    void execute({ queryId: "Q-101", custom: null });
  }, [dataset, execute]);

  useEffect(() => {
    if (!playing) return;
    if (revealed >= STAGE_ORDER.length) {
      setPlaying(false);
      setActiveId("agent");
      return;
    }
    setActiveId(STAGE_ORDER[revealed]);
    const timer = window.setTimeout(() => setRevealed((count) => count + 1), 1400);
    return () => window.clearTimeout(timer);
  }, [playing, revealed]);

  useEffect(() => {
    if (view !== "diff" || baseline) return;
    setDiffLoading(true);
    void Promise.all([
      runAudit({ mock_mode: true, experiment: "baseline" }),
      runAudit({ mock_mode: true, experiment: "candidate" }),
    ])
      .then(([v1, v2]) => {
        setBaseline(v1);
        setCandidate(v2);
      })
      .catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : "The experiment diff could not be loaded.");
      })
      .finally(() => setDiffLoading(false));
  }, [baseline, view]);

  const stages = useMemo(() => buildStages(item, revealed), [item, revealed]);
  const totals = revealedTotals(stages);
  const liveBlocked = !demoMode && health !== null && !health.llm_configured;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-zinc-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4 px-4 py-3 lg:px-6">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-400">Assurance workbench</p>
            <h1 className="text-base font-semibold tracking-tight">Component-Level RAG & Agentic Assurance Analyzer</h1>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={!demoMode}
            onClick={() => setDemoMode((current) => !current)}
            className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-xs font-medium"
          >
            <span className={demoMode ? "text-zinc-950" : "text-zinc-400"}>Demo</span>
            <span className={`relative h-5 w-9 rounded-full ${demoMode ? "bg-emerald-600" : "bg-zinc-900"}`}>
              <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${demoMode ? "left-0.5" : "left-4"}`} />
            </span>
            <span className={demoMode ? "text-zinc-400" : "text-zinc-950"}>Live LLM</span>
          </button>
        </div>
      </header>

      <main className="mx-auto flex max-w-[1440px] flex-col gap-3 px-4 py-4 lg:px-6">
        <section className="rounded-xl border border-zinc-200 bg-white p-3">
          <div className="grid gap-3 lg:grid-cols-[minmax(220px,0.9fr)_minmax(0,1.2fr)_150px_170px_auto] lg:items-end">
            <label className="block text-xs font-medium text-zinc-500">
              Test case
              <select
                value={selection}
                onChange={(event) => {
                  const queryId = event.target.value;
                  setSelection(queryId);
                  setView("trace");
                  void execute({ queryId, custom: null });
                }}
                className="mt-1 w-full rounded-lg border border-zinc-200 bg-white px-2 py-2 text-sm text-zinc-950 outline-none focus:ring-2 focus:ring-zinc-400"
              >
                {dataset.map((row) => (
                  <option key={row.query_id} value={row.query_id} title={row.user_query}>
                    {caseLabel(row.query_id, row.user_query)}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs font-medium text-zinc-500">
              Custom question
              <input
                value={customQuery}
                onChange={(event) => setCustomQuery(event.target.value)}
                className="mt-1 w-full rounded-lg border border-zinc-200 bg-white px-2 py-2 text-sm text-zinc-950 outline-none focus:ring-2 focus:ring-zinc-400"
                placeholder="Type an ad-hoc financial question"
              />
            </label>
            <label className="block text-xs font-medium text-zinc-500">
              Retrieval Top-K
              <input
                type="range"
                min={1}
                max={3}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="mt-3 w-full accent-zinc-900"
              />
              <span className="mt-1 block font-mono text-[11px] text-zinc-700">k = {topK}</span>
            </label>
            <div className="text-xs font-medium text-zinc-500">
              Grounding prompt
              <div className="mt-1 flex rounded-lg bg-zinc-100 p-1">
                {(["v1", "v2"] as const).map((version) => (
                  <button
                    key={version}
                    type="button"
                    onClick={() => setPromptVersion(version)}
                    className={`flex-1 rounded-md px-2 py-1.5 font-mono text-xs ${
                      promptVersion === version ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-500"
                    }`}
                  >
                    {version === "v1" ? "v1 loose" : "v2 strict"}
                  </button>
                ))}
              </div>
            </div>
            <button
              type="button"
              onClick={() =>
                void execute({
                  queryId: selection,
                  custom: customQuery.trim() ? customQuery.trim() : null,
                })
              }
              disabled={loading}
              className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-zinc-950 px-4 text-sm font-semibold text-white hover:bg-zinc-800 disabled:cursor-wait disabled:opacity-70"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Execute Live Trace
            </button>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3 font-mono text-[11px] text-zinc-500">
            <span>{formatDuration(totals.latencyMs)}</span>
            <span>{totals.tokens.toLocaleString()} tok</span>
            <span>${totals.costUsd.toFixed(4)}</span>
            <span className="text-zinc-400">{demoMode ? "offline mock" : "live judge"}</span>
          </div>
        </section>

        {error && <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">{error}</div>}
        {liveBlocked && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
            Live LLM Judge needs OPENAI_API_KEY in backend/.env. Demo mode still executes a full trace.
          </div>
        )}

        <div className="flex gap-1">
          {(
            [
              ["trace", "Trace playground"],
              ["diff", "Experiment Diff (v1 vs v2)"],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => setView(id)}
              className={`rounded-md px-3 py-1.5 text-xs font-semibold ${
                view === id ? "bg-zinc-950 text-white" : "text-zinc-500 hover:bg-zinc-100"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {view === "trace" ? (
          <>
            <PipelineGraph
              stages={stages}
              activeId={activeId}
              expandedId={expandedId}
              onSelect={(id) => {
                const parent = parentOf(id);
                setActiveId(id);
                const parentStage = stages.find((stage) => stage.id === parent);
                if (!parentStage?.subs.length) {
                  setExpandedId(null);
                  return;
                }
                setExpandedId((current) => (id === parent && current === parent ? null : parent));
              }}
            />
            <TraceInspector item={revealed > 0 ? item : null} focus={activeId} />
          </>
        ) : (
          <ExperimentDiff baseline={baseline} candidate={candidate} loading={diffLoading} />
        )}
      </main>
    </div>
  );
}
