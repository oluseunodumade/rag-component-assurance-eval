import { percent, score } from "@/lib/format";
import type { BatchAuditReport } from "@/lib/types";

type Row = {
  metric: string;
  baseline: string;
  candidate: string;
  delta: string;
  improved: boolean;
};

function rows(baseline: BatchAuditReport, candidate: BatchAuditReport): Row[] {
  const routingDelta = candidate.routing_accuracy - baseline.routing_accuracy;
  const f1Delta = candidate.avg_retrieval_f1 - baseline.avg_retrieval_f1;
  const extrasDelta = candidate.ungrounded_extras_rate - baseline.ungrounded_extras_rate;
  const passDelta = candidate.overall_pass_rate - baseline.overall_pass_rate;
  const faithDelta = candidate.avg_faithfulness - baseline.avg_faithfulness;
  return [
    {
      metric: "Routing accuracy",
      baseline: percent(baseline.routing_accuracy),
      candidate: percent(candidate.routing_accuracy),
      delta: signedPercent(routingDelta),
      improved: routingDelta > 0,
    },
    {
      metric: "Retrieval F1",
      baseline: score(baseline.avg_retrieval_f1),
      candidate: score(candidate.avg_retrieval_f1),
      delta: `${f1Delta >= 0 ? "+" : ""}${f1Delta.toFixed(2)}`,
      improved: f1Delta > 0,
    },
    {
      metric: "Ungrounded extras",
      baseline: percent(baseline.ungrounded_extras_rate),
      candidate: percent(candidate.ungrounded_extras_rate),
      delta: signedPercent(extrasDelta),
      improved: extrasDelta < 0,
    },
    {
      metric: "Overall pass rate",
      baseline: percent(baseline.overall_pass_rate),
      candidate: percent(candidate.overall_pass_rate),
      delta: signedPercent(passDelta),
      improved: passDelta > 0,
    },
    {
      metric: "Avg faithfulness",
      baseline: score(baseline.avg_faithfulness),
      candidate: score(candidate.avg_faithfulness),
      delta: `${faithDelta >= 0 ? "+" : ""}${faithDelta.toFixed(2)}`,
      improved: faithDelta > 0,
    },
  ];
}

function signedPercent(delta: number): string {
  const points = delta * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(0)}%`;
}

export function ExperimentDiff({
  baseline,
  candidate,
  loading,
}: {
  baseline: BatchAuditReport | null;
  candidate: BatchAuditReport | null;
  loading: boolean;
}) {
  if (loading || !baseline || !candidate) {
    return (
      <section className="rounded-xl border border-zinc-200 bg-white px-4 py-12 text-center text-sm text-zinc-500">
        Comparing v1 baseline and v2 candidate across the benchmark suite…
      </section>
    );
  }
  const table = rows(baseline, candidate);
  return (
    <section className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
      <header className="border-b border-zinc-200 px-4 py-3">
        <h2 className="text-sm font-semibold">Experiment diff · v1 baseline vs v2 candidate</h2>
        <p className="mt-1 text-xs text-zinc-500">
          v2 keeps the strict grounding prompt and the full chunk set. v1 keeps a single chunk on two queries, misroutes one case, and adds an ungrounded forecast.
        </p>
      </header>
      <table className="min-w-full text-sm">
        <thead className="bg-zinc-50 font-mono text-[10px] uppercase tracking-[0.14em] text-zinc-500">
          <tr>
            <th className="px-4 py-2 text-left font-medium">Metric</th>
            <th className="px-4 py-2 text-right font-medium">v1 Baseline</th>
            <th className="px-4 py-2 text-right font-medium">v2 Candidate</th>
            <th className="px-4 py-2 text-right font-medium">Delta</th>
          </tr>
        </thead>
        <tbody>
          {table.map((row) => (
            <tr key={row.metric} className="border-t border-zinc-100">
              <td className="px-4 py-3 font-medium text-zinc-900">{row.metric}</td>
              <td className="px-4 py-3 text-right font-mono text-zinc-600">{row.baseline}</td>
              <td className="px-4 py-3 text-right font-mono text-zinc-950">{row.candidate}</td>
              <td className={`px-4 py-3 text-right font-mono font-semibold ${row.improved ? "text-emerald-700" : "text-rose-700"}`}>
                {row.delta}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
