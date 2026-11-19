import type { AuditDatasetItem, BatchAuditReport, HealthResponse } from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) return body.detail;
  } catch {
    // The body was not JSON.
  }
  return `Request failed (${response.status})`;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<HealthResponse>;
}

export async function fetchDataset(): Promise<AuditDatasetItem[]> {
  const response = await fetch(`${API_BASE}/api/dataset`, { cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<AuditDatasetItem[]>;
}

export type EvaluatePayload = {
  mock_mode: boolean;
  custom_query?: string | null;
  query_id?: string | null;
  top_k?: number;
  prompt_version?: "v1" | "v2";
  experiment?: "baseline" | "candidate" | null;
};

export async function runAudit(payload: EvaluatePayload): Promise<BatchAuditReport> {
  const response = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await readError(response));
  return response.json() as Promise<BatchAuditReport>;
}
