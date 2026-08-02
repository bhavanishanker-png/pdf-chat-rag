const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Source {
  source: string;
  page: string | number;
  chunk_index: string | number;
  text: string;
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  steps: string[];
}

export interface HealthResponse {
  status: string;
  chunk_count: number;
}

export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json();
}

export async function ingestPDF(
  file: File,
  chunkSize: number,
  chunkOverlap: number,
): Promise<{ message: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${BASE}/ingest?chunk_size=${chunkSize}&chunk_overlap=${chunkOverlap}`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Ingest failed");
  }
  return res.json();
}

export async function ask(query: string, topK: number): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "Ask failed");
  }
  return res.json();
}

export async function clearDB(): Promise<{ message: string }> {
  const res = await fetch(`${BASE}/clear`, { method: "POST" });
  if (!res.ok) throw new Error("Clear failed");
  return res.json();
}
