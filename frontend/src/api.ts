import type { AgentRun, Handoff, Message, PendingAction, RAGIndex, Session } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export function authHeaders(userId: string, staff = false): HeadersInit {
  return {
    "Content-Type": "application/json",
    "X-User-ID": userId,
    "X-Scopes": staff
      ? "order:read,shipment:read,return:create,ticket:create,policy:read,handoff:manage,rag:manage"
      : "order:read,shipment:read,return:create,ticket:create,policy:read"
  };
}

async function request<T>(path: string, userId: string, init: RequestInit = {}, staff = false): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(userId, staff), ...(init.headers ?? {}) }
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  createSession: (userId: string) =>
    request<Session>("/sessions", userId, { method: "POST", body: JSON.stringify({ title: "售后咨询" }) }),
  listSessions: (userId: string) => request<Session[]>("/sessions", userId),
  getSession: (userId: string, sessionId: string) => request<Session>(`/sessions/${sessionId}`, userId),
  listMessages: (userId: string, sessionId: string) =>
    request<Message[]>(`/sessions/${sessionId}/messages`, userId),
  sendMessage: (userId: string, sessionId: string, content: string) =>
    request<{ run_id: string; session_id: string; status: string }>(`/sessions/${sessionId}/messages`, userId, {
      method: "POST",
      body: JSON.stringify({ content, client_message_id: crypto.randomUUID() })
    }),
  getPendingAction: (userId: string, sessionId: string) =>
    request<PendingAction | null>(`/sessions/${sessionId}/pending-action`, userId),
  confirmAction: (userId: string, action: PendingAction) =>
    request(`/pending-actions/${action.id}/confirm`, userId, {
      method: "POST",
      body: JSON.stringify({
        confirmation_token: action.confirmation_token,
        state_version: action.state_version
      })
    }),
  cancelAction: (userId: string, action: PendingAction) =>
    request(`/pending-actions/${action.id}/cancel`, userId, {
      method: "POST",
      body: JSON.stringify({ state_version: action.state_version })
    }),
  listRuns: (userId: string, sessionId: string) =>
    request<AgentRun[]>(`/sessions/${sessionId}/runs`, userId),
  feedback: (userId: string, runId: string, helpful: boolean) =>
    request(`/runs/${runId}/feedback`, userId, {
      method: "POST",
      body: JSON.stringify({ helpful, rating: helpful ? 5 : 2 })
    }),
  listHandoffs: async (staffId: string) => {
    const [open, claimed] = await Promise.all([
      request<Handoff[]>("/handoffs?status=open", staffId, {}, true),
      request<Handoff[]>("/handoffs?status=claimed", staffId, {}, true)
    ]);
    return [...open, ...claimed];
  },
  claimHandoff: (staffId: string, id: string) =>
    request<Handoff>(`/handoffs/${id}/claim`, staffId, {
      method: "POST",
      body: JSON.stringify({ assignee: staffId })
    }, true),
  resolveHandoff: (staffId: string, id: string, resolution: string) =>
    request<Handoff>(`/handoffs/${id}/resolve`, staffId, {
      method: "POST",
      body: JSON.stringify({ resolution })
    }, true),
  listIndexes: (staffId: string) => request<RAGIndex[]>("/rag/indexes", staffId, {}, true),
  publishIndex: (staffId: string, version: string) =>
    request<RAGIndex>(`/rag/indexes/${encodeURIComponent(version)}/publish`, staffId, { method: "POST" }, true),
  rollbackIndex: (staffId: string, version: string) =>
    request<RAGIndex>(`/rag/indexes/${encodeURIComponent(version)}/rollback`, staffId, { method: "POST" }, true),
  metrics: async () => {
    const response = await fetch(`${API_BASE}/metrics`);
    if (!response.ok) throw new Error("无法读取指标");
    return response.text();
  },
  eventUrl: (sessionId: string, userId: string, after = 0) =>
    `${API_BASE}/sessions/${sessionId}/events?user_id=${encodeURIComponent(userId)}&after_sequence=${after}`
};

