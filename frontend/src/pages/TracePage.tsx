import { FormEvent, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import type { AgentRun } from "../types";

export default function TracePage() {
  const params = useParams();
  const [userId, setUserId] = useState(localStorage.getItem("agent-user-id") ?? "U001");
  const [sessionId, setSessionId] = useState(params.sessionId ?? localStorage.getItem(`agent-session-${userId}`) ?? "");
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [error, setError] = useState("");

  useEffect(() => { if (sessionId) void load(); }, [params.sessionId]);

  async function load(event?: FormEvent) {
    event?.preventDefault();
    if (!sessionId) return;
    try { setRuns(await api.listRuns(userId, sessionId)); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "读取失败"); }
  }

  return (
    <div className="page">
      <header className="page-header"><div><p className="eyebrow">Auditability</p><h1>Agent Trace</h1><p>查看状态 checkpoint、路由决策、专家链和工具调用。</p></div></header>
      <form className="filter-bar" onSubmit={load}><input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="用户 ID" /><input className="wide" value={sessionId} onChange={(e) => setSessionId(e.target.value)} placeholder="Session ID" /><button className="primary">查询</button></form>
      {error && <div className="error-banner">{error}</div>}
      <div className="trace-list">
        {runs.map((run) => (
          <details key={run.id} className="trace-card">
            <summary><div><b>{run.id}</b><span>{new Date(run.created_at).toLocaleString()}</span></div><span className={`pill ${run.status}`}>{run.status}</span></summary>
            <div className="trace-section"><h4>Response</h4><p>{run.response_text ?? run.error_message ?? "尚未完成"}</p></div>
            <div className="trace-section"><h4>Route Trace</h4><pre>{JSON.stringify(run.route_trace_json, null, 2)}</pre></div>
            <div className="trace-section"><h4>Agent Trace</h4><pre>{JSON.stringify(run.trace_json, null, 2)}</pre></div>
            <div className="trace-section"><h4>Checkpoint</h4><pre>{JSON.stringify(run.checkpoint_json, null, 2)}</pre></div>
          </details>
        ))}
      </div>
    </div>
  );
}

