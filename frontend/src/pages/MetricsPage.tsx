import { useEffect, useState } from "react";
import { api } from "../api";

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [error, setError] = useState("");
  useEffect(() => { void refresh(); const timer = setInterval(() => void refresh(), 5000); return () => clearInterval(timer); }, []);
  async function refresh() { try { setMetrics(parse(await api.metrics())); setError(""); } catch (err) { setError(err instanceof Error ? err.message : "读取失败"); } }
  const cards = [
    ["已完成 Agent Runs", find(metrics, "agent_runs_completed_total")],
    ["失败 Agent Runs", find(metrics, "agent_runs_failed_total")],
    ["高风险确认", find(metrics, "pending_actions_created_total")],
    ["人工升级", find(metrics, "handoffs_created_total")],
    ["用户反馈", find(metrics, "feedback_total")],
    ["有帮助反馈", find(metrics, "feedback_helpful_total")]
  ];
  return <div className="page"><header className="page-header"><div><p className="eyebrow">Observability</p><h1>基础指标面板</h1><p>每 5 秒刷新 FastAPI 暴露的 Prometheus 指标。</p></div><button className="secondary" onClick={() => void refresh()}>刷新</button></header>{error && <div className="error-banner">{error}</div>}<div className="metric-grid">{cards.map(([label, value]) => <article className="metric-card" key={String(label)}><span>{label}</span><strong>{Number(value).toLocaleString()}</strong></article>)}</div><section className="raw-metrics"><h3>全部 Metrics</h3><pre>{JSON.stringify(metrics, null, 2)}</pre></section></div>;
}

function parse(text: string): Record<string, number> { const output: Record<string, number> = {}; for (const line of text.split("\n")) { if (!line || line.startsWith("#")) continue; const [key, value] = line.trim().split(/\s+/); if (key && value) output[key] = Number(value); } return output; }
function find(values: Record<string, number>, suffix: string) { const key = Object.keys(values).find((item) => item.endsWith(suffix)); return key ? values[key] : 0; }

