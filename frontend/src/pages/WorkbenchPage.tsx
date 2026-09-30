import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import type { Handoff } from "../types";

export default function WorkbenchPage() {
  const [staffId, setStaffId] = useState(localStorage.getItem("agent-staff-id") ?? "CS001");
  const [handoffs, setHandoffs] = useState<Handoff[]>([]);
  const [resolution, setResolution] = useState<Record<string, string>>({});
  const [error, setError] = useState("");

  useEffect(() => { void refresh(); }, []);

  async function refresh() {
    try { setHandoffs(await api.listHandoffs(staffId)); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "加载失败"); }
  }

  async function claim(item: Handoff) {
    await api.claimHandoff(staffId, item.id);
    await refresh();
  }

  async function resolve(item: Handoff) {
    const text = resolution[item.id]?.trim();
    if (!text) return;
    await api.resolveHandoff(staffId, item.id, text);
    await refresh();
  }

  return (
    <div className="page">
      <header className="page-header">
        <div><p className="eyebrow">Human operations</p><h1>人工客服工作台</h1><p>接管 Agent 无法安全完成的售后会话。</p></div>
        <div className="header-actions"><label>客服 ID <input value={staffId} onChange={(e) => { setStaffId(e.target.value); localStorage.setItem("agent-staff-id", e.target.value); }} /></label><button className="secondary" onClick={() => void refresh()}>刷新</button></div>
      </header>
      {error && <div className="error-banner">{error}</div>}
      <div className="stats-strip"><div><strong>{handoffs.filter((x) => x.status === "open").length}</strong><span>待领取</span></div><div><strong>{handoffs.filter((x) => x.status === "claimed").length}</strong><span>处理中</span></div><div><strong>{handoffs.filter((x) => x.priority === "high").length}</strong><span>高优先级</span></div></div>
      <div className="card-grid">
        {handoffs.map((item) => (
          <article className="ops-card" key={item.id}>
            <div className="card-top"><span className={`pill ${item.priority}`}>{item.priority}</span><span className={`pill ${item.status}`}>{item.status}</span></div>
            <h3>{item.ticket_id ?? "未生成业务工单号"}</h3>
            <p>{item.reason}</p>
            <dl className="compact-dl"><div><dt>会话</dt><dd><code>{item.session_id}</code></dd></div><div><dt>负责人</dt><dd>{item.assigned_to ?? "未领取"}</dd></div><div><dt>创建时间</dt><dd>{new Date(item.created_at).toLocaleString()}</dd></div></dl>
            <div className="action-row">
              <Link className="text-link" to={`/traces/${item.session_id}`}>查看 Agent Trace</Link>
              {item.status === "open" && <button className="primary" onClick={() => void claim(item)}>领取</button>}
            </div>
            {item.status === "claimed" && (
              <div className="resolution-box"><textarea placeholder="填写处理结论和正确政策…" value={resolution[item.id] ?? ""} onChange={(e) => setResolution((old) => ({ ...old, [item.id]: e.target.value }))} /><button className="primary" onClick={() => void resolve(item)}>解决并关闭</button></div>
            )}
          </article>
        ))}
        {!handoffs.length && <div className="empty-state card-span"><h3>当前没有待处理会话</h3><p>Agent 升级或工具未知提交会出现在这里。</p></div>}
      </div>
    </div>
  );
}

