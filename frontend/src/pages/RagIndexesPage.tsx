import { useEffect, useState } from "react";
import { api } from "../api";
import type { RAGIndex } from "../types";

export default function RagIndexesPage() {
  const [staffId] = useState(localStorage.getItem("agent-staff-id") ?? "CS001");
  const [indexes, setIndexes] = useState<RAGIndex[]>([]);
  const [error, setError] = useState("");
  useEffect(() => { void refresh(); }, []);
  async function refresh() { try { setIndexes(await api.listIndexes(staffId)); setError(""); } catch (err) { setError(err instanceof Error ? err.message : "读取失败"); } }
  async function action(version: string, rollback = false) { rollback ? await api.rollbackIndex(staffId, version) : await api.publishIndex(staffId, version); await refresh(); }
  return (
    <div className="page">
      <header className="page-header"><div><p className="eyebrow">Knowledge operations</p><h1>RAG 索引版本</h1><p>不可变索引、CURRENT 发布别名和回滚。</p></div><button className="secondary" onClick={() => void refresh()}>刷新</button></header>
      {error && <div className="error-banner">{error}</div>}
      <div className="table-wrap"><table><thead><tr><th>版本</th><th>状态</th><th>Embedding</th><th>文档 / Chunk</th><th>语料 Hash</th><th>操作</th></tr></thead><tbody>
        {indexes.map((item) => <tr key={item.version}><td><b>{item.version}</b>{item.current && <span className="current-badge">CURRENT</span>}</td><td><span className={`pill ${item.status}`}>{item.status}</span></td><td>{String(item.manifest.embedding_model ?? "-")}<small>{String(item.manifest.embedding_dimension ?? "-")} dimensions</small></td><td>{String(item.manifest.document_count ?? 0)} / {String(item.manifest.chunk_count ?? 0)}</td><td><code>{String(item.manifest.corpus_hash ?? "-").slice(0, 16)}</code></td><td><div className="action-row">{!item.current && <button className="primary" onClick={() => void action(item.version)}>发布</button>}<button className="secondary" onClick={() => void action(item.version, true)}>回滚到此版本</button></div></td></tr>)}
      </tbody></table>{!indexes.length && <div className="empty-state"><h3>还没有已构建索引</h3><p>先运行 rag.ingest_cli 构建并发布一个索引版本。</p></div>}</div>
    </div>
  );
}

