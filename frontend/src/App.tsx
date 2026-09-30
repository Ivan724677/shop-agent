import { NavLink, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import MetricsPage from "./pages/MetricsPage";
import RagIndexesPage from "./pages/RagIndexesPage";
import TracePage from "./pages/TracePage";
import WorkbenchPage from "./pages/WorkbenchPage";

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">RA</span>
          <div><strong>Risk-Aware Agent</strong><small>Stage 7 Console</small></div>
        </div>
        <nav>
          <NavLink className={({ isActive }) => isActive ? "active" : ""} to="/">用户聊天</NavLink>
          <NavLink className={({ isActive }) => isActive ? "active" : ""} to="/workbench">人工工作台</NavLink>
          <NavLink className={({ isActive }) => isActive ? "active" : ""} to="/traces">Agent Trace</NavLink>
          <NavLink className={({ isActive }) => isActive ? "active" : ""} to="/rag-indexes">RAG 索引</NavLink>
          <NavLink className={({ isActive }) => isActive ? "active" : ""} to="/metrics">指标面板</NavLink>
        </nav>
        <div className="sidebar-note">
          <span className="status-dot" /> PostgreSQL + Redis Worker
        </div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/workbench" element={<WorkbenchPage />} />
          <Route path="/traces" element={<TracePage />} />
          <Route path="/traces/:sessionId" element={<TracePage />} />
          <Route path="/rag-indexes" element={<RagIndexesPage />} />
          <Route path="/metrics" element={<MetricsPage />} />
        </Routes>
      </main>
    </div>
  );
}
