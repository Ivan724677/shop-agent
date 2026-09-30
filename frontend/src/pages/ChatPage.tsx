import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "../api";
import CitationList from "../components/CitationList";
import ConfirmationCard from "../components/ConfirmationCard";
import type { AgentEvent, Message, PendingAction, Session } from "../types";

const storedUser = localStorage.getItem("agent-user-id") ?? "U001";

export default function ChatPage() {
  const [userId, setUserId] = useState(storedUser);
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [handoff, setHandoff] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void initialize();
    return () => eventSourceRef.current?.close();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function initialize() {
    try {
      let current: Session;
      const storedSession = localStorage.getItem(`agent-session-${storedUser}`);
      if (storedSession) {
        try {
          current = await api.getSession(storedUser, storedSession);
        } catch {
          current = await api.createSession(storedUser);
        }
      } else {
        current = await api.createSession(storedUser);
      }
      localStorage.setItem(`agent-session-${storedUser}`, current.id);
      setSession(current);
      setHandoff(current.status === "waiting_for_human");
      const [history, currentPending] = await Promise.all([
        api.listMessages(storedUser, current.id),
        api.getPendingAction(storedUser, current.id)
      ]);
      setMessages(history);
      setPending(currentPending);
      connectEvents(current.id, storedUser);
    } catch (err) {
      setError(err instanceof Error ? err.message : "初始化失败");
    }
  }

  function connectEvents(sessionId: string, currentUser: string) {
    eventSourceRef.current?.close();
    const source = new EventSource(api.eventUrl(sessionId, currentUser));
    eventSourceRef.current = source;
    source.addEventListener("run.started", () => setBusy(true));
    source.addEventListener("run.failed", (raw) => {
      setBusy(false);
      const event = parseEvent(raw as MessageEvent);
      setError(String(event?.payload?.error ?? "Agent 处理失败"));
    });
    source.addEventListener("message.completed", (raw) => {
      const event = parseEvent(raw as MessageEvent);
      const message = event?.payload?.message as any;
      if (message) {
        setMessages((current) => current.some((item) => item.id === message.id)
          ? current
          : [...current, {
            ...message,
            session_id: sessionId,
            metadata_json: message.metadata ?? {},
            created_at: new Date().toISOString()
          }]);
      }
      setBusy(false);
      setHandoff(Boolean(event?.payload?.handoff));
      if (event?.payload?.pending_action) {
        void api.getPendingAction(currentUser, sessionId).then(setPending);
      } else {
        setPending(null);
      }
    });
    source.addEventListener("handoff.claimed", () => setHandoff(true));
    source.addEventListener("handoff.resolved", () => setHandoff(false));
    source.onerror = () => setError("实时连接暂时中断，浏览器会自动重连。");
  }

  async function send(event: FormEvent) {
    event.preventDefault();
    if (!session || !input.trim() || busy) return;
    const content = input.trim();
    setInput("");
    setError("");
    setBusy(true);
    setMessages((current) => [...current, {
      id: crypto.randomUUID(),
      session_id: session.id,
      role: "user",
      content,
      sequence: current.length + 1,
      metadata_json: { optimistic: true },
      created_at: new Date().toISOString()
    }]);
    try {
      await api.sendMessage(userId, session.id, content);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "发送失败");
    }
  }

  async function confirm(confirm: boolean) {
    if (!pending) return;
    setBusy(true);
    setError("");
    try {
      if (confirm) await api.confirmAction(userId, pending);
      else await api.cancelAction(userId, pending);
      setPending(null);
    } catch (err) {
      setBusy(false);
      setError(err instanceof Error ? err.message : "操作失败");
      if (session) setPending(await api.getPendingAction(userId, session.id));
    }
  }

  async function newSession() {
    eventSourceRef.current?.close();
    const created = await api.createSession(userId);
    localStorage.setItem(`agent-session-${userId}`, created.id);
    setSession(created);
    setMessages([]);
    setPending(null);
    setHandoff(false);
    connectEvents(created.id, userId);
  }

  function changeUser(value: string) {
    localStorage.setItem("agent-user-id", value);
    setUserId(value);
  }

  return (
    <div className="page chat-page">
      <header className="page-header">
        <div><p className="eyebrow">Customer channel</p><h1>售后客服</h1><p>结构化状态、多专家路由和可追溯政策证据。</p></div>
        <div className="header-actions">
          <label>用户 <input value={userId} onChange={(e) => changeUser(e.target.value)} /></label>
          <button className="secondary" onClick={() => void newSession()}>新会话</button>
        </div>
      </header>

      {handoff && <div className="handoff-banner"><b>已转人工处理</b><span>客服人员可以查看当前任务状态、工具审计和政策证据。</span></div>}
      {error && <div className="error-banner">{error}</div>}

      <section className="chat-panel">
        <div className="session-meta">
          <span>Session</span><code>{session?.id ?? "正在创建..."}</code>
          <span className={`pill ${session?.status ?? "loading"}`}>{session?.status ?? "loading"}</span>
        </div>
        <div className="message-list">
          {!messages.length && <div className="empty-state"><h3>可以从这些问题开始</h3><p>“我要退订单 O10086 里的鞋，耳机不退”</p><p>“帮我查一下 O10086 的物流”</p></div>}
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="message-role">{message.role === "user" ? "你" : "Agent"}</div>
              <div className="message-body">
                <p>{message.content}</p>
                <CitationList ids={message.metadata_json.policy_evidence ?? message.metadata_json.citation_ids ?? []} />
                {message.role === "assistant" && message.metadata_json.run_id && (
                  <div className="feedback-row">
                    <button onClick={() => void api.feedback(userId, String(message.metadata_json.run_id), true)}>有帮助</button>
                    <button onClick={() => void api.feedback(userId, String(message.metadata_json.run_id), false)}>需改进</button>
                  </div>
                )}
              </div>
            </article>
          ))}
          {busy && <article className="message assistant"><div className="message-role">Agent</div><div className="typing"><i /><i /><i /></div></article>}
          {pending && <ConfirmationCard action={pending} busy={busy} onConfirm={() => void confirm(true)} onCancel={() => void confirm(false)} />}
          <div ref={bottomRef} />
        </div>
        <form className="composer" onSubmit={send}>
          <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="描述订单、商品和售后诉求…" rows={2} />
          <button className="primary" disabled={!session || busy || !input.trim()}>发送</button>
        </form>
      </section>
    </div>
  );
}

function parseEvent(message: MessageEvent): AgentEvent | null {
  try { return JSON.parse(message.data) as AgentEvent; }
  catch { return null; }
}

