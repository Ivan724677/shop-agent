export type Session = {
  id: string;
  user_id: string;
  title: string;
  status: string;
  state_json: Record<string, unknown>;
  state_version: number;
  created_at: string;
  updated_at: string;
};

export type Message = {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sequence: number;
  metadata_json: {
    run_id?: string;
    policy_evidence?: string[];
    citation_ids?: string[];
    stage?: string;
    [key: string]: unknown;
  };
  created_at: string;
};

export type PendingAction = {
  id: string;
  session_id: string;
  action_type: string;
  payload_json: {
    order_id?: string;
    item_ids?: string[];
    amount?: number;
    reason?: string;
    [key: string]: unknown;
  };
  state_version: number;
  status: string;
  expires_at: string;
  confirmation_token?: string | null;
};

export type AgentRun = {
  id: string;
  session_id: string;
  status: string;
  current_node: string;
  checkpoint_json: Record<string, unknown>;
  trace_json: Array<Record<string, unknown>>;
  route_trace_json: Array<Record<string, unknown>>;
  response_text?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
};

export type Handoff = {
  id: string;
  session_id: string;
  run_id?: string | null;
  ticket_id?: string | null;
  reason: string;
  priority: string;
  status: string;
  assigned_to?: string | null;
  resolution?: string | null;
  created_at: string;
  updated_at: string;
};

export type RAGIndex = {
  version: string;
  status: string;
  current: boolean;
  manifest: Record<string, unknown>;
};

export type AgentEvent = {
  id?: string;
  sequence?: number;
  event: string;
  payload: Record<string, any>;
};

