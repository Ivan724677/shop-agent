import type { PendingAction } from "../types";

type Props = {
  action: PendingAction;
  busy: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

const reasonLabel: Record<string, string> = {
  no_reason_return: "无理由退货",
  quality_issue: "质量问题"
};

export default function ConfirmationCard({ action, busy, onConfirm, onCancel }: Props) {
  const payload = action.payload_json;
  return (
    <section className="confirmation-card">
      <div className="risk-header"><span>高风险操作确认</span><b>需要明确授权</b></div>
      <dl>
        <div><dt>操作</dt><dd>创建退货申请</dd></div>
        <div><dt>订单</dt><dd>{String(payload.order_id ?? "-")}</dd></div>
        <div><dt>商品范围</dt><dd>{(payload.item_ids ?? []).join("、") || "-"}</dd></div>
        <div><dt>退款金额</dt><dd>¥{Number(payload.amount ?? 0).toFixed(2)}</dd></div>
        <div><dt>售后原因</dt><dd>{reasonLabel[String(payload.reason)] ?? String(payload.reason ?? "-")}</dd></div>
        <div><dt>确认有效期</dt><dd>{new Date(action.expires_at).toLocaleString()}</dd></div>
      </dl>
      <p className="confirmation-warning">确认令牌与当前会话版本绑定；如果商品、金额或原因发生变化，本次确认自动失效。</p>
      <div className="action-row">
        <button className="primary danger" disabled={busy} onClick={onConfirm}>确认提交</button>
        <button className="secondary" disabled={busy} onClick={onCancel}>取消操作</button>
      </div>
    </section>
  );
}

