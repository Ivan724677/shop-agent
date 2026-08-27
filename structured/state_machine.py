"""对话任务阶段状态机（阶段三对话流程层的守卫）。

复用 domain.state_machine.StateMachine 基类，定义 9 个对话阶段的
合法转换表（idle → collecting_info → ... → completed）。

与 domain/state_machine.py（业务记录状态机）是两层不同的状态机：
业务层管数据事实（订单/物流/退货/退款/工单），对话层管任务进度（这里）。

REJECTED/HANDOFF 是逃生舱，可从任意阶段直接进入。
"""

from __future__ import annotations

from domain.state_machine import InvalidTransition, StateMachine, TransitionRule

from .models import StructuredTaskState, TaskStage


DIALOGUE_STATE_MACHINE = StateMachine(
    "structured_dialogue",
    [
        TransitionRule("idle", "collecting_info"),
        TransitionRule("idle", "resolving_entities"),
        TransitionRule("collecting_info", "resolving_entities"),
        TransitionRule("collecting_info", "checking_policy"),
        TransitionRule("resolving_entities", "collecting_info"),
        TransitionRule("resolving_entities", "checking_policy"),
        TransitionRule("resolving_entities", "executing"),
        TransitionRule("checking_policy", "collecting_info"),
        TransitionRule("checking_policy", "awaiting_confirmation"),
        TransitionRule("checking_policy", "executing"),
        TransitionRule("checking_policy", "rejected"),
        TransitionRule("awaiting_confirmation", "collecting_info"),
        TransitionRule("awaiting_confirmation", "checking_policy"),
        TransitionRule("awaiting_confirmation", "executing"),
        TransitionRule("executing", "completed"),
        TransitionRule("executing", "collecting_info"),
        TransitionRule("completed", "collecting_info"),
        TransitionRule("completed", "resolving_entities"),
        TransitionRule("rejected", "collecting_info"),
        TransitionRule("rejected", "resolving_entities"),
        TransitionRule("handoff", "collecting_info"),
    ],
)


TERMINAL_OR_ESCAPE = {TaskStage.REJECTED, TaskStage.HANDOFF}


def transition_state(state: StructuredTaskState, target: TaskStage) -> None:
    if state.stage == target:
        return
    if target in TERMINAL_OR_ESCAPE:
        state.stage = target
        return
    DIALOGUE_STATE_MACHINE.transition(state.stage.value, target.value)
    state.stage = target


__all__ = ["DIALOGUE_STATE_MACHINE", "InvalidTransition", "transition_state"]
