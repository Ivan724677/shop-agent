"""阶段三：结构化状态 + 多轮记忆 Agent 包。

暴露核心类型：StructuredCustomerServiceAgent、StructuredTaskState、
Intent、TaskStage、RiskLevel、ConfirmationStatus。
"""

from .agent import StructuredCustomerServiceAgent
from .models import (
    ConfirmationStatus,
    Intent,
    RiskLevel,
    StructuredTaskState,
    TaskStage,
)

__all__ = [
    "ConfirmationStatus",
    "Intent",
    "RiskLevel",
    "StructuredCustomerServiceAgent",
    "StructuredTaskState",
    "TaskStage",
]
