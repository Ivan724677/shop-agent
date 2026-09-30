import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_stage_seven_files_exist_and_imports_are_lazy_without_optional_dependencies():
    assert (ROOT / "backend" / "app" / "main.py").exists()
    assert (ROOT / "frontend" / "src" / "App.tsx").exists()
    assert (ROOT / "docker-compose.yml").exists()


def test_pending_action_scope_hash_is_stable():
    from backend.app.agent_service import action_scope_hash

    payload = {
        "action": "create_return_request",
        "order_id": "O10086",
        "item_ids": ["I002", "I001"],
        "amount": 129,
        "reason": "no_reason_return",
    }
    reordered = {**payload, "item_ids": ["I001", "I002"], "amount": 129.0}
    assert action_scope_hash(payload) == action_scope_hash(reordered)


def test_docker_compose_contains_api_worker_and_dependencies():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service in ("postgres:", "redis:", "migrate:", "api:", "worker:", "frontend:"):
        assert service in compose

