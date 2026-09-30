#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API="${BASE_URL}/api/v1"
USER_ID="${USER_ID:-U001}"

echo "[1/6] readiness"
curl --fail --silent "${API}/health/ready" | python3 -m json.tool

echo "[2/6] create session"
SESSION_JSON="$(curl --fail --silent \
  -X POST "${API}/sessions" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: ${USER_ID}" \
  -d '{"title":"阶段七冒烟测试"}')"
SESSION_ID="$(printf '%s' "${SESSION_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
echo "session=${SESSION_ID}"

echo "[3/6] enqueue message"
RUN_JSON="$(curl --fail --silent \
  -X POST "${API}/sessions/${SESSION_ID}/messages" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: ${USER_ID}" \
  -d '{"content":"帮我查询订单 O10086"}')"
RUN_ID="$(printf '%s' "${RUN_JSON}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
echo "run=${RUN_ID}"

echo "[4/6] wait for worker"
for _ in $(seq 1 30); do
  RUN_STATUS="$(curl --fail --silent "${API}/sessions/${SESSION_ID}/runs/${RUN_ID}" -H "X-User-ID: ${USER_ID}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
  echo "status=${RUN_STATUS}"
  case "${RUN_STATUS}" in
    completed|failed) break ;;
  esac
  sleep 1
done

echo "[5/6] messages"
curl --fail --silent "${API}/sessions/${SESSION_ID}/messages" -H "X-User-ID: ${USER_ID}" | python3 -m json.tool

echo "[6/6] trace"
curl --fail --silent "${API}/sessions/${SESSION_ID}/runs/${RUN_ID}" -H "X-User-ID: ${USER_ID}" | python3 -m json.tool

echo "stage-seven smoke test passed: session=${SESSION_ID} run=${RUN_ID}"

