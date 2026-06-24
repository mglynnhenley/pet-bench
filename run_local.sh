#!/usr/bin/env bash
# Run pet_bench end-to-end WITHOUT docker, using host processes + Inspect's
# local sandbox. Needed where docker can't nest containers (see AGENT.md).
#
#   pet_api   -> uvicorn on 127.0.0.1:8000
#   mailhog   -> ./mailhog binary, API 127.0.0.1:8025 / SMTP 127.0.0.1:1025
#   agent     -> sandbox="local" (the agent's bash runs on THIS host)
#
# Requires OPENAI_API_KEY in the environment.
#   OPENAI_API_KEY=sk-... ./run_local.sh [--model openai/gpt-4o-mini]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-openai/gpt-4o-mini}"
[ "${1:-}" = "--model" ] && MODEL="$2"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

LOGDIR="$(mktemp -d)"
echo "logs: $LOGDIR"
pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "starting pet_api (uvicorn :8000)…"
( cd pet_api && python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 ) \
  > "$LOGDIR/petapi.log" 2>&1 & pids+=($!)

echo "starting mailhog (:8025 / :1025)…"
./mailhog -api-bind-addr 127.0.0.1:8025 -ui-bind-addr 127.0.0.1:8025 \
  -smtp-bind-addr 127.0.0.1:1025 > "$LOGDIR/mailhog.log" 2>&1 & pids+=($!)

echo "waiting for services…"
for i in $(seq 1 20); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 \
    && curl -sf http://127.0.0.1:8025/api/v2/messages >/dev/null 2>&1 \
    && { echo "services up"; break; }
  sleep 1
  [ "$i" = 20 ] && { echo "services failed to start"; cat "$LOGDIR"/*.log; exit 1; }
done

export PET_SANDBOX=local
export PET_API=http://localhost:8000
export MAILHOG_HOST=localhost

echo "running inspect eval (model=$MODEL)…"
inspect eval pet_bench.py --model "$MODEL"
