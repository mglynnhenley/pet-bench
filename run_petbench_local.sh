#!/usr/bin/env bash
# Run Pet-Bench end-to-end WITHOUT docker (host processes + Inspect local sandbox).
#   petcare-system -> uvicorn petcare.server:app on 127.0.0.1:8000
#   mailhog        -> ./mailhog, API :8025 / SMTP :1025
#   agent          -> sandbox="local" (the agent's bash runs on THIS host)
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-openrouter/openai/gpt-4o-mini}"
[ "${1:-}" = "--model" ] && MODEL="$2"

LOGDIR="$(mktemp -d)"; echo "logs: $LOGDIR"
pids=(); cleanup(){ for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

echo "starting petcare-system (uvicorn :8000)…"
PETBENCH_SEED="${PETBENCH_SEED:-1}" PETBENCH_SCENARIO="${PETBENCH_SCENARIO:-baseline}" \
  python3 -m uvicorn petcare.server:app --host 127.0.0.1 --port 8000 \
  > "$LOGDIR/petcare.log" 2>&1 & pids+=($!)
echo "starting mailhog…"
./mailhog -api-bind-addr 127.0.0.1:8025 -ui-bind-addr 127.0.0.1:8025 \
  -smtp-bind-addr 127.0.0.1:1025 > "$LOGDIR/mailhog.log" 2>&1 & pids+=($!)

for i in $(seq 1 20); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 \
    && curl -sf http://127.0.0.1:8025/api/v2/messages >/dev/null 2>&1 \
    && { echo "services up"; break; }
  sleep 1; [ "$i" = 20 ] && { echo "services failed"; cat "$LOGDIR"/*.log; exit 1; }
done

export PET_SANDBOX=local PET_API=http://localhost:8000 MAILHOG_HOST=localhost
# Every run is saved as a self-contained .eval log here, so trajectories can be
# re-graded post-mortem (`python regrade.py` / `inspect score`) and browsed
# (`inspect view --log-dir logs`). PETBENCH_GRADER picks the judge model.
export INSPECT_LOG_DIR="${INSPECT_LOG_DIR:-logs}"

# behaviour_judge is the eval's SOLE scorer. With PETBENCH_GRADER unset it falls
# back to --model, i.e. the model under test grades its own trajectory — fine for a
# pipeline smoke test, worthless as a result. Warn loudly rather than let a whole
# sweep land in logs/ before anyone notices.
if [ -z "${PETBENCH_GRADER:-}" ]; then
  echo "WARNING: PETBENCH_GRADER is unset — the judge will be $MODEL, the same model"
  echo "         under test. It is grading itself. Set PETBENCH_GRADER to an"
  echo "         independent model before trusting any score from this run."
fi
echo "running inspect eval (model=$MODEL, grader=${PETBENCH_GRADER:-SELF (see warning)}, logs=$INSPECT_LOG_DIR)…"
inspect eval petbench.py --model "$MODEL" --display plain "${@:1}"
