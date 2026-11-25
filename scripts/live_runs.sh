#!/bin/bash
# The live runs of the mini-project: each case 3 times on Claude Opus 5.5, through Claude Code on the seat.
# Interleaved (round 1 of every case, then round 2, then round 3), so a stop part-way still has every case once.
# It stops at the first run whose model gave no usable reply (a seat limit, say): nothing is retried on its own.
# Usage: scripts/live_runs.sh [ROUNDS]   (default 3)
set -uo pipefail
cd "$(dirname "$0")/.."
ROUNDS=${1:-3}
MODEL=claude-opus-5-5
for round in $(seq 1 "$ROUNDS"); do
  for case in duplicate-delivery poison-message missing-deploys; do
    echo "=== round $round, $case, $(date -u +%H:%M:%SZ)"
    .venv/bin/python -m diagnosis_agent run "$case" --model "$MODEL"
    status=$?
    newest=$(ls -td runs/*_"$case"_"$MODEL"* 2>/dev/null | head -1)
    if [ -n "$newest" ] && grep -q '"reason": "model_error"' "$newest/run.json"; then
      echo "=== stopped: the model gave no usable reply in $newest (see its trace); nothing more is run"
      exit 3
    fi
    echo "=== exit $status"
  done
done
echo "=== done $(date -u +%H:%M:%SZ)"
