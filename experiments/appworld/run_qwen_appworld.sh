#!/usr/bin/env bash
# =============================================================================
# run_qwen_appworld.sh
# -----------------------------------------------------------------------------
# Driver for running AppWorld experiments against a *remote* vLLM server that
# is serving an open-weights model (here: Qwen/Qwen3.5-35B-A3B).
#
# It:
#   1. Activates the `smolagents` conda env (where productive_agents and
#      appworld are installed editable).
#   2. Points the codebase at the remote vLLM endpoint via VLLM_BASE_URL
#      (consumed by productive_agents.llm.vLLM).
#   3. Runs the AppWorld agent (no context compression — baseline) on every
#      requested split, in parallel via run_parallel.py with NUM_WORKERS
#      threads hitting the shared vLLM server.
#   4. Repeats each split NUM_REPS times for variance estimation (all reps
#      use the SAME seed=42 — variance comes from sampling, not seed).
#   5. Aggregates per-split success rates across reps and prints a final
#      easy/normal + average summary.
#
# Outputs land in:
#   experiments/appworld/outputs/<safe_model>_<tag>_repK/<split>/
#       experiment_summary.json     <-- aggregate per-split summary
#       task_<id>/                  <-- per-task trajectories + logs
#
# Usage:
#   bash run_qwen_appworld.sh                # default: 3 reps, dev + test_normal
#   bash run_qwen_appworld.sh --debug        # quick smoke test (1 task, 1 worker)
#   NUM_REPS=1 SPLITS="dev" NUM_WORKERS=32 bash run_qwen_appworld.sh
# =============================================================================

set -euo pipefail

# --- Configuration (override via env) ----------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"     # served by vLLM
VLLM_BASE_URL="${VLLM_BASE_URL:-http://r818u33n06:8000/v1}"
TAG="${TAG:-qwen35_a3b_baseline}"
NUM_REPS="${NUM_REPS:-3}"
# AppWorld splits: only test_normal is evaluated by default (dev / easy and
# test_challenge / hard are skipped). Override with e.g. SPLITS="dev test_normal".
SPLITS="${SPLITS:-test_normal}"
MAX_ITER="${MAX_ITER:-50}"
NUM_WORKERS="${NUM_WORKERS:-128}"
SEED="${SEED:-42}"
CONDA_ENV="${CONDA_ENV:-smolagents}"

DEBUG_FLAG=""
if [[ "${1:-}" == "--debug" ]]; then
    NUM_REPS=1
    SPLITS="test_normal"
    NUM_WORKERS=2
    echo "[debug] Running smoke test on test_normal with 2 workers (still all 168 tasks)."
fi

# --- Activate env -------------------------------------------------------------
# `source activate` is the legacy form that works in non-interactive shells
# without needing `conda init`.
# shellcheck disable=SC1091
source activate "$CONDA_ENV"

# --- Move into AppWorld experiment directory ---------------------------------
cd "$(dirname "$0")"

# --- Sanity: vLLM endpoint reachable -----------------------------------------
export VLLM_BASE_URL
export MODEL_NAME
echo "[setup] vLLM endpoint: $VLLM_BASE_URL"
python - <<'PY'
import os, sys, urllib.request, json
url = os.environ["VLLM_BASE_URL"].rstrip("/") + "/models"
try:
    data = json.loads(urllib.request.urlopen(url, timeout=5).read())
    ids = [m["id"] for m in data.get("data", [])]
    print(f"[setup] vLLM reachable. Served models: {ids}")
    if os.environ["MODEL_NAME"] not in ids:
        print(f"[setup] WARNING: {os.environ['MODEL_NAME']} not in served list; continuing.", file=sys.stderr)
except Exception as e:
    print(f"[setup] FATAL: cannot reach vLLM at {url}: {e}", file=sys.stderr)
    sys.exit(1)
PY

# A token is required by the OpenAI client even though vLLM ignores it.
export OPENAI_API_KEY="${OPENAI_API_KEY:-token-abc}"

SAFE_MODEL="${MODEL_NAME//\//_}"

# --- Per-rep, per-split run loop ---------------------------------------------
# Reps are sequential (the parallelism is *within* a split via threads). All
# reps use SEED=42 — the point of repeating is to wash out vLLM sampling
# stochasticity, not seed stochasticity.
declare -A TGC_TOTALS   # split -> sum of per-rep official TGC (%)
declare -A TGC_COUNTS   # split -> rep count

for REP in $(seq 1 "$NUM_REPS"); do
    REP_TAG="${TAG}_rep${REP}"
    echo "========================================================"
    echo "[rep $REP/$NUM_REPS] tag=$REP_TAG  seed=$SEED  workers=$NUM_WORKERS"
    echo "========================================================"

    for SPLIT in $SPLITS; do
        echo "-------- split: $SPLIT (rep $REP) --------"
        # run_parallel.py runs the agent AND then invokes the official
        # AppWorld evaluator, writing outputs/<exp>/summary.jsonl.
        python run_parallel.py \
            --split "$SPLIT" \
            --model_name "$MODEL_NAME" \
            --tag "$REP_TAG" \
            --max_iter "$MAX_ITER" \
            --num_workers "$NUM_WORKERS" \
            --seed "$SEED"

        SUMMARY="./outputs/${SAFE_MODEL}_${REP_TAG}/summary.jsonl"
        if [[ -f "$SUMMARY" ]]; then
            TGC=$(SPLIT="$SPLIT" python -c "
import json, os, sys
split = os.environ['SPLIT']
tgc = None
for line in open('$SUMMARY'):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get('split') == split:
        tgc = r.get('task_goal_completion')
print(tgc if tgc is not None else 'MISSING')
")
            if [[ "$TGC" == "MISSING" ]]; then
                echo "[rep $REP][$SPLIT] WARNING: no $SPLIT line in $SUMMARY"
            else
                echo "[rep $REP][$SPLIT] official TGC=${TGC}%"
                TGC_TOTALS[$SPLIT]=$(python -c "print(${TGC_TOTALS[$SPLIT]:-0} + $TGC)")
                TGC_COUNTS[$SPLIT]=$((${TGC_COUNTS[$SPLIT]:-0} + 1))
            fi
        else
            echo "[rep $REP][$SPLIT] WARNING: $SUMMARY missing"
        fi
    done
done

# --- Aggregate across reps ----------------------------------------------------
echo
echo "========================================================"
echo "FINAL SUMMARY (official AppWorld TGC): ${MODEL_NAME}  (${NUM_REPS} reps)"
echo "========================================================"
{
    echo "totals = {"
    for k in "${!TGC_TOTALS[@]}"; do
        echo "    '$k': ${TGC_TOTALS[$k]},"
    done
    echo "}"
    echo "counts = {"
    for k in "${!TGC_COUNTS[@]}"; do
        echo "    '$k': ${TGC_COUNTS[$k]},"
    done
    echo "}"
    cat <<'PY'
# TGC values are already percentages (0-100), averaged over reps (seed fixed).
order = [s for s in "dev test_normal test_challenge".split() if s in totals]
means = []
for s in order:
    mean = totals[s] / counts[s] if counts[s] else float("nan")
    means.append(mean)
    print(f"{s:>16s}: mean TGC = {mean:6.2f}%  ({counts[s]} reps)")
if means:
    overall = sum(means) / len(means)
    print(f"{'AVERAGE':>16s}: mean TGC = {overall:6.2f}%  (macro over splits)")
PY
} | python
