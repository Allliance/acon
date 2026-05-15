#!/usr/bin/env bash
# launch.sh — unified launcher for AppWorld compression-baseline experiments.
#
# Mirrors experiments/smolagents/launch.sh but for AppWorld:
#   * no retriever (AppWorld runs in-process; there is no BM25 server)
#   * resolves an agent vLLM and, when the co_config uses an LLM compressor,
#     a second compressor vLLM
#   * then runs run_parallel.py for NUM_REPS reps over SPLITS, and aggregates
#     the official AppWorld TGC across reps (same logic as run_qwen_appworld.sh)
#
# Server reuse:
#   Before spawning anything, ports 8000..8010 are probed for an already
#   running vLLM (anything answering /v1/models). A match is reused; otherwise
#   a server is spawned. Pin an endpoint with --vllm-url / --compressor-url to
#   skip the scan entirely (useful for a remote vLLM on another node).
#
# ─── Usage ─────────────────────────────────────────────────────────────────────
#
#   ./launch.sh [launcher flags] -- [run_parallel.py flags]
#
# Everything after "--" is forwarded verbatim to run_parallel.py. The launcher
# injects --split / --model_name / --tag / --num_workers / --max_iter / --seed,
# so you only pass --co_config_path (and optionally --history_threshold /
# --compression_budget / --continue_existing).
#
# ─── Launcher flags / env vars ────────────────────────────────────────────────
#
#   Agent vLLM:
#     --vllm-url URL        | env VLLM_BASE_URL            pin agent endpoint
#     --model MODEL         | env MODEL_NAME               model to serve (if launching)
#                                                          default: Qwen/Qwen3.5-35B-A3B
#   Compressor vLLM (only if the co_config uses an LLM compressor):
#     --compressor-url URL  | env VLLM_COMPRESSOR_BASE_URL pin compressor endpoint
#     --compressor-model M  | env COMPRESSOR_MODEL         compressor to serve (if launching)
#                                                          empty ⇒ selection-only / OpenAI
#                                                          compressor; no server spawned
#   Run loop (env only):
#     TAG          output tag prefix              (default appworld_baseline)
#     NUM_REPS     repetitions per split          (default 3)
#     SPLITS       space-separated splits         (default test_normal)
#     MAX_ITER     max agent iterations           (default 50)
#     NUM_WORKERS  parallel task workers          (default 128)
#     SEED         fixed seed (all reps)          (default 42)
#     CONDA_ENV    conda env to run in            (default smolagents)
#
# ─── Examples ─────────────────────────────────────────────────────────────────
#
#   # Selection-only baseline (no compressor LLM), defaults:
#   ./launch.sh -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo
#
#   # GPT-4.1-mini compressor (OpenAI API; no compressor vLLM needed):
#   ./launch.sh -- --co_config_path configs/context_opt/gpt-4.1-mini_t8k_b4k.yaml --tag gpt41mini
#
#   # Qwen compressor — launches a second vLLM unless one already runs:
#   ./launch.sh --compressor-model Qwen/Qwen3.5-9B \
#               -- --co_config_path configs/context_opt/qwen3p5_9b_prompting_t8k_b4k.yaml --tag qwen9b
#
#   # Reuse a remote agent vLLM on another node:
#   ./launch.sh --vllm-url http://r818u33n08:8000 \
#               -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo
#
#   # Same model for agent and compressor (one shared vLLM server):
#   ./launch.sh --compressor-url http://localhost:8000 \
#               -- --co_config_path configs/context_opt/qwen3p5_35b_a3b_prompting_t8k_b4k.yaml --tag qwen35b

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── defaults ─────────────────────────────────────────────────────────────────
MODEL="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
COMPRESSOR_MODEL="${COMPRESSOR_MODEL:-}"
AGENT_URL="${VLLM_BASE_URL:-}"
COMPRESSOR_URL="${VLLM_COMPRESSOR_BASE_URL:-}"

VLLM_PORT_START="${VLLM_PORT:-8000}"
VLLM_COMPRESSOR_PORT_START="${VLLM_COMPRESSOR_PORT:-8010}"
PORT_SCAN_LO=8000
PORT_SCAN_HI=8010

TAG="${TAG:-appworld_baseline}"
NUM_REPS="${NUM_REPS:-3}"
SPLITS="${SPLITS:-test_normal}"
MAX_ITER="${MAX_ITER:-50}"
NUM_WORKERS="${NUM_WORKERS:-128}"
SEED="${SEED:-42}"
CONDA_ENV="${CONDA_ENV:-smolagents}"

# ─── flag parsing (launcher flags before --, run_parallel flags after) ────────
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-url)         AGENT_URL="$2";        shift 2 ;;
        --compressor-url)   COMPRESSOR_URL="$2";   shift 2 ;;
        --model)            MODEL="$2";            shift 2 ;;
        --compressor-model) COMPRESSOR_MODEL="$2"; shift 2 ;;
        --)                 shift; RUN_ARGS+=("$@"); break  ;;
        *)                  RUN_ARGS+=("$1");      shift    ;;
    esac
done

# ─── helpers ──────────────────────────────────────────────────────────────────
normalize_url() {
    local u="$1"
    [[ -z "$u" ]] && return
    [[ "$u" =~ ^https?:// ]] && { echo "$u"; return; }
    echo "http://$u"
}

is_port_listening() {
    ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"
}

probe_vllm() {
    local port="$1" body
    body=$(curl -sf -m 2 "http://localhost:${port}/v1/models" 2>/dev/null) || return 1
    echo "$body" | python -c 'import sys,json;d=json.load(sys.stdin);print(d.get("data",[{}])[0].get("id",""))' 2>/dev/null
}

next_free_port() {
    local p="$1"
    while is_port_listening "$p"; do p=$((p + 1)); done
    echo "$p"
}

# ─── agent vLLM resolution ────────────────────────────────────────────────────
VLLM_PID=""
if [[ -n "$AGENT_URL" ]]; then
    AGENT_URL=$(normalize_url "$AGENT_URL")
    echo "[launch] agent vLLM: pinned $AGENT_URL"
else
    found_port=""; found_model=""
    for p in $(seq $PORT_SCAN_LO $PORT_SCAN_HI); do
        mid=$(probe_vllm "$p" || true)
        [[ -z "$mid" ]] && continue
        if [[ "$mid" == "$MODEL" ]]; then found_port="$p"; found_model="$mid"; break; fi
        [[ -z "$found_port" ]] && { found_port="$p"; found_model="$mid"; }
    done
    if [[ -n "$found_port" ]]; then
        VLLM_PORT="$found_port"
        AGENT_URL="http://localhost:${VLLM_PORT}"
        echo "[launch] agent vLLM: reusing existing on port $VLLM_PORT (model: $found_model)"
        MODEL="$found_model"
    else
        VLLM_PORT=$(next_free_port "$VLLM_PORT_START")
        AGENT_URL="http://localhost:${VLLM_PORT}"
        echo "[launch] agent vLLM: spawning on port $VLLM_PORT (model: $MODEL)"
        VLLM_LOG="$HOME/scratch/logs/vllm_server/${SLURM_JOB_ID:-local}.log"
        mkdir -p "$(dirname "$VLLM_LOG")"
        setsid env \
            VLLM_BIN="$HOME/scratch/envs/vllm/bin/vllm" \
            VLLM_LIB="$HOME/scratch/envs/vllm/lib" \
            VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.45}" \
            bash "$HOME/serve_vllm.sh" \
                -m "$MODEL" \
                -p "$VLLM_PORT" \
                -l 65536 \
                --thinking \
            > "$VLLM_LOG" 2>&1 &
        VLLM_PID=$!
    fi
fi

# ─── compressor vLLM resolution ───────────────────────────────────────────────
COMPRESSOR_PID=""
if [[ -n "$COMPRESSOR_URL" ]]; then
    COMPRESSOR_URL=$(normalize_url "$COMPRESSOR_URL")
    echo "[launch] compressor vLLM: pinned $COMPRESSOR_URL"
elif [[ -n "$COMPRESSOR_MODEL" ]]; then
    agent_port_re=$(echo "$AGENT_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
    found_port=""
    for p in $(seq $PORT_SCAN_LO $PORT_SCAN_HI); do
        [[ "$p" == "$agent_port_re" ]] && continue
        mid=$(probe_vllm "$p" || true)
        [[ "$mid" == "$COMPRESSOR_MODEL" ]] && { found_port="$p"; break; }
    done
    if [[ -n "$found_port" ]]; then
        VLLM_COMPRESSOR_PORT="$found_port"
        COMPRESSOR_URL="http://localhost:${VLLM_COMPRESSOR_PORT}"
        echo "[launch] compressor vLLM: reusing existing on port $VLLM_COMPRESSOR_PORT"
    else
        VLLM_COMPRESSOR_PORT=$(next_free_port "$VLLM_COMPRESSOR_PORT_START")
        COMPRESSOR_URL="http://localhost:${VLLM_COMPRESSOR_PORT}"
        echo "[launch] compressor vLLM: spawning on port $VLLM_COMPRESSOR_PORT (model: $COMPRESSOR_MODEL)"
        COMPRESSOR_LOG="$HOME/scratch/logs/vllm_compressor/${SLURM_JOB_ID:-local}.log"
        mkdir -p "$(dirname "$COMPRESSOR_LOG")"
        setsid env \
            VLLM_BIN="$HOME/scratch/envs/vllm/bin/vllm" \
            VLLM_LIB="$HOME/scratch/envs/vllm/lib" \
            VLLM_GPU_MEM_UTIL="${VLLM_COMPRESSOR_GPU_MEM_UTIL:-0.45}" \
            bash "$HOME/serve_vllm.sh" \
                -m "$COMPRESSOR_MODEL" \
                -p "$VLLM_COMPRESSOR_PORT" \
                -l 32768 \
            > "$COMPRESSOR_LOG" 2>&1 &
        COMPRESSOR_PID=$!
    fi
else
    echo "[launch] compressor vLLM: skipped (selection-only / OpenAI compressor)"
fi

# ─── cleanup only what we spawned ─────────────────────────────────────────────
cleanup() {
    echo "[launch] cleaning up spawned servers..."
    [[ -n "$VLLM_PID"       ]] && kill -- -"$VLLM_PID"       2>/dev/null || true
    [[ -n "$COMPRESSOR_PID" ]] && kill -- -"$COMPRESSOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── wait for readiness (only on things we spawned) ───────────────────────────
if [[ -n "$VLLM_PID" ]]; then
    echo "[launch] waiting for agent vLLM..."
    until curl -sf -m 2 "${AGENT_URL}/health" >/dev/null 2>&1; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "[launch] agent vLLM died"; exit 1; }
        sleep 5
    done
fi
if [[ -n "$COMPRESSOR_PID" ]]; then
    echo "[launch] waiting for compressor vLLM..."
    until curl -sf -m 2 "${COMPRESSOR_URL}/health" >/dev/null 2>&1; do
        kill -0 "$COMPRESSOR_PID" 2>/dev/null || { echo "[launch] compressor vLLM died"; exit 1; }
        sleep 5
    done
fi

# ─── export endpoints for run_parallel.py / productive_agents ─────────────────
# productive_agents.llm.vLLM reads VLLM_BASE_URL; ctxopt reads
# VLLM_COMPRESSOR_BASE_URL for the prompting baselines.
agent_port=$(echo "$AGENT_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
export VLLM_BASE_URL="$AGENT_URL"
export VLLM_PORT="${agent_port:-8000}"
if [[ -n "$COMPRESSOR_URL" ]]; then
    comp_port=$(echo "$COMPRESSOR_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
    export VLLM_COMPRESSOR_BASE_URL="$COMPRESSOR_URL"
    export VLLM_COMPRESSOR_PORT="${comp_port:-8010}"
fi
# OpenAI client needs a token even when talking to vLLM (which ignores it).
export OPENAI_API_KEY="${OPENAI_API_KEY:-token-abc}"

# ─── activate env + move into the experiment dir ──────────────────────────────
# shellcheck disable=SC1091
source activate "$CONDA_ENV"
cd "$SCRIPT_DIR"

SAFE_MODEL="${MODEL//\//_}"
echo "[launch] model=$MODEL  splits='$SPLITS'  reps=$NUM_REPS  workers=$NUM_WORKERS"

# ─── per-rep, per-split run loop + official-TGC aggregation ───────────────────
declare -A TGC_TOTALS
declare -A TGC_COUNTS

for REP in $(seq 1 "$NUM_REPS"); do
    REP_TAG="${TAG}_rep${REP}"
    echo "========================================================"
    echo "[rep $REP/$NUM_REPS] tag=$REP_TAG seed=$SEED workers=$NUM_WORKERS"
    echo "========================================================"

    for SPLIT in $SPLITS; do
        echo "-------- split: $SPLIT (rep $REP) --------"
        python run_parallel.py \
            --split "$SPLIT" \
            --model_name "$MODEL" \
            --tag "$REP_TAG" \
            --max_iter "$MAX_ITER" \
            --num_workers "$NUM_WORKERS" \
            --seed "$SEED" \
            "${RUN_ARGS[@]}"

        # run_parallel.py may append _t../_b.. to the tag; resolve the dir the
        # same way it does so we read the right summary.jsonl.
        EFF_TAG=$(REP_TAG="$REP_TAG" python - "${RUN_ARGS[@]}" <<'PY'
import sys
args = sys.argv[1:]
import os
def _fmt_k(n):
    if n >= 1024 and n % 1024 == 0: return f"{n // 1024}k"
    if n >= 1000 and n % 1000 == 0: return f"{n // 1000}k"
    return str(n)
ht, cb = 8192, 4096
for i, a in enumerate(args):
    if a == "--history_threshold":   ht = int(args[i + 1])
    elif a == "--compression_budget": cb = int(args[i + 1])
bits = []
if ht != 8192: bits.append(f"t{_fmt_k(ht)}")
if cb != 4096: bits.append(f"b{_fmt_k(cb)}")
tag = os.environ["REP_TAG"] + ("_" + "_".join(bits) if bits else "")
print(tag)
PY
)
        SUMMARY="./outputs/${SAFE_MODEL}_${EFF_TAG}/summary.jsonl"
        if [[ -f "$SUMMARY" ]]; then
            TGC=$(SPLIT="$SPLIT" SUMMARY="$SUMMARY" python -c "
import json, os
split = os.environ['SPLIT']
tgc = None
for line in open(os.environ['SUMMARY']):
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

# ─── aggregate across reps ────────────────────────────────────────────────────
echo
echo "========================================================"
echo "FINAL SUMMARY (official AppWorld TGC): ${MODEL}  (${NUM_REPS} reps)  tag=${TAG}"
echo "========================================================"
{
    echo "totals = {"
    for k in "${!TGC_TOTALS[@]}"; do echo "    '$k': ${TGC_TOTALS[$k]},"; done
    echo "}"
    echo "counts = {"
    for k in "${!TGC_COUNTS[@]}"; do echo "    '$k': ${TGC_COUNTS[$k]},"; done
    echo "}"
    cat <<'PY'
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

echo "[launch] done."
