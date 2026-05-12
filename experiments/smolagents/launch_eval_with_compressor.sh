#!/usr/bin/env bash
# Launch retriever + TWO vLLM servers (agent + compressor) + run evaluation.
#
# Usage:
#   ./launch_eval_with_compressor.sh --co_config_path configs/context_opt/qwen3.5-9b_history.yaml [run.py args...]
#
# Environment overrides:
#   RETRIEVER_PORT         starting port for retriever  (default 8001)
#   VLLM_PORT              starting port for agent vLLM (default 8000)
#   VLLM_COMPRESSOR_PORT   starting port for compressor (default 8010)
#   MODEL                  agent model      (default Qwen/Qwen3.5-35B-A3B)
#   COMPRESSOR_MODEL       compressor model (REQUIRED unless using a selection-only baseline)
#
# Selection-based baselines (fifo / mask_obs / mask_action / random) do NOT
# need a compressor LLM; just omit COMPRESSOR_MODEL and the second server is
# skipped automatically.
#
# Skip-server mode:
#   Set VLLM_BASE_URL and/or VLLM_COMPRESSOR_BASE_URL to use *already-running*
#   vLLM servers. The script will not start those servers itself, and will
#   only wait on the retriever. URLs can be "host:port" or "http://host:port".
#   Example:
#     VLLM_BASE_URL=r818u33n08:8000 \
#     VLLM_COMPRESSOR_BASE_URL=r818u33n08:8001 \
#       ./launch_eval_with_compressor.sh --co_config_path ... --tag ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
COMPRESSOR_MODEL="${COMPRESSOR_MODEL:-}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8001}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_COMPRESSOR_PORT="${VLLM_COMPRESSOR_PORT:-8010}"

# If these are set, we use the existing remote vLLM and skip launching one.
AGENT_URL="${VLLM_BASE_URL:-}"
COMPRESSOR_URL="${VLLM_COMPRESSOR_BASE_URL:-}"
SKIP_AGENT_VLLM=0
SKIP_COMPRESSOR_VLLM=0
[[ -n "$AGENT_URL"      ]] && SKIP_AGENT_VLLM=1
[[ -n "$COMPRESSOR_URL" ]] && SKIP_COMPRESSOR_VLLM=1
export VLLM_BASE_URL VLLM_COMPRESSOR_BASE_URL

BM25_DATA_DIR=/gpfs/radev/project/cohan/hl2222/data/search-r1
BM25_INDEX_PATH=$BM25_DATA_DIR/bm25
BM25_CORPUS_PATH=$BM25_DATA_DIR/wiki-18.jsonl

INDEX_PATH=$BM25_INDEX_PATH
CORPUS_PATH=$BM25_CORPUS_PATH

# ── port discovery ────────────────────────────────────────────────────────────
while ss -tlnp 2>/dev/null | grep -q ":${RETRIEVER_PORT} "; do
    RETRIEVER_PORT=$((RETRIEVER_PORT + 1))
done
if [[ "$SKIP_AGENT_VLLM" -eq 0 ]]; then
    while ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT} "; do
        VLLM_PORT=$((VLLM_PORT + 1))
    done
fi
if [[ "$SKIP_COMPRESSOR_VLLM" -eq 0 ]]; then
    while ss -tlnp 2>/dev/null | grep -q ":${VLLM_COMPRESSOR_PORT} "; do
        VLLM_COMPRESSOR_PORT=$((VLLM_COMPRESSOR_PORT + 1))
    done
    if [[ "$VLLM_COMPRESSOR_PORT" == "$VLLM_PORT" && "$SKIP_AGENT_VLLM" -eq 0 ]]; then
        VLLM_COMPRESSOR_PORT=$((VLLM_PORT + 10))
    fi
fi

echo "[launch] Retriever  port: $RETRIEVER_PORT"
if [[ "$SKIP_AGENT_VLLM" -eq 1 ]]; then
    echo "[launch] Agent vLLM:      using remote $AGENT_URL"
else
    echo "[launch] Agent vLLM port: $VLLM_PORT       model: $MODEL"
fi
if [[ "$SKIP_COMPRESSOR_VLLM" -eq 1 ]]; then
    echo "[launch] Comp.  vLLM:      using remote $COMPRESSOR_URL"
elif [[ -n "$COMPRESSOR_MODEL" ]]; then
    echo "[launch] Comp.  vLLM port: $VLLM_COMPRESSOR_PORT  model: $COMPRESSOR_MODEL"
else
    echo "[launch] Compressor vLLM: skipped (selection-only baseline)"
fi

export RETRIEVER_PORT VLLM_PORT VLLM_COMPRESSOR_PORT

# ── cleanup on exit ────────────────────────────────────────────────────────────
RETRIEVER_PID=""
VLLM_PID=""
COMPRESSOR_PID=""

cleanup() {
    echo "[launch] Shutting down servers..."
    [[ -n "$RETRIEVER_PID"  ]] && kill "$RETRIEVER_PID"  2>/dev/null || true
    [[ -n "$VLLM_PID"       ]] && kill -- -"$VLLM_PID"       2>/dev/null || true
    [[ -n "$COMPRESSOR_PID" ]] && kill -- -"$COMPRESSOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── retriever ─────────────────────────────────────────────────────────────────
RETRIEVER_LOG="/home/aa3242/scratch/logs/retriever/${SLURM_JOB_ID:-local}.log"
mkdir -p "$(dirname "$RETRIEVER_LOG")"
module load Java/21.0.2 2>/dev/null || true

nohup conda run -n retriever python "$SCRIPT_DIR/search/retriever_server.py" \
    --index_path  "$INDEX_PATH"  \
    --corpus_path "$CORPUS_PATH" \
    --port        "$RETRIEVER_PORT" \
    > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

# ── agent vLLM ────────────────────────────────────────────────────────────────
if [[ "$SKIP_AGENT_VLLM" -eq 0 ]]; then
    VLLM_LOG="/home/aa3242/scratch/logs/vllm_server/${SLURM_JOB_ID:-local}.log"
    mkdir -p "$(dirname "$VLLM_LOG")"
    setsid env \
        VLLM_BIN="$HOME/scratch/envs/vllm/bin/vllm" \
        VLLM_LIB="$HOME/scratch/envs/vllm/lib" \
        VLLM_GPU_MEM_UTIL=0.45 \
        bash "$HOME/serve_vllm.sh" \
            -m "$MODEL" \
            -p "$VLLM_PORT" \
            -l 65536 \
            --thinking \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
fi

# ── compressor vLLM (optional) ────────────────────────────────────────────────
if [[ "$SKIP_COMPRESSOR_VLLM" -eq 0 && -n "$COMPRESSOR_MODEL" ]]; then
    COMPRESSOR_LOG="/home/aa3242/scratch/logs/vllm_compressor/${SLURM_JOB_ID:-local}.log"
    mkdir -p "$(dirname "$COMPRESSOR_LOG")"
    setsid env \
        VLLM_BIN="$HOME/scratch/envs/vllm/bin/vllm" \
        VLLM_LIB="$HOME/scratch/envs/vllm/lib" \
        VLLM_GPU_MEM_UTIL=0.45 \
        bash "$HOME/serve_vllm.sh" \
            -m "$COMPRESSOR_MODEL" \
            -p "$VLLM_COMPRESSOR_PORT" \
            -l 32768 \
        > "$COMPRESSOR_LOG" 2>&1 &
    COMPRESSOR_PID=$!
fi

# ── wait for retriever ────────────────────────────────────────────────────────
echo "[launch] Waiting for retriever..."
until curl -sf "http://localhost:${RETRIEVER_PORT}/retrieve" \
        -X POST -H "Content-Type: application/json" \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; do
    kill -0 "$RETRIEVER_PID" 2>/dev/null || { echo "[launch] retriever died"; exit 1; }
    sleep 3
done

# ── wait for agent vLLM ───────────────────────────────────────────────────────
if [[ "$SKIP_AGENT_VLLM" -eq 0 ]]; then
    echo "[launch] Waiting for agent vLLM..."
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "[launch] agent vLLM died"; exit 1; }
        sleep 5
    done
fi

# ── wait for compressor vLLM ──────────────────────────────────────────────────
if [[ "$SKIP_COMPRESSOR_VLLM" -eq 0 && -n "$COMPRESSOR_MODEL" ]]; then
    echo "[launch] Waiting for compressor vLLM..."
    until curl -sf "http://localhost:${VLLM_COMPRESSOR_PORT}/health" > /dev/null 2>&1; do
        kill -0 "$COMPRESSOR_PID" 2>/dev/null || { echo "[launch] compressor vLLM died"; exit 1; }
        sleep 5
    done
fi

# ── run evaluation ────────────────────────────────────────────────────────────
echo "[launch] Starting evaluation..."
cd "$SCRIPT_DIR"
conda run -p ~/scratch/envs/smolagents python run.py \
    --model_name  "$MODEL" \
    --data_folder data/nq_multi_8 \
    "$@"
echo "[launch] Done."
