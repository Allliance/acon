#!/usr/bin/env bash
# Launch retriever + vLLM servers, then run evaluation.
#
# Usage:
#   ./launch_eval.sh [run.py args...]
#
# Environment overrides:
#   RETRIEVER_PORT  starting port for retriever (default 8005)
#   VLLM_PORT       starting port for vLLM      (default 8000)
#   MODEL           model to serve              (default Qwen/Qwen3.5-4B)
#
# Any extra arguments are forwarded to run.py.
# Example:
#   ./launch_eval.sh --split test --limit 10 --tag local_qwen

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8005}"
VLLM_PORT="${VLLM_PORT:-8000}"

INDEX_PATH="/home/aa3242/scratch/search-r1/bm25"
CORPUS_PATH="/home/aa3242/scratch/search-r1/wiki-18.jsonl"

# ── port discovery ─────────────────────────────────────────────────────────────
while ss -tlnp 2>/dev/null | grep -q ":${RETRIEVER_PORT} "; do
    RETRIEVER_PORT=$((RETRIEVER_PORT + 1))
done
echo "[launch] Retriever will use port $RETRIEVER_PORT"

while ss -tlnp 2>/dev/null | grep -q ":${VLLM_PORT} "; do
    VLLM_PORT=$((VLLM_PORT + 1))
done
echo "[launch] vLLM will use port $VLLM_PORT"

export RETRIEVER_PORT VLLM_PORT

# ── cleanup on exit ────────────────────────────────────────────────────────────
RETRIEVER_PID=""
VLLM_PID=""

cleanup() {
    echo "[launch] Shutting down servers..."
    [[ -n "$RETRIEVER_PID" ]] && kill "$RETRIEVER_PID" 2>/dev/null || true
    # serve_vllm.sh spawns its own background process; the script itself blocks
    # on wait, so killing its process group covers everything.
    [[ -n "$VLLM_PID" ]] && kill -- -"$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── start retriever server ─────────────────────────────────────────────────────
RETRIEVER_LOG="/home/aa3242/scratch/logs/retriever/${SLURM_JOB_ID}.log"
echo "[launch] Starting retriever server (log: $RETRIEVER_LOG)..."

nohup conda run -n retriever python "$SCRIPT_DIR/search/retriever_server.py" \
    --index_path  "$INDEX_PATH"  \
    --corpus_path "$CORPUS_PATH" \
    --port        "$RETRIEVER_PORT" \
    > "$RETRIEVER_LOG" 2>&1 &
RETRIEVER_PID=$!

# ── start vLLM server ──────────────────────────────────────────────────────────
VLLM_LOG="/home/aa3242/scratch/logs/vllm_server/${SLURM_JOB_ID}.log"
echo "[launch] Starting vLLM server (log: $VLLM_LOG)..."

# Run serve_vllm.sh in its own process group so we can kill all children.
# Redirect output so we can monitor it; the script itself waits for readiness.
setsid bash /home/aa3242/serve_vllm.sh \
    -m "$MODEL" \
    -p "$VLLM_PORT" \
    -l 65536 \
    --thinking \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# ── wait for retriever ─────────────────────────────────────────────────────────
echo "[launch] Waiting for retriever server on port $RETRIEVER_PORT..."
until curl -sf "http://localhost:${RETRIEVER_PORT}/retrieve" \
        -X POST -H "Content-Type: application/json" \
        -d '{"queries":["test"],"topk":1}' > /dev/null 2>&1; do
    if ! kill -0 "$RETRIEVER_PID" 2>/dev/null; then
        echo "[launch] ERROR: retriever process died. Check $RETRIEVER_LOG"
        exit 1
    fi
    sleep 3
done
echo "[launch] Retriever server ready."

# ── wait for vLLM ─────────────────────────────────────────────────────────────
echo "[launch] Waiting for vLLM server on port $VLLM_PORT..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[launch] ERROR: vLLM process died. Check $VLLM_LOG"
        exit 1
    fi
    sleep 5
done
echo "[launch] vLLM server ready."

# ── run evaluation ─────────────────────────────────────────────────────────────
echo "[launch] Starting evaluation..."
cd "$SCRIPT_DIR"

conda run -p ~/scratch/envs/smolagents python run.py \
    --model_name  "$MODEL" \
    --data_folder data/nq_multi_8 \
    "$@"

echo "[launch] Evaluation complete."