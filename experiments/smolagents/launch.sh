#!/usr/bin/env bash
# launch.sh — unified launcher for the smolagents 8-objective QA eval.
#
# This single script replaces the old launch_eval.sh / launch_eval_with_compressor.sh.
# It works for:
#   * no-compressor runs (just an agent vLLM + retriever)
#   * LLM-based compression runs (agent vLLM + compressor vLLM + retriever)
#   * selection-based baselines (no compressor LLM, just agent + retriever)
#
# Server reuse:
#   Before spawning anything, the script probes ports 8000..8010 looking for an
#   already-running vLLM (any process that answers /v1/models) and an
#   already-running retriever (any process that answers POST /retrieve). When
#   found, it reuses them instead of starting a new server.
#
#   You can also bypass the scan entirely by giving an explicit URL — see
#   "Pinning a server" below. This is the right thing to do when your vLLM
#   lives on another host (e.g. when MODEL is being served remotely).
#
#   Each spawned-vs-reused decision is logged so you can see which servers
#   the run will hit.
#
# ─── Usage ─────────────────────────────────────────────────────────────────────
#
#   ./launch.sh [launcher flags] -- [run.py flags]
#
# Anything before "--" is parsed by the launcher; everything after "--" is
# forwarded verbatim to run.py. If you omit "--", all unknown flags are
# forwarded to run.py.
#
# ─── Launcher flags / env vars ────────────────────────────────────────────────
#
#   Agent vLLM:
#     --vllm-url URL          | env VLLM_BASE_URL            pin agent endpoint
#     --model MODEL           | env MODEL                    model to serve (if launching)
#                                                            default: Qwen/Qwen3.5-35B-A3B
#
#   Compressor vLLM (optional — only if your co_config uses an LLM compressor):
#     --compressor-url URL    | env VLLM_COMPRESSOR_BASE_URL pin compressor endpoint
#     --compressor-model M    | env COMPRESSOR_MODEL         compressor to serve (if launching)
#                                                            empty ⇒ selection-only baseline,
#                                                            no compressor server spawned
#
#   Retriever:
#     --retriever-url URL     | env RETRIEVER_URL            pin retriever endpoint
#     --retriever-port PORT   | env RETRIEVER_PORT           starting port if launching (8001)
#
#   Port scan range: ports 8000..8010 are scanned for existing servers.
#
# ─── Pinning a server (no spawning) ───────────────────────────────────────────
#
# When --vllm-url / --compressor-url / --retriever-url is set, the launcher
# does NOT scan, does NOT spawn, and does NOT verify model identity beyond a
# /health check — it just trusts you. URLs may be "host:port" or
# "http://host:port".
#
#   ./launch.sh --vllm-url r818u33n08:8000 \
#               --retriever-url localhost:8005 \
#               -- --tag remote_agent --co_config_path configs/context_opt/fifo_t8k_b4k.yaml
#
# ─── Examples ─────────────────────────────────────────────────────────────────
#
#   # Selection-only baseline, defaults (no compressor LLM):
#   ./launch.sh -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo
#
#   # GPT-4.1 compressor (OpenAI API; no compressor vLLM needed):
#   ./launch.sh -- --co_config_path configs/context_opt/gpt-4.1-mini_t8k_b4k.yaml --tag gpt41mini
#
#   # Qwen compressor — launches a second vLLM unless one already runs:
#   ./launch.sh --compressor-model Qwen/Qwen3.5-9B \
#               -- --co_config_path configs/context_opt/qwen3p5_9b_prompting_t8k_b4k.yaml --tag qwen9b
#
#   # Reuse a remote agent vLLM you already have on another node:
#   ./launch.sh --vllm-url http://r818u33n08:8000 \
#               -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── defaults ─────────────────────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
COMPRESSOR_MODEL="${COMPRESSOR_MODEL:-}"
AGENT_URL="${VLLM_BASE_URL:-}"
COMPRESSOR_URL="${VLLM_COMPRESSOR_BASE_URL:-}"
RETRIEVER_URL="${RETRIEVER_URL:-}"

# Port-scan starting points / fallbacks
RETRIEVER_PORT="${RETRIEVER_PORT:-8001}"
VLLM_PORT_START="${VLLM_PORT:-8000}"
VLLM_COMPRESSOR_PORT_START="${VLLM_COMPRESSOR_PORT:-8010}"
PORT_SCAN_LO=8000
PORT_SCAN_HI=8010

BM25_DATA_DIR=/gpfs/radev/project/cohan/hl2222/data/search-r1
INDEX_PATH="$BM25_DATA_DIR/bm25"
CORPUS_PATH="$BM25_DATA_DIR/wiki-18.jsonl"

# ─── flag parsing (launcher flags before --, run.py flags after) ──────────────
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-url)         AGENT_URL="$2";          shift 2 ;;
        --compressor-url)   COMPRESSOR_URL="$2";     shift 2 ;;
        --retriever-url)    RETRIEVER_URL="$2";      shift 2 ;;
        --model)            MODEL="$2";              shift 2 ;;
        --compressor-model) COMPRESSOR_MODEL="$2";   shift 2 ;;
        --retriever-port)   RETRIEVER_PORT="$2";     shift 2 ;;
        --)                 shift; RUN_ARGS+=("$@"); break    ;;
        *)                  RUN_ARGS+=("$1");        shift    ;;
    esac
done

# ─── helpers ──────────────────────────────────────────────────────────────────
normalize_url() {
    # "host:port" -> "http://host:port"; already-http URLs pass through.
    local u="$1"
    [[ -z "$u" ]] && return
    [[ "$u" =~ ^https?:// ]] && { echo "$u"; return; }
    echo "http://$u"
}

is_port_listening() {
    ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"
}

probe_vllm() {
    # Returns "MODEL_ID" on stdout if port serves vLLM (/v1/models), else empty.
    local port="$1"
    local body
    body=$(curl -sf -m 2 "http://localhost:${port}/v1/models" 2>/dev/null) || return 1
    # Extract first "id" from response, naive but works for OpenAI-shape JSON.
    echo "$body" | python -c 'import sys,json;d=json.load(sys.stdin);print(d.get("data",[{}])[0].get("id",""))' 2>/dev/null
}

probe_retriever() {
    local port="$1"
    curl -sf -m 2 -X POST "http://localhost:${port}/retrieve" \
        -H "Content-Type: application/json" \
        -d '{"queries":["test"],"topk":1}' >/dev/null 2>&1
}

next_free_port() {
    # Start at $1, scan upward until a free port is found.
    local p="$1"
    while is_port_listening "$p"; do p=$((p + 1)); done
    echo "$p"
}

# ─── retriever resolution ─────────────────────────────────────────────────────
RETRIEVER_PID=""
if [[ -n "$RETRIEVER_URL" ]]; then
    RETRIEVER_URL=$(normalize_url "$RETRIEVER_URL")
    echo "[launch] retriever: pinned $RETRIEVER_URL"
else
    found=""
    for p in $(seq $PORT_SCAN_LO $PORT_SCAN_HI); do
        if probe_retriever "$p"; then found="$p"; break; fi
    done
    if [[ -n "$found" ]]; then
        RETRIEVER_PORT="$found"
        RETRIEVER_URL="http://localhost:${RETRIEVER_PORT}"
        echo "[launch] retriever: reusing existing on port $RETRIEVER_PORT"
    else
        RETRIEVER_PORT=$(next_free_port "$RETRIEVER_PORT")
        RETRIEVER_URL="http://localhost:${RETRIEVER_PORT}"
        echo "[launch] retriever: spawning on port $RETRIEVER_PORT"
        RETRIEVER_LOG="$HOME/scratch/logs/retriever/${SLURM_JOB_ID:-local}.log"
        mkdir -p "$(dirname "$RETRIEVER_LOG")"
        module load Java/21.0.2 2>/dev/null || true
        nohup conda run -n retriever python "$SCRIPT_DIR/search/retriever_server.py" \
            --index_path  "$INDEX_PATH"  \
            --corpus_path "$CORPUS_PATH" \
            --port        "$RETRIEVER_PORT" \
            > "$RETRIEVER_LOG" 2>&1 &
        RETRIEVER_PID=$!
    fi
fi

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
        # Prefer an exact model match; remember the first vLLM seen as fallback.
        if [[ "$mid" == "$MODEL" ]]; then found_port="$p"; found_model="$mid"; break; fi
        [[ -z "$found_port" ]] && { found_port="$p"; found_model="$mid"; }
    done
    if [[ -n "$found_port" ]]; then
        VLLM_PORT="$found_port"
        AGENT_URL="http://localhost:${VLLM_PORT}"
        echo "[launch] agent vLLM: reusing existing on port $VLLM_PORT (model: $found_model)"
        # Update MODEL so run.py talks to whatever's actually being served.
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
    # Try to find an existing vLLM serving the compressor model (skip the agent's port).
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
    echo "[launch] compressor vLLM: skipped (no --compressor-model / COMPRESSOR_MODEL)"
fi

# ─── cleanup only what we spawned ─────────────────────────────────────────────
cleanup() {
    echo "[launch] cleaning up spawned servers..."
    [[ -n "$RETRIEVER_PID"  ]] && kill "$RETRIEVER_PID"  2>/dev/null || true
    [[ -n "$VLLM_PID"       ]] && kill -- -"$VLLM_PID"       2>/dev/null || true
    [[ -n "$COMPRESSOR_PID" ]] && kill -- -"$COMPRESSOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── wait for readiness (only on things we spawned) ───────────────────────────
if [[ -n "$RETRIEVER_PID" ]]; then
    echo "[launch] waiting for retriever..."
    until probe_retriever "$RETRIEVER_PORT"; do
        kill -0 "$RETRIEVER_PID" 2>/dev/null || { echo "[launch] retriever died"; exit 1; }
        sleep 3
    done
fi
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

# ─── export endpoints for run.py / productive_agents ──────────────────────────
# These env vars are what llm.py and ctxopt/history_optimizer.py read.
agent_port=$(echo "$AGENT_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
retr_port=$(echo "$RETRIEVER_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
export VLLM_BASE_URL="$AGENT_URL"
export VLLM_PORT="${agent_port:-8000}"
export RETRIEVER_URL
export RETRIEVER_PORT="${retr_port:-$RETRIEVER_PORT}"
if [[ -n "$COMPRESSOR_URL" ]]; then
    comp_port=$(echo "$COMPRESSOR_URL" | sed -nE 's|.*:([0-9]+).*|\1|p')
    export VLLM_COMPRESSOR_BASE_URL="$COMPRESSOR_URL"
    export VLLM_COMPRESSOR_PORT="${comp_port:-8010}"
fi

# ─── run evaluation ───────────────────────────────────────────────────────────
echo "[launch] starting evaluation: model=$MODEL"
cd "$SCRIPT_DIR"
conda run -p ~/scratch/envs/smolagents python run.py \
    --model_name  "$MODEL" \
    --data_folder data/nq_multi_8 \
    "${RUN_ARGS[@]}"
echo "[launch] done."
