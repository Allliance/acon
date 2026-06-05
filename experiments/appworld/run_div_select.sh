#!/usr/bin/env bash
# run_div_select.sh — AppWorld run with ONLINE best-of-N compression selection
# (trajectory-divergence scorer).
#
#   agent      : Qwen/Qwen3.5-35B-A3B  @ r4519u01n01:8000   (VLLM_BASE_URL)
#   compressor : Qwen/Qwen3.5-9B       @ r817u29n05:8000    (VLLM_COMPRESSOR_BASE_URL)
#   judge      : Gemini (gemini_key in configs/private_config.yaml)
#   max tool calls 100 | history threshold 6144 | compression budget 2048
#
# At each compression event the compressor samples 5 candidate summaries; the
# agent forecasts its next 5 actions under each candidate AND under the
# uncompressed history; a judge scores plan similarity; the least-divergent
# candidate is installed. Per-event detail -> outputs/.../compression_selection.json.
#
# Thin wrapper over launch.sh (pins both endpoints, aggregates official TGC).
# Everything after `--` is forwarded to run_parallel.py (e.g. --limit 3 for a
# smoke run, --continue_existing to redo).
#
# Usage:
#   ./run_div_select.sh                          # full SPLITS x NUM_REPS
#   ./run_div_select.sh -- --limit 3             # smoke (3 tasks, no official eval)
#   NUM_REPS=3 NUM_WORKERS=64 ./run_div_select.sh
#   CONFIG=configs/context_opt/qwen3p5_9b_div_rubric_t6k_b2k.yaml TAG=div_rubric_9b ./run_div_select.sh

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT_URL="${AGENT_URL:-http://r4519u01n01:8000}"
COMPRESSOR_URL="${COMPRESSOR_URL:-http://r817u29n05:8000}"
CONFIG="${CONFIG:-configs/context_opt/qwen3p5_9b_div_select_t6k_b2k.yaml}"
# launch.sh reads TAG from the environment for the group dir + TGC aggregation
# AND injects `--tag "$TAG"` into run_parallel.py, so it must be exported here
# (not just passed as a forwarded arg) or the group dir won't match the outputs.
export TAG="${TAG:-div_select_9b}"

# Forwarded to launch.sh via env.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
export MAX_ITER="${MAX_ITER:-100}"
export NUM_REPS="${NUM_REPS:-1}"
export SPLITS="${SPLITS:-test_normal}"
# Divergence selection makes each compression event ~6 agent generations + 5
# judge calls, so keep per-endpoint concurrency moderate by default.
export NUM_WORKERS="${NUM_WORKERS:-32}"
export SEED="${SEED:-42}"

exec "$SCRIPT_DIR/launch.sh" \
    --vllm-url "$AGENT_URL" \
    --compressor-url "$COMPRESSOR_URL" \
    -- \
    --co_config_path "$CONFIG" \
    --history_threshold 6144 \
    --compression_budget 2048 \
    "$@"
