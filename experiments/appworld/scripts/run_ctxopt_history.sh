#!/usr/bin/env bash
set -euo pipefail

# Simple CLI to run the ctxopt pipeline for history prompts.

PROMPTS_DIR_DEFAULT="../prompt_optimizer/outputs/appworld/251010_history_optimized_retry/optimized_prompts"
PROMPTS_DIR="${1:-$PROMPTS_DIR_DEFAULT}"
# Drop the prompts-dir positional if provided so extra flags can be forwarded
if [[ $# -gt 0 ]]; then
  shift
fi

# Defaults (override via env: MODEL, TAG, SPLIT, CTXOPT_TYPE)
MODEL="${MODEL:-gpt-4.1-mini}"
TAG="${TAG:-2501013_prompteval_hist_opt}"
# For history runs, use the history tiny split by default
SPLIT="${SPLIT:-train_history_tiny}"
CTXOPT_TYPE="${CTXOPT_TYPE:-history}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # experiments/appworld
PIPELINE_PY="$SCRIPT_DIR/run_ctxopt_pipeline.py"

if [[ ! -f "$PIPELINE_PY" ]]; then
  echo "Cannot find pipeline script at $PIPELINE_PY" >&2
  exit 1
fi

echo "Running ctxopt pipeline"
echo "  prompts-dir: $PROMPTS_DIR"
echo "  model:       $MODEL"
echo "  tag:         $TAG"
echo "  split:       $SPLIT"
echo "  type:        $CTXOPT_TYPE"

PYTHON_BIN="$(command -v python3 || command -v python)"
exec "$PYTHON_BIN" "$PIPELINE_PY" \
  --prompts-dir "$PROMPTS_DIR" \
  --main-model-name "$MODEL" \
  --model-name "$MODEL" \
  --ctxopt-type "$CTXOPT_TYPE" \
  --tag "$TAG" \
  --split "$SPLIT" \
  "$@"
