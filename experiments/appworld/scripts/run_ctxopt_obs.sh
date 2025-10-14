#!/usr/bin/env bash
set -euo pipefail

# Simple CLI to run the ctxopt pipeline for observation prompts.
# Default prompts dir targets the 250818_observation_regression optimized outputs.

PROMPTS_DIR_DEFAULT="/home/t-minkikang/productive-agents/experiments/prompt_optimizer/outputs_appworld/250821_observation_regression_gpt5/optimized_prompts"
PROMPTS_DIR="${1:-$PROMPTS_DIR_DEFAULT}"
# Drop the prompts-dir positional if provided so extra flags can be forwarded
if [[ $# -gt 0 ]]; then
  shift
fi

# Defaults (override via env: MODEL, TAG, SPLIT, CTXOPT_TYPE)
MODEL="${MODEL:-gpt-4.1}"
TAG="${TAG:-250831_prompteval_obs_opt_version2_fixed}"
# For observation runs, use the obs tiny split by default
SPLIT="${SPLIT:-train_obs_tiny_version2}"
CTXOPT_TYPE="${CTXOPT_TYPE:-obs}"
OPT_VERSION="${OPT_VERSION:-1}"

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
  --model-name "$MODEL" \
  --ctxopt-type "$CTXOPT_TYPE" \
  --tag "$TAG" \
  --split "$SPLIT" \
  --opt-version $OPT_VERSION \
  "$@"
