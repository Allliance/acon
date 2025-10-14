#!/usr/bin/env bash
set -euo pipefail

# Smolagents ctxopt pipeline runner for history prompts.
# Usage: ./experiments/smolagents/scripts/run_ctxopt_history.sh [PROMPTS_DIR] [additional args passed to pipeline]
# Environment overrides:
#   MODEL (default: gpt-4.1)
#   TAG   (default auto-derived)
#   SPLIT (default: train)
#   CTXOPT_TYPE (forced to history unless overridden explicitly)
#   LIMIT (optional) -> passed as --limit

PROMPTS_DIR_DEFAULT="../prompt_optimizer/outputs_smolagents/history_regression/optimized_prompts"
PROMPTS_DIR="${1:-$PROMPTS_DIR_DEFAULT}"
if [[ $# -gt 0 ]]; then shift; fi

MODEL="${MODEL:-gpt-4.1}"
# Derive default tag from date
DATE_TAG="$(date +%y%m%d)"
TAG="${TAG:-${DATE_TAG}_prompteval_hist_opt_4096}"  # user can override
SPLIT="${SPLIT:-train}"
CTXOPT_TYPE="${CTXOPT_TYPE:-history}"
OPT_VERSION="${OPT_VERSION:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # experiments/smolagents
PIPELINE_PY="$SCRIPT_DIR/run_ctxopt_pipeline.py"

if [[ ! -f "$PIPELINE_PY" ]]; then
  echo "Cannot find pipeline script at $PIPELINE_PY" >&2
  exit 1
fi

if [[ ! -d "$PROMPTS_DIR" ]]; then
  echo "Prompts directory not found: $PROMPTS_DIR" >&2
  exit 1
fi

echo "Running smolagents ctxopt (history) pipeline"
echo "  prompts-dir: $PROMPTS_DIR"
echo "  model:       $MODEL"
echo "  tag:         $TAG"
echo "  split:       $SPLIT"
echo "  type:        $CTXOPT_TYPE"

PYTHON_BIN="$(command -v python3 || command -v python)"
CMD=("$PYTHON_BIN" "$PIPELINE_PY" \
  --prompts-dir "$PROMPTS_DIR" \
  --model-name "$MODEL" \
  --ctxopt-type "$CTXOPT_TYPE" \
  --tag "$TAG" \
  --split "$SPLIT" \
  --opt-version $OPT_VERSION \
  --id-list-file "data/nq_multi_8/folds/train_history_tiny_version2.txt"
)

if [[ -n "${LIMIT:-}" ]]; then
  CMD+=(--limit "$LIMIT")
fi

exec "${CMD[@]}" "$@"
