#!/usr/bin/env bash
set -euo pipefail

# OfficeBench ctxopt pipeline runner (history prompts)
# Usage: ./experiments/officebench/scripts/run_ctxopt_history.sh [PROMPTS_DIR] [extra args passed to pipeline]
# Environment overrides:
#   MODEL (default: gpt-4.1)
#   TAG (default: <YYMMDD>_officebench_hist_ctxopt)
#   SPLIT (default: train)
#   HISTORY_THRESHOLD (default: 4096)
#   PRESERVE_LAST_K (default: 1)
#   OPT_VERSION (default: 1)
#   LIMIT_CONFIGS (optional)
#   DEBUG=1 to add --debug
# Example:
#   MODEL=gpt-4o TAG=250901_hist_opt ./experiments/officebench/scripts/run_ctxopt_history.sh

PROMPTS_DIR_DEFAULT="$(cd "$(dirname "$0")/.." && pwd)/prompts/context_opt"
PROMPTS_DIR="${1:-$PROMPTS_DIR_DEFAULT}"
if [[ $# -gt 0 ]]; then shift; fi

MODEL="${MODEL:-gpt-4.1}"
DATE_TAG="$(date +%y%m%d)"
TAG="${TAG:-${DATE_TAG}_officebench_hist_ctxopt}"
SPLIT="${SPLIT:-train_history_tiny}"
HISTORY_THRESHOLD="${HISTORY_THRESHOLD:-4096}"
PRESERVE_LAST_K="${PRESERVE_LAST_K:-1}"
OPT_VERSION="${OPT_VERSION:-1}"
LIMIT_CONFIGS="${LIMIT_CONFIGS:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # experiments/officebench
PIPELINE_PY="$SCRIPT_DIR/run_ctxopt_pipeline.py"

if [[ ! -f "$PIPELINE_PY" ]]; then
  echo "Cannot find pipeline script at $PIPELINE_PY" >&2
  exit 1
fi
if [[ ! -d "$PROMPTS_DIR" ]]; then
  echo "Prompts directory not found: $PROMPTS_DIR" >&2
  exit 1
fi

echo "Running OfficeBench ctxopt (history) pipeline"
echo "  prompts-dir: $PROMPTS_DIR"
echo "  model:       $MODEL"
echo "  tag:         $TAG"
echo "  split:       $SPLIT"
echo "  threshold:   $HISTORY_THRESHOLD"
echo "  preserve_k:  $PRESERVE_LAST_K"

PYTHON_BIN="$(command -v python3 || command -v python)"
CMD=("$PYTHON_BIN" "$PIPELINE_PY" \
  --prompts-dir "$PROMPTS_DIR" \
  --model-name "$MODEL" \
  --ctxopt-type history \
  --tag "$TAG" \
  --split "$SPLIT" \
  --history-threshold "$HISTORY_THRESHOLD" \
  --preserve-last-k-turns "$PRESERVE_LAST_K" \
  --opt-version "$OPT_VERSION" --run-eval
)
if [[ -n "$LIMIT_CONFIGS" ]]; then
  CMD+=(--limit-configs "$LIMIT_CONFIGS")
fi
if [[ "${DEBUG:-}" == "1" ]]; then
  CMD+=(--debug)
fi

exec "${CMD[@]}" "$@"
