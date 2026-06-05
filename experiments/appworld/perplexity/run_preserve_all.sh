#!/usr/bin/env bash
# Perplexity sweep under the preserve-recent-segment scheme, for all four
# compressors at a 2k budget. Reports next-1 and next-5 action PPL/NLL.
# The full-context action scores are compressor-independent and cached on disk,
# so running these sequentially makes baselines 2-4 nearly free.
#
# Usage: bash perplexity/run_preserve_all.sh
set -euo pipefail

PY=/gpfs/radev/home/aa3242/scratch/envs/smolagents/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # experiments/appworld
cd "$HERE"

AGENT_BASE_URL="${AGENT_BASE_URL:-http://r4519u01n01:8000/v1}"
AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3.5-35B-A3B}"
TRAJ_DIR="${TRAJ_DIR:-trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev}"
BUDGET="${BUDGET:-2048}"
OUT_DIR="perplexity/outputs"
mkdir -p "$OUT_DIR"

# name -> "config[:budget_flag]". Cumulative uses its config budget (cached);
# selection baselines get an explicit 2k override.
run_one () {
  local name="$1" config="$2" budget_override="$3"
  echo "=================== $name (preserve, budget=$BUDGET) ==================="
  local extra=()
  if [[ -n "$budget_override" ]]; then extra=(--compression_budget "$BUDGET"); fi
  "$PY" -m perplexity.run \
    --trajectory_dir "$TRAJ_DIR" \
    --compressor_config "$config" \
    --agent_base_url "$AGENT_BASE_URL" \
    --agent_model "$AGENT_MODEL" \
    --max_next_actions 5 \
    "${extra[@]}" \
    --output "$OUT_DIR/${name}_preserve_b${BUDGET}.json"
}

# Cumulative first: it fully populates the shared full-context cache.
run_one cumulative configs/context_opt/qwen35a3b_self_cumulative_b2048.yaml ""
run_one mask_obs   configs/context_opt/mask_obs_t6k_b2k.yaml                 override
run_one fifo       configs/context_opt/fifo_t8k_b4k.yaml                     override
run_one random     configs/context_opt/random_t6k_b2k.yaml                   override

echo "=================== DONE ==================="
