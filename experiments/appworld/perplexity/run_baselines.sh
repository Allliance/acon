#!/usr/bin/env bash
# Naive-baseline perplexity sweep (mask_obs, fifo, random) at a 2k budget.
# Usage: bash perplexity/run_baselines.sh
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

declare -A CONFIGS=(
  [mask_obs]=configs/context_opt/mask_obs_t6k_b2k.yaml
  [fifo]=configs/context_opt/fifo_t8k_b4k.yaml
  [random]=configs/context_opt/random_t6k_b2k.yaml
)

for name in mask_obs fifo random; do
  echo "=================== $name (budget=$BUDGET) ==================="
  "$PY" -m perplexity.run \
    --trajectory_dir "$TRAJ_DIR" \
    --compressor_config "${CONFIGS[$name]}" \
    --compression_budget "$BUDGET" \
    --agent_base_url "$AGENT_BASE_URL" \
    --agent_model "$AGENT_MODEL" \
    --output "$OUT_DIR/${name}_b${BUDGET}.json"
done

echo "=================== DONE ==================="
