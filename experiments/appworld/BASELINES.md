# AppWorld Compression Baselines — Usage Guide

The AppWorld port of `experiments/smolagents/BASELINES.md`. Same baseline
families and the same shared code path (`MemoryManager` + `ctxopt/`); only the
runner and eval differ (AppWorld runs in-process, has no retriever, and the
trustworthy metric is the official **TGC**, not internal `success_rate`).

Agent model for the open-weights runs: **Qwen3.5-35B-A3B** served on a vLLM
endpoint. Defaults baked into every `*_t8k_b4k.yaml`: **threshold 8192 tokens,
budget 4096 tokens**. Override per run with `--history_threshold` /
`--compression_budget` (non-default values are appended to the output tag,
e.g. `t4k_b2k`).

## Family A — Prompting baselines (LLM-based compression)

Naive-prompt history compression where the compressor LLM is swapped out. The
compressor talks to a **second** vLLM on `VLLM_COMPRESSOR_PORT` (default 8010),
unless the compressor model equals the agent model (point `--compressor-url`
at the agent endpoint to share one server).

| Config | Compressor LLM |
| --- | --- |
| `configs/context_opt/qwen3p5_4b_prompting_t8k_b4k.yaml`     | Qwen3.5-4B |
| `configs/context_opt/qwen3p5_9b_prompting_t8k_b4k.yaml`     | Qwen3.5-9B |
| `configs/context_opt/qwen3p5_27b_prompting_t8k_b4k.yaml`    | Qwen3.5-27B |
| `configs/context_opt/qwen3p5_35b_a3b_prompting_t8k_b4k.yaml`| Qwen3.5-35B-A3B (share agent server) |
| `configs/context_opt/gpt-4.1-mini_t8k_b4k.yaml`             | GPT-4.1-mini (OpenAI API) |
| `configs/context_opt/gpt-4.1_t8k_b4k.yaml`                  | GPT-4.1 (OpenAI API) |

## Family B — Selection-based baselines (no LLM)

Pick / mask turns to fit a token budget. No compressor LLM — omit
`--compressor-model`.

| Config | Strategy |
| --- | --- |
| `configs/context_opt/fifo_t8k_b4k.yaml`        | FIFO: keep newest turns within budget |
| `configs/context_opt/mask_obs_t8k_b4k.yaml`    | Mask observation content, keep newest turns within budget |
| `configs/context_opt/mask_action_t8k_b4k.yaml` | Mask action content, keep newest turns within budget |
| `configs/context_opt/random_t8k_b4k.yaml`      | Random subsample within budget |

## Family C — ACON / other baselines

| Config | Strategy |
| --- | --- |
| `configs/context_opt/discard_history_keep5.yaml` | Discard older turns, keep last 5 |
| `configs/context_opt/llmlingua_t8k_b4k.yaml`     | LLMLingua-2 compression (HTTP endpoint) |

(`retrieve` is not ported: AppWorld has no embedding endpoint configured here.)

## How to run

Everything goes through `./launch.sh`, which resolves the agent vLLM (and a
compressor vLLM when needed), exports the endpoint env vars, then runs
`run_parallel.py` for `NUM_REPS` reps over `SPLITS` and aggregates the official
AppWorld TGC across reps. See the `launch.sh` header for the full flag/env list.

Run-loop env knobs: `TAG` (default `appworld_baseline`), `NUM_REPS` (3),
`SPLITS` (`test_normal`), `MAX_ITER` (50), `NUM_WORKERS` (128), `SEED` (42),
`CONDA_ENV` (`smolagents`). Splits: `dev` (56), `test_normal` (167, default),
`test_challenge` (416), `train` (89).

Anything after `--` is forwarded to `run_parallel.py`; the launcher injects
`--split / --model_name / --tag / --num_workers / --max_iter / --seed`, so you
only pass `--co_config_path` (+ optional `--history_threshold` /
`--compression_budget` / `--continue_existing`).

### Selection-based (no compressor LLM)

```bash
TAG=fifo ./launch.sh -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml
```

### GPT-4.1-mini compressor (OpenAI API, no compressor vLLM)

```bash
TAG=gpt41mini ./launch.sh -- --co_config_path configs/context_opt/gpt-4.1-mini_t8k_b4k.yaml
```

### Qwen compressor (spawns / reuses a second vLLM)

```bash
TAG=qwen9b ./launch.sh --compressor-model Qwen/Qwen3.5-9B \
    -- --co_config_path configs/context_opt/qwen3p5_9b_prompting_t8k_b4k.yaml
```

### Same model for agent and compressor (one shared vLLM)

```bash
TAG=qwen35b ./launch.sh --compressor-url http://localhost:8000 \
    -- --co_config_path configs/context_opt/qwen3p5_35b_a3b_prompting_t8k_b4k.yaml
```

### Reuse a remote agent vLLM on another node

```bash
TAG=fifo ./launch.sh --vllm-url http://r818u33n08:8000 \
    -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml
```

### LLMLingua-2 baseline

Start the (shared) LLMLingua server first, then point the run at it:

```bash
conda run -n <env-with-llmlingua> python tools/llmlingua_server.py --port 9999 &
LLMLINGUA_BASE_URL=http://localhost:9999 \
TAG=llmlingua ./launch.sh -- --co_config_path configs/context_opt/llmlingua_t8k_b4k.yaml
```

### Sweeping threshold / budget

```bash
TAG=fifo ./launch.sh -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml \
    --history_threshold 4096 --compression_budget 2048
# -> outputs/<safe_model>_fifo_rep1_t4k_b2k/...
```

## How endpoints are resolved

Resolution order inside `HistoryOptimizer` / `ObservationOptimizer`:

1. `compressor_port` in the YAML (if set).
2. `VLLM_COMPRESSOR_BASE_URL` / `VLLM_COMPRESSOR_PORT` env vars (set by `launch.sh`).
3. Falls back to the agent's `VLLM_PORT` (single shared server).

OpenAI-routed compressors (`gpt-4.1`, `gpt-4.1-mini`) ignore the port and use
the OpenAI API. LLMLingua uses `LLMLINGUA_BASE_URL` (default
`http://localhost:9999`).

## Output layout & evaluation

```
outputs/<safe_model>_<TAG>_rep<K>[_t..][_b..]/
    summary.jsonl                 # official TGC/SGC per split (the real metric)
    <split>/
      experiment_summary.json     # internal success only — OVERSTATES ~2-3x
      task_<id>/                  # per-task trajectories
```

`run_parallel.py` runs the agent then **automatically** calls the official
`appworld evaluate` (`evaluate.evaluate_and_summarize`). `launch.sh` reads
`summary.jsonl`, prints per-rep TGC, and reports the mean TGC across reps. Must
run inside the `smolagents` conda env (needs the `appworld` CLI on PATH).
