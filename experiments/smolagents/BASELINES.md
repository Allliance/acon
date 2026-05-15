# Compression Baselines — Usage Guide

Agent model for all runs below: **Qwen3.5-35B-A3B** served on `VLLM_PORT`.

There are two families of baselines.

## Family A — Prompting baselines (LLM-based compression)

Same naive-prompt history compression as the existing `gpt-4.1_history.yaml`,
but the compressor LLM is swapped out. The compressor talks to a **second**
vLLM server on `VLLM_COMPRESSOR_PORT` (default `8010`).

| Config | Compressor LLM |
| --- | --- |
| `configs/context_opt/qwen3.5-35b-a3b_history.yaml` | Qwen3.5-35B-A3B |
| `configs/context_opt/qwen3.5-27b_history.yaml`     | Qwen3.5-27B     |
| `configs/context_opt/qwen3.5-9b_history.yaml`      | Qwen3.5-9B      |
| `configs/context_opt/gemma-4-31b_history.yaml`     | Gemma-4-31B     |
| `configs/context_opt/gpt-oss-120b_history.yaml`    | GPT-OSS-120B (served via vLLM) |
| `configs/context_opt/gpt-4.1_history.yaml`         | GPT-4.1 (OpenAI API; original) |

If the compressor model equals the agent model, you can share one vLLM server
by leaving `COMPRESSOR_MODEL` empty (or pointing the compressor port at the
agent's port).

## Family B — Selection-based baselines (no LLM)

Pick / mask turns to fit a token budget. No compressor LLM needed; you can
omit `COMPRESSOR_MODEL` when launching.

| Config | Strategy |
| --- | --- |
| `configs/context_opt/fifo_budget2048.yaml`        | FIFO: keep newest turns within budget |
| `configs/context_opt/mask_obs_budget2048.yaml`    | Mask observation content, keep newest turns within budget |
| `configs/context_opt/mask_action_budget2048.yaml` | Mask action content, keep newest turns within budget |
| `configs/context_opt/random_budget2048.yaml`      | Random subsample within budget |

Tune `compression_budget` (tokens) in the config to vary the budget.

## Family C — ACON baselines already present

| Config | Strategy |
| --- | --- |
| `configs/context_opt/discard_history_keep5.yaml`   | Discard older turns, keep last k |
| `configs/context_opt/retrieval_history_keep5.yaml` | Embedding-based retrieval of past turns |
| `configs/context_opt/llmlingua_history.yaml`       | LLMLingua compression |

## How to run

All baselines go through the single unified `./launch.sh`. It auto-detects
running vLLM / retriever servers on ports 8000–8010, or pin endpoints with
`--vllm-url` / `--compressor-url` / `--retriever-url`. See `launch.sh` header
for full docs.

Defaults baked into the output tag: `t8k_b4k_w128` (threshold 8192 tokens,
budget 4096 tokens, 128 workers). Override with run.py's
`--history_threshold` / `--compression_budget` / `--num_workers`.

### Selection-based (no compressor LLM)

```bash
./launch.sh -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo
```

### Prompting baseline (separate compressor vLLM)

```bash
./launch.sh --compressor-model Qwen/Qwen3.5-9B \
    -- --co_config_path configs/context_opt/qwen3p5_9b_prompting_t8k_b4k.yaml --tag qwen9b
```

### Same model for agent and compressor (single vLLM server)

Point the compressor URL at the agent URL — no second server is launched:

```bash
./launch.sh --compressor-url http://localhost:8000 \
    -- --co_config_path configs/context_opt/qwen3p5_35b_a3b_prompting.yaml --tag qwen35b
```

### GPT-4.1 baseline (OpenAI API, no second vLLM)

```bash
./launch.sh -- --co_config_path configs/context_opt/gpt-4.1-mini_t8k_b4k.yaml --tag gpt41mini
```

### Reuse a remote vLLM

```bash
./launch.sh --vllm-url r818u33n08:8000 \
    -- --co_config_path configs/context_opt/fifo_t8k_b4k.yaml --tag fifo
```

## How the compressor port is resolved

Resolution order inside `HistoryOptimizer` / `ObservationOptimizer`:

1. `compressor_port` field in the YAML (if set).
2. `VLLM_COMPRESSOR_PORT` env var.
3. Falls back to `VLLM_PORT` (single shared vLLM server).

For OpenAI-routed models (`gpt-4.1`, `gpt-5`, etc.), the port is ignored.
`gpt-oss` is explicitly routed through vLLM (see `LLMManager.create_llm`).

## What changes in the code

- `llm.py::vLLM` now accepts a `port=` kwarg.
- `agents/utils.py::LLMManager.create_llm` plumbs `port=` through and routes
  `gpt-oss` to vLLM.
- `ctxopt/history_optimizer.py` and `ctxopt/obs_optimizer.py` read
  `compressor_port` / `VLLM_COMPRESSOR_PORT` and skip LLM init for
  selection-only baselines.
- `ctxopt/selection_strategies.py` implements FIFO / mask-obs / mask-action /
  random selection on `(assistant, user)` pairs against a token budget.
- `agents/memory.py::MemoryManager.optimize_history` dispatches the new
  `baseline_strategy` values.
