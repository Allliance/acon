# Compressor-induced perplexity

Measures how much a context compressor changes the agent's perplexity on its own
next action when the prior context is **compressed** instead of given in
**full**. A lower induced perplexity means the compression preserved the
information the agent actually relied on.

## Method

For one trajectory (a saved full-context `llm_history.json`):

1. **Segment.** Reconstruct `(action, observation)` steps and greedily pack
   consecutive steps into segments of at most `--max_segment_tokens` (default
   6000), measured with the *agent's own* tokenizer.
2. **For each segment boundary `b`** (between segment `b` and `b+1`):
   - *prefix* = all steps in segments `S_0 .. S_b`.
   - *action* = the first assistant action of segment `S_{b+1}`.
   - Score the agent's mean per-token perplexity on the action tokens under:
     - **full** context: `system + task + prefix steps (verbatim)`
     - **compressed** context: `system + (task + <HISTORY_SUMMARY> of prefix)`
   - Induced perplexity at the boundary = `ppl_compressed - ppl_full`.
3. **Trajectory score** = mean induced perplexity over boundaries.
4. **Compressor score** = mean of trajectory scores over all trajectories.

The compressed context mirrors production (`history_summary_rule: reset` in
`agents/memory.py`): the summary is appended to the task message inside
`<HISTORY_SUMMARY>` tags. The compression itself reuses the project's
`HistoryOptimizer`, so any `configs/context_opt/*.yaml` compressor (LLM prompt,
LLMLingua, etc.) can be scored as-is.

`nll_diff` (mean negative log-likelihood difference) is reported alongside
`ppl_diff` and is the more outlier-robust quantity.

## Files

| file                | role |
|---------------------|------|
| `segmentation.py`   | load `llm_history.json`, build steps, token-bounded segments |
| `vllm_client.py`    | `/tokenize` + `/v1/completions` (echo) → action-token perplexity |
| `compressor.py`     | selection baselines + LLM summary (full / cumulative) + caching |
| `pipeline.py`       | per-boundary full-vs-compressed scoring + aggregation |
| `compress.py`       | CLI: generate + cache the cumulative compression chain |
| `run.py`            | CLI: perplexity scoring over a directory of `task_<id>/` trajectories |
| `selftest.py`       | offline wiring check (stub agent/compressor, no server) |

## Compressors

`make_compressor()` picks the family from the config:

* **Selection baselines** (`baseline_strategy: fifo|mask_obs|mask_action|random`)
  — no LLM; select/mask prefix turns to fit the budget.
* **LLM summary, full** (`summary_mode: full`, default) — re-summarize the whole
  prefix at each boundary.
* **LLM summary, cumulative** (`summary_mode: cumulative`) — recurrent chain:
  `summary_0 = LLM(segment_0)`, `summary_i = LLM(prev=summary_{i-1}, new=segment_i)`.
  At boundary `b` the context uses `summary_b`. The chain is generated **once per
  trajectory** and cached to `perplexity/compressions/<tag>/<task_id>.json`.

### Cumulative caching (reusable artifact)

The cache is the deliverable: each `<task_id>.json` stores the segmentation,
compressor metadata, and every cumulative summary (with input/prev/output token
counts). Generate it standalone:

```bash
python -m perplexity.compress \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressor_config configs/context_opt/qwen35a3b_self_cumulative_b2048.yaml \
  --agent_base_url http://<host>:8000/v1        # compressor shares the agent endpoint
```

`run.py` reuses the exact same cache (same default tag), so scoring re-runs cost
no extra LLM calls. The cache is regenerated only if the segmentation or budget
changes, or with `--overwrite_cache`. Cache tag = `<config_name>_t<max_segment_tokens>`.

## Perplexity scoring

Action perplexity is computed by teacher forcing on the vLLM server:

1. `/tokenize` renders `system + task + prefix` with the chat template and
   `add_generation_prompt=True`, returning the exact prompt token ids.
2. The action token ids (tokenized once, reused for both conditions) are
   appended.
3. `/v1/completions` with `echo=True` returns per-token log-probs; only the
   action tokens are averaged: `ppl = exp(mean(-logprob))`.

This needs no local tokenizer — the served model's template and tokenizer are
used directly. The action tokens are identical across the full and compressed
conditions, so only the conditioning context differs.

## Usage

Run inside the `smolagents` conda env (provides `yaml`, `openai`,
`transformers`, and `productive_agents`).

```bash
cd experiments/appworld
export VLLM_BASE_URL=http://<agent-host>:8000/v1            # the agent being measured
export VLLM_COMPRESSOR_BASE_URL=http://<compressor-host>:8000/v1

python -m perplexity.run \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressor_config configs/context_opt/qwen35a3b_self_cumulative_b2048.yaml \
  --agent_base_url http://<host>:8000/v1 \
  --output perplexity/outputs/qwen35a3b_cumulative_b2048.json
```

Naive baselines at a 2k budget (mask_obs / fifo / random) in one shot:

```bash
bash perplexity/run_baselines.sh
```

Offline smoke test (no server needed):

```bash
python -m perplexity.selftest \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev --limit 4
```

The output JSON has `summary.compressor_ppl_score` /
`compressor_nll_score` plus per-trajectory and per-boundary detail.
