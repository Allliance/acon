# Trajectory divergence

Measures how much an agent's **near-future plan** changes when it is conditioned
on a **compressed** prefix instead of the **full** history. A high score means
the compression preserved the information the agent actually relied on to decide
its next steps; a low score means the compression pushed the agent off its
trajectory.

This package is a standalone copy of the divergence code (split out of
`perplexity/` so it can be used in production). It depends only on the
`productive_agents` package (for `ctxopt.selection_strategies` and, for the
LLM-summary baselines, `ctxopt.history_optimizer`) and a vLLM-served agent plus
a Gemini judge.

## Method

For one trajectory (a saved full-context `llm_history.json`):

1. **Segment.** Reconstruct `(action, observation)` steps and greedily pack them
   into token-bounded segments (the segmentation is reused from the cached
   `compressions/<tag>/<task>.json` artifacts so every runner scores the *same*
   boundaries).
2. **For each boundary `i`** (segments `0..i-1` are the *old prefix*, segment
   `S_i` is kept *verbatim*):
   - Build the agent context for a condition (see below), then append a
     meta-instruction asking the agent to forecast its next `N` (=5) code blocks
     in **one shot**, with no intervening execution outputs.
   - Parse the `N` predicted blocks and ask the judge how close they are, as a
     plan, to the `N` actions the agent **actually** took next (0-100).
3. **Condition score** = mean judge score over all boundaries.

### Conditions

| condition | prefix the agent sees |
|-----------|-----------------------|
| `self` | full real prefix, verbatim (self-consistency control) |
| `cumulative` | recurrent LLM summary of the prefix (cached) |
| `fifo` / `random` / `mask_obs` | selection-compressed prefix to fit a token budget |
| full-mode baselines (`35b_*`, `27b_*`, `9b_*`, `4b_*`, `120b_*`) | **non-recurrent** LLM re-summary of the whole prefix |

In every condition the recent segment `S_i` is appended verbatim, and the agent
scores the same real next-`N` actions, so conditions differ only in how the old
prefix is represented.

## Files

| file | role |
|------|------|
| `core.py` | context construction + plan instruction + plan parsing (the metric primitives) |
| `judge.py` | `GeminiJudge` — 0-100 plan-similarity scoring with a strict rubric |
| `segmentation.py` | load `llm_history.json` → steps; token-bounded segmentation |
| `vllm_client.py` | OpenAI-compatible chat client (used to sample the agent's plan) |
| `compressor.py` | LLM-summary + selection compressors (only `run_baselines` needs it) |
| `run.py` | self + cumulative + selection, reusing cached cumulative summaries |
| `run_baselines.py` | full-mode (non-recurrent) LLM-summary compressor baselines |
| `run_naive.py` | selection-only baselines, with an agent-decoding sweep |
| `run_resample.py` | re-decode LLM compressors reusing cached summaries (judge + agent only) |
| `score_next1.py` | add a next-1-action score to an existing cache (judge-only) |

## Cached artifacts (copied in)

- `compressions/<tag>/<task>.json` — segmentation + cumulative summaries (input).
- `outputs/divergence_cache*/` — per-cell plan + judge caches (resumable; each
  runner's default `--cache_dir` points here, so re-runs are free).
- `outputs/divergence_*_b2048.json` — aggregated reports.

## Usage

Run inside the `smolagents` conda env (provides `yaml`, `openai`, `google-genai`,
and `productive_agents`). The Gemini key is read from
`configs/private_config.yaml`.

```bash
cd experiments/appworld

# self + cumulative + selection (reuses cached cumulative summaries)
export VLLM_BASE_URL=http://<agent-host>:8000/v1
python -m divergence.run \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
  --output divergence/outputs/divergence_b2048.json

# full-mode LLM-summary compressor baselines (12 cells)
python -m divergence.run_baselines \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
  --agent_base_url http://<agent-host>:8000/v1 \
  --compressor_base_url http://<27b-host>:8000/v1 \
  --output divergence/outputs/divergence_fullbaselines_b2048.json

# naive selection baselines, sampled decoding
python -m divergence.run_naive \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
  --agent_base_url http://<agent-host>:8000/v1 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 \
  --presence_penalty 0.0 --repetition_penalty 1.0 \
  --tag sampled --output divergence/outputs/divergence_naive_sampled_b2048.json

# re-score the LLM compressors under a different decoding regime (no compressor calls)
python -m divergence.run_resample \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
  --source_cache_dir divergence/outputs/divergence_cache_fullbaselines/qwen35a3b_self_cumulative_b2048_t6000 \
  --agent_base_url http://<agent-host>:8000/v1 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 \
  --tag sampled

# add a next-1-action score to an existing cache (judge-only, no agent calls)
python -m divergence.score_next1 \
  --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
  --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
  --output divergence/outputs/divergence_b2048.json
```

Higher mean judge score = less divergence. `self` is the self-consistency
ceiling each compressor is compared against.
