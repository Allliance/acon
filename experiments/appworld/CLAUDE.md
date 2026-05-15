# AppWorld experiments — local notes for Claude

Quick orientation that complements `/gpfs/radev/project/cohan/aa3242/acon/CLAUDE.md`.

## Layout

```
experiments/appworld/
  run.py                       # runs ONE task
  run_all.py                   # runs every task in a split (calls run.main)
  run_parallel.py              # parallel split runner; auto-runs evaluate.py
  evaluate.py                  # OFFICIAL appworld eval -> summary.jsonl
  run_qwen_appworld.sh         # driver for remote-vLLM (Qwen) runs, 3 reps
  configs/
    base_config.yaml           # default agent params (max_iter, prompt_file)
    context_opt/               # compression configs (history / obs)
  prompts/prompts_v1.json      # AppWorld agent prompt
  data/                        # downloaded with `appworld download data`
  experiments/outputs/<exp>/   # CANONICAL world state appworld evaluate reads
      evaluations/<split>.json # official report (task/scenario goal completion)
  outputs/<exp>/               # <exp> = <safe_model>_<tag>  (tag includes _repK)
      summary.jsonl            # one line/split: official TGC/SGC + cost/tokens
      <split>/
        experiment_summary.json# generation-side stats (internal success only)
        task_<id>_<rep>/       # per-task trajectories
```

`<safe_model>` is `model_name` with `/` → `_` (e.g. `Qwen_Qwen3.5-35B-A3B`).

## Evaluation (IMPORTANT)

`experiment_summary.json`'s `success_rate` only means "agent terminated
without crashing" — it overstates real performance ~2-3x. The trustworthy
metric is **TGC** (task_goal_completion) / **SGC** (scenario_goal_completion)
from the official `appworld evaluate`, which re-runs hidden unit tests against
the saved world state.

- `run_parallel.py` runs the agent then **automatically** calls
  `evaluate.evaluate_and_summarize()`, writing/upserting
  `outputs/<exp>/summary.jsonl` (keyed by `split`).
- Re-evaluate without re-generating:
  `python evaluate.py --experiment_name <exp> --split test_normal`
- Default split is **test_normal only** (`SPLITS` in the driver). The driver's
  FINAL SUMMARY now averages official TGC across reps, not `success_rate`.
- Must run inside the `smolagents` env (needs the `appworld` CLI on PATH).

## Local conda env

Env: `smolagents` at `/gpfs/radev/home/aa3242/scratch/envs/smolagents` —
has `productive_agents` (editable from this repo), `appworld` (editable from
`~/scratch/appworld`), `git-lfs` (via conda-forge), and the OpenAI client used
by `productive_agents.llm.vLLM`.

The appworld repo was cloned to `~/scratch/appworld` (needed for the
git-lfs–pinned `apps.bundle` / `tests.bundle`). `appworld install --repo`
and `appworld download data` were run there; `data/` was moved into this
folder so the runners resolve paths.

## Remote vLLM wiring

`productive_agents.llm.vLLM` reads endpoint from `VLLM_BASE_URL` env var
(falls back to `localhost:8000`). The driver script sets it to
`http://r818u33n06:8000/v1`. Anything whose `model_name` doesn't match
`gpt|o1|o3|o4` routes to vLLM via `LLMManager.create_llm`. For Qwen3
the server emits a `<think>...</think>` block in `reasoning_content`; the
client already handles `content == None`.

## Splits

- `dev` (56 tasks)            ← "easy"  (skipped by default)
- `test_normal` (167 tasks)   ← "normal" — **the only split run by default**
- `test_challenge` (416 tasks) ← "hard"  (skipped)
- `train` (89 tasks)          ← not used for evaluation

## Reruns / resuming

`run_all.py` *skips* a task whose `task_<id>` output dir already exists
unless `--continue_existing` or `--rerun_failed` is passed. To redo a rep
cleanly, delete the per-rep output dir first.

## Common gotchas

- `private_config.yaml` needs to exist (even with a placeholder key) — agent
  initialization touches `load_openai_key_from_config()` regardless of model.
  We copied the dummy file in `configs/`.
- `--debug` in `run_all.py` runs ONE task but still writes a summary with
  `total_tasks = len(split)`, so `success_rate` will look like 1/56.
- Pricing table in `llm.py` has no Qwen entry, so cost prints as $0 —
  expected.
