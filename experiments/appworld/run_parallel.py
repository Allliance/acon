#!/usr/bin/env python3
"""
Parallel AppWorld runner.

Runs `run.main(...)` across many tasks in a thread pool. Each task is
embarrassingly parallel — the LLM calls go over HTTP to the shared vLLM
server, and AppWorld itself runs in-process so there's no port conflict.

Use this instead of `run_all.py` when you want N concurrent workers
hammering a single vLLM endpoint.

Output layout matches `run_all.py`:
    outputs/<safe_model>_<tag>/<split>/
        experiment_summary.json
        task_<id>_<rep>/...
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# NOTE: we intentionally do *not* import `run.main` at module top level,
# because each worker process imports this file fresh; the heavy imports
# (appworld, productive_agents) happen inside the worker function so they
# pay the import cost once-per-process, not once-per-task.


def _worker_run_one(task_id: str, kwargs: dict):
    """Top-level (picklable) worker. Imports happen inside the child process."""
    import os
    import logging as _logging

    # Quiet down per-task spam in worker processes.
    _logging.basicConfig(level=_logging.WARNING)
    for n in ["httpx", "httpcore", "openai", "llm", "productive_agents"]:
        _logging.getLogger(n).setLevel(_logging.ERROR)

    from run import main as run_one  # heavy imports happen here, once per worker

    task_out = os.path.join(kwargs["output_root"], f"task_{task_id}")
    try:
        result = run_one(
            task_id=task_id,
            split=kwargs["split"],
            output_dir=task_out,
            exp_config=kwargs["exp_config"],
            model_name=kwargs["model_name"],
            debug_mode=False,
            experiment_name=kwargs["experiment_name"],
            max_iter=kwargs["max_iter"],
            model_ctxopt=None,
            lora_name=None,
        )
        return task_id, result, None
    except Exception as e:  # noqa: BLE001
        return task_id, None, repr(e)


def parse_args():
    p = argparse.ArgumentParser(description="Parallel AppWorld task runner")
    p.add_argument("--split", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--co_config_path", default=None)
    p.add_argument("--max_iter", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument(
        "--history_threshold",
        type=int,
        default=8192,
        help="Token threshold that triggers history compression. Overrides the "
        "co_config; non-default values are baked into the output tag (e.g. t4k).",
    )
    p.add_argument(
        "--compression_budget",
        type=int,
        default=4096,
        help="Target token budget for the compressed history. Overrides the "
        "co_config; non-default values are baked into the output tag (e.g. b2k).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks of the split (smoke testing). The "
        "official evaluator also only scores this subset.",
    )
    p.add_argument(
        "--continue_existing",
        action="store_true",
        help="Re-run tasks even if their output dir already exists.",
    )
    return p.parse_args()


def _fmt_k(n: int | None) -> str:
    """Format a token count compactly: 8192 -> '8k', 4096 -> '4k', 1500 -> '1500'."""
    if n is None:
        return "x"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}k"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def load_base_config(tag: str, model_name: str, co_config: dict | None) -> dict:
    base_path = Path("configs/base_config.yaml")
    config = yaml.safe_load(open(base_path)) if base_path.exists() else {}
    config.update(
        {
            "exp_id": f"{tag}_{model_name}",
            "model_name": model_name,
            "tag": tag,
            "max_iter": 50,
            "use_workflow_memory": False,
            "use_thinking_tokens": True,
            "prompt_file": "./prompts/prompts_v1.json",
            "co_config": co_config,
            "experiment_name": f"experiment_{tag}",
        }
    )
    return config


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ["httpx", "httpcore", "openai", "azure.identity", "azure.core"]:
        logging.getLogger(n).setLevel(logging.ERROR)

    co_config = None
    if args.co_config_path:
        with open(args.co_config_path) as f:
            co_config = yaml.safe_load(f)

    # Override threshold/budget from the CLI so the run is reproducible even
    # when the config omits them. Defaults are 8192 / 4096 (same as the
    # *_t8k_b4k configs), so a default run leaves the config values intact.
    if co_config is not None:
        co_config["history_summarization_threshold"] = args.history_threshold
        co_config["obs_summarization_threshold"] = args.history_threshold
        co_config["compression_budget"] = args.compression_budget

    # Bake non-default threshold/budget into the tag so distinct sweeps land in
    # distinct output dirs. Default (8k / 4k) keeps the bare tag unchanged.
    param_bits = []
    if args.history_threshold != 8192:
        param_bits.append(f"t{_fmt_k(args.history_threshold)}")
    if args.compression_budget != 4096:
        param_bits.append(f"b{_fmt_k(args.compression_budget)}")
    eff_tag = args.tag + ("_" + "_".join(param_bits) if param_bits else "")

    exp_config = load_base_config(eff_tag, args.model_name, co_config)
    safe_model = args.model_name.replace("/", "_")
    experiment_name = f"{safe_model}_{eff_tag}"
    exp_config.update(
        {
            "max_iter": args.max_iter,
            "experiment_name": experiment_name,
            "debug_mode": False,
            "co_config_path": args.co_config_path,
            "seed": args.seed,
        }
    )

    # Load task list via AppWorld's loader.
    from appworld import load_task_ids

    task_ids = load_task_ids(args.split)
    if args.limit is not None:
        # Smoke-test knob: restrict the whole task set (so the official
        # evaluator also only scores this subset).
        task_ids = task_ids[: args.limit]
    output_root = f"./outputs/{experiment_name}/{args.split}"
    os.makedirs(output_root, exist_ok=True)

    # Filter out already-done tasks unless we're explicitly continuing.
    pending = []
    for tid in task_ids:
        out_dir = os.path.join(output_root, f"task_{tid}")
        if not args.continue_existing and os.path.exists(out_dir):
            continue
        pending.append(tid)

    print(
        f"[parallel] split={args.split} model={args.model_name} tag={args.tag} "
        f"workers={args.num_workers} pending={len(pending)}/{len(task_ids)}"
    )
    if not pending:
        print("[parallel] nothing to do; all task dirs already exist.")
        return

    successes: list[str] = []
    failures: list[str] = []
    task_costs: dict[str, float] = {}
    total_cost = 0.0
    total_in = total_out = total_req = 0

    start = time.time()

    # Submit via top-level worker so ProcessPoolExecutor can pickle the call.
    submit_kwargs = dict(
        split=args.split,
        output_root=output_root,
        exp_config=exp_config,
        model_name=args.model_name,
        experiment_name=experiment_name,
        max_iter=args.max_iter,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("ok:{task.fields[ok]} fail:{task.fields[fail]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        pb = progress.add_task(
            f"{args.split}", total=len(pending), ok=0, fail=0
        )

        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(_worker_run_one, tid, submit_kwargs): tid
                for tid in pending
            }
            for fut in as_completed(futures):
                tid, result, err = fut.result()
                if err is not None:
                    failures.append(tid)
                    print(f"[parallel] task {tid} raised: {err}", file=sys.stderr)
                else:
                    tok = result.get("token_usage") or {}
                    cost = tok.get("total_cost_usd", 0.0)
                    task_costs[tid] = cost
                    total_cost += cost
                    total_in += tok.get("total_input_tokens", 0)
                    total_out += tok.get("total_output_tokens", 0)
                    total_req += tok.get("total_requests", 0)
                    if result.get("success", False):
                        successes.append(tid)
                    else:
                        failures.append(tid)
                progress.update(
                    pb, advance=1, ok=len(successes), fail=len(failures)
                )

    elapsed = time.time() - start
    total_seen = len(successes) + len(failures)
    summary = {
        "experiment_config": exp_config,
        "split": args.split,
        "total_tasks": len(task_ids),
        "ran_tasks": total_seen,
        "successful_tasks": successes,
        "failed_tasks": failures,
        "success_rate": (len(successes) / len(task_ids)) if task_ids else 0.0,
        "success_rate_over_ran": (len(successes) / total_seen) if total_seen else 0.0,
        "total_time_seconds": elapsed,
        "run_timestamp": time.time(),
        "num_workers": args.num_workers,
        "cost_summary": {
            "total_cost_usd": total_cost,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_requests": total_req,
            "task_costs": task_costs,
        },
    }

    sp = os.path.join(output_root, "experiment_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n[parallel] split={args.split}  internal_success={len(successes)}/{len(task_ids)} "
        f"({summary['success_rate']*100:.2f}%)  "
        f"failed={len(failures)}  "
        f"elapsed={int(elapsed//60)}m{int(elapsed%60)}s  "
        f"summary={sp}"
    )

    # ---- Official AppWorld evaluation (the real correctness signal) ---------
    # The internal success_rate above only means "agent terminated without
    # crashing"; it overstates real performance a lot. Run the real evaluator
    # and write outputs/<experiment_name>/summary.jsonl.
    #
    # `appworld evaluate` always scores the FULL split, so it cannot run on a
    # --limit subset (it would look for DBs of tasks we never ran). Skip it.
    if args.limit is not None:
        print(
            f"[parallel] skipping official eval: --limit {args.limit} ran a "
            f"subset; `appworld evaluate` requires the full split."
        )
        return
    try:
        from evaluate import evaluate_and_summarize

        rec = evaluate_and_summarize(experiment_name, args.split, root=".")
        jsonl = os.path.join("outputs", experiment_name, "summary.jsonl")
        print(
            f"[parallel] OFFICIAL eval split={args.split}  "
            f"TGC={rec['task_goal_completion']}%  "
            f"SGC={rec['scenario_goal_completion']}%  "
            f"({rec['num_tasks_passed']}/{rec['num_tasks_evaluated']} tasks)  "
            f"-> {jsonl}"
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[parallel] WARNING: official evaluation failed for "
            f"{experiment_name}/{args.split}: {e!r}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
