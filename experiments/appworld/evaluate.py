#!/usr/bin/env python3
"""
Official AppWorld evaluation + summary writer.

This wraps AppWorld's *real* evaluator (`appworld evaluate`), which re-runs the
hidden unit tests against the world state the agent left behind. That is the
only trustworthy correctness signal — the `success` flag in
`experiment_summary.json` merely records that the agent *terminated by calling
task-complete without crashing*, which massively overstates real performance.

It is used two ways:

  * automatically — `run_parallel.py` calls `evaluate_and_summarize()` right
    after a split finishes, so generation -> eval -> summary is one workflow.
  * standalone    — re-evaluate an already-generated experiment without
    re-running the agent:

        python evaluate.py --experiment_name <name> --split test_normal

Headline statistics are written (one JSON object per split, upserted) to:

    outputs/<experiment_name>/summary.jsonl

The official evaluator also drops its own reports at
`experiments/outputs/<experiment_name>/evaluations/<split>.{json,txt}`.

NOTE: `appworld` must be importable on PATH, i.e. run inside the `smolagents`
conda env (`source activate smolagents`).
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def run_official_evaluator(experiment_name: str, split: str, root: str = ".") -> dict:
    """Invoke `appworld evaluate` and return the parsed JSON report.

    The evaluator reads the canonical world state from
    `<root>/experiments/outputs/<experiment_name>/` and writes its report to
    `<root>/experiments/outputs/<experiment_name>/evaluations/<split>.json`.
    """
    cmd = ["appworld", "evaluate", experiment_name, split, "--root", root]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    report_path = (
        Path(root)
        / "experiments"
        / "outputs"
        / experiment_name
        / "evaluations"
        / f"{split}.json"
    )
    if not report_path.exists():
        raise RuntimeError(
            f"`{' '.join(cmd)}` produced no report (rc={proc.returncode}).\n"
            f"--- stdout (tail) ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
        )
    with open(report_path) as f:
        return json.load(f)


def compute_stats(report: dict) -> dict:
    """Derive headline + per-difficulty stats from the evaluator report.

    `task_goal_completion` / `scenario_goal_completion` are taken verbatim from
    the evaluator's authoritative `aggregate` block. Per-difficulty TGC and the
    pass/fail counts are derived from the per-task `individual` block.
    """
    agg = report["aggregate"]
    ind = report.get("individual", {})

    n_total = len(ind)
    n_pass = sum(1 for v in ind.values() if v.get("success"))

    by_diff_total: dict = defaultdict(int)
    by_diff_pass: dict = defaultdict(int)
    scen_tasks: dict = defaultdict(list)  # scenario = id-prefix before first "_"
    for tid, v in ind.items():
        d = v.get("difficulty")
        by_diff_total[d] += 1
        by_diff_pass[d] += int(bool(v.get("success")))
        scen_tasks[tid.split("_")[0]].append(v)

    tgc_by_difficulty = {
        str(d): round(100.0 * by_diff_pass[d] / by_diff_total[d], 2)
        for d in sorted(by_diff_total, key=lambda x: (x is None, x))
    }
    n_scen = len(scen_tasks)
    n_scen_pass = sum(
        1 for ts in scen_tasks.values() if ts and all(t.get("success") for t in ts)
    )

    return {
        "task_goal_completion": agg["task_goal_completion"],
        "scenario_goal_completion": agg["scenario_goal_completion"],
        "num_tasks_evaluated": n_total,
        "num_tasks_passed": n_pass,
        "num_tasks_failed": n_total - n_pass,
        "num_scenarios": n_scen,
        "num_scenarios_passed": n_scen_pass,
        "tgc_by_difficulty": tgc_by_difficulty,
    }


def _load_run_summary(split_dir: Path) -> dict:
    """Pull cost/token/runtime context from the generation-side summary."""
    f = split_dir / "experiment_summary.json"
    if not f.exists():
        return {}
    d = json.load(open(f))
    cost = d.get("cost_summary", {})
    return {
        "internal_success_rate": d.get("success_rate"),
        "ran_tasks": d.get("ran_tasks"),
        "total_tasks": d.get("total_tasks"),
        "run_time_seconds": d.get("total_time_seconds"),
        "num_workers": d.get("num_workers"),
        "seed": (d.get("experiment_config") or {}).get("seed"),
        "model_name": (d.get("experiment_config") or {}).get("model_name"),
        "total_cost_usd": cost.get("total_cost_usd"),
        "total_input_tokens": cost.get("total_input_tokens"),
        "total_output_tokens": cost.get("total_output_tokens"),
        "total_requests": cost.get("total_requests"),
    }


def _upsert_jsonl(path: Path, record: dict, key: str = "split") -> None:
    """Append `record`, replacing any existing line with the same `key` value."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get(key) != record.get(key):
                rows.append(row)
    rows.append(record)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def evaluate_and_summarize(
    experiment_name: str,
    split: str,
    root: str = ".",
    extra: dict | None = None,
) -> dict:
    """Run the official evaluator and upsert `outputs/<exp>/summary.jsonl`.

    Returns the record that was written.
    """
    report = run_official_evaluator(experiment_name, split, root)
    stats = compute_stats(report)

    output_root = Path(root) / "outputs" / experiment_name
    run_stats = _load_run_summary(output_root / split)

    record = {
        "experiment_name": experiment_name,
        "split": split,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        **stats,
        **run_stats,
    }
    if extra:
        record.update(extra)

    output_root.mkdir(parents=True, exist_ok=True)
    _upsert_jsonl(output_root / "summary.jsonl", record)
    return record


def parse_args():
    p = argparse.ArgumentParser(
        description="Official AppWorld evaluation + summary.jsonl writer"
    )
    p.add_argument(
        "--experiment_name",
        required=True,
        help="e.g. Qwen_Qwen3.5-35B-A3B_qwen35_a3b_baseline_rep1",
    )
    p.add_argument("--split", default="test_normal")
    p.add_argument(
        "--root",
        default=".",
        help="AppWorld root (must contain experiments/outputs/ and data/).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rec = evaluate_and_summarize(args.experiment_name, args.split, args.root)
    summary_path = Path(args.root) / "outputs" / args.experiment_name / "summary.jsonl"
    print(json.dumps(rec, indent=2))
    print(
        f"\n[evaluate] {args.experiment_name} / {args.split}: "
        f"TGC={rec['task_goal_completion']}%  SGC={rec['scenario_goal_completion']}%  "
        f"({rec['num_tasks_passed']}/{rec['num_tasks_evaluated']} tasks)  "
        f"-> {summary_path}"
    )


if __name__ == "__main__":
    main()
