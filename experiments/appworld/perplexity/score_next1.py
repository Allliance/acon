"""Augment the cached divergence run with a *next-1-action* score.

Re-uses the already-generated speculative plans (cached in
``outputs/divergence_cache/<tag>/<task>.json``): for each boundary/condition it
judges only the FIRST predicted block against the agent's actual next action, so
no agent generations are repeated — only judge calls, which are cached back into
the same files (``score_next1`` / ``reasoning_next1``).

Reports both metrics per condition:
  * next1 : similarity of the immediate next action.
  * next5 : the existing 5-action-plan similarity (cached as ``score``).

Example:
    python -m perplexity.score_next1 \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressions_dir perplexity/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
        --output perplexity/outputs/divergence_b2048.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, pstdev
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity.divergence import CONDITIONS  # noqa: E402
from perplexity.judge import GeminiJudge  # noqa: E402
from perplexity.segmentation import load_trajectory  # noqa: E402

_LOCK = threading.Lock()


def augment_task(task_id: str, task_dir: str, cache_path: str, judge: GeminiJudge,
                 overwrite: bool) -> Dict:
    data = json.load(open(cache_path))
    task_text = load_trajectory(task_dir).task
    changed = False
    for key, rec in data.items():
        if not key.isdigit():
            continue
        real = rec.get("real_actions") or []
        for cond, c in rec.get("conditions", {}).items():
            if not c:
                continue
            if c.get("score_next1") is not None and not overwrite:
                continue
            pred = c.get("predicted") or []
            if not real or not pred:
                c["score_next1"] = None
                c["reasoning_next1"] = "no actions"
                changed = True
                continue
            v = judge.score(task_text, real[:1], pred[:1])
            c["score_next1"] = v["score"]
            c["reasoning_next1"] = v["reasoning"]
            changed = True
    if changed:
        with _LOCK:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
    return data


def aggregate(task_caches: List[Dict]) -> Dict:
    conds: List[str] = []
    out: Dict[str, Dict[str, List[float]]] = {}
    n_boundaries = 0
    for data in task_caches:
        for key, rec in data.items():
            if not key.isdigit():
                continue
            n_boundaries += 1
            for c, cc in rec.get("conditions", {}).items():
                if c not in out:
                    out[c] = {"next1": [], "next5": []}
                    conds.append(c)
                if cc.get("score_next1") is not None:
                    out[c]["next1"].append(float(cc["score_next1"]))
                if cc.get("score") is not None:
                    out[c]["next5"].append(float(cc["score"]))
    summary = {"num_boundaries": n_boundaries, "scores": {}}
    for c in conds:
        row = {}
        for m in ("next1", "next5"):
            vals = out[c][m]
            row[m] = {
                "mean": round(mean(vals), 2) if vals else None,
                "std": round(pstdev(vals), 2) if len(vals) > 1 else None,
                "n": len(vals),
            }
        summary["scores"][c] = row
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressions_dir", required=True)
    p.add_argument("--judge_model", default="gemini-3.5-flash")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default=None, help="Re-write this report with next1+next5 summary")
    return p.parse_args()


def main():
    args = parse_args()
    judge = GeminiJudge(model=args.judge_model)
    tag = os.path.basename(args.compressions_dir.rstrip("/"))
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(__file__), "outputs", "divergence_cache", tag
    )

    tasks = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".json"):
            continue
        tid = fname[: -len(".json")]
        td = os.path.join(args.trajectory_dir, tid)
        if os.path.isdir(td):
            tasks.append((tid, td, os.path.join(cache_dir, fname)))
    print(f"[next1] augmenting {len(tasks)} cached task files | judge={args.judge_model}")

    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(augment_task, tid, td, cp, judge, args.overwrite): tid
                for tid, td, cp in tasks}
        done = 0
        for fut in as_completed(futs):
            tid = futs[fut]
            done += 1
            try:
                results[tid] = fut.result()
                print(f"[{done}/{len(tasks)}] {tid}: ok")
            except Exception as e:
                print(f"[{done}/{len(tasks)}] {tid}: ERROR {e}")

    summary = aggregate(list(results.values()))

    print("\n=== TRAJECTORY DIVERGENCE (mean judge similarity 0-100; higher = less divergence) ===")
    print(f"{'condition':18s} {'next1':>14s} {'next5':>14s}")
    ordered = (["self"] if "self" in summary["scores"] else []) + \
              [c for c in summary["scores"] if c != "self"]
    for c in ordered:
        s = summary["scores"][c]
        label = "self-consistency" if c == "self" else c
        n1, n5 = s["next1"], s["next5"]
        print(f"{label:18s} {str(n1['mean'])+' (±'+str(n1['std'])+')':>14s} "
              f"{str(n5['mean'])+' (±'+str(n5['std'])+')':>14s}")

    if args.output and os.path.exists(args.output):
        report = json.load(open(args.output))
        report["summary_next1_next5"] = summary
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nUpdated {args.output} (added summary_next1_next5)")


if __name__ == "__main__":
    main()
