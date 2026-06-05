"""Trajectory-divergence runner.

For every preserve-recent-segment boundary, prompt the agent for a single-shot
5-action speculative plan under each condition (self / cumulative / mask_obs /
fifo / random), judge it (gemini) against the actions the agent actually took,
and average the 0-100 similarity into one number per condition.

Reuses the cached divergence artifacts:
  * segmentation + cumulative summaries  <- divergence/compressions/<tag>/<task>.json
  * the real trajectories                <- <trajectory_dir>/task_<id>/llm_history.json

Generated plans and judge scores are cached to
``divergence/outputs/divergence_cache/<tag>/<task>.json`` so re-runs are cheap.

Example:
    cd experiments/appworld
    export VLLM_BASE_URL=http://r4519u01n01:8000/v1
    python -m divergence.run \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
        --output divergence/outputs/divergence_b2048.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, pstdev
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divergence.core import (  # noqa: E402
    CONDITIONS,
    build_context,
    parse_plan,
    with_plan_instruction,
)
from divergence.judge import GeminiJudge  # noqa: E402
from divergence.segmentation import load_trajectory  # noqa: E402
from divergence.vllm_client import VLLMClient  # noqa: E402


# --------------------------------------------------------------------------- #
# Cache (per task).
# --------------------------------------------------------------------------- #
class DivergenceCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> str:
        return os.path.join(self.cache_dir, f"{task_id}.json")

    def load(self, task_id: str) -> Dict:
        path = self._path(task_id)
        if os.path.exists(path):
            try:
                return json.load(open(path))
            except Exception:
                return {}
        return {}

    def save(self, task_id: str, data: Dict) -> None:
        with self._lock:
            with open(self._path(task_id), "w") as f:
                json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- #
# One boundary: generate + judge all conditions (with caching).
# --------------------------------------------------------------------------- #
def process_boundary(
    traj,
    segments: List[List[int]],
    summaries: List[str],
    i: int,
    agent: VLLMClient,
    judge: GeminiJudge,
    cache: DivergenceCache,
    cached_task: Dict,
    max_next_actions: int,
    budget: int,
    max_gen_tokens: int,
    seed: int,
    overwrite: bool,
) -> Dict:
    end_i = segments[i][-1]
    n_steps = len(traj.steps)
    r = min(max_next_actions, n_steps - 1 - end_i)
    real_actions = [traj.steps[end_i + k].action for k in range(1, r + 1)]

    key = str(i)
    rec = cached_task.get(key, {}) if not overwrite else {}
    rec.setdefault("verbatim_segment", i)
    rec.setdefault("end_step", end_i)
    rec["num_real_actions"] = r
    rec["real_actions"] = real_actions
    conds = rec.setdefault("conditions", {})

    for cond in CONDITIONS:
        existing = conds.get(cond)
        if existing and existing.get("score") is not None and not overwrite:
            continue
        try:
            ctx = build_context(
                traj, segments, i, cond, summaries, agent.count_tokens, budget=budget, seed=seed
            )
            ctx = with_plan_instruction(ctx, max_next_actions)
            plan_text = None
            for attempt in range(4):
                try:
                    plan_text = agent.chat(ctx, max_tokens=max_gen_tokens, temperature=0.0, seed=seed)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt + 0.5)
            predicted = parse_plan(plan_text, max_next_actions)[:r]
            verdict = judge.score(traj.task, real_actions, predicted)
            conds[cond] = {
                "score": verdict["score"],
                "reasoning": verdict["reasoning"],
                "predicted": predicted,
                "plan_text": plan_text,
            }
        except Exception as e:
            conds[cond] = {"score": None, "error": f"{type(e).__name__}: {e}"}

    cached_task[key] = rec
    cache.save(traj.task_id, cached_task)
    return rec


# --------------------------------------------------------------------------- #
# Per-task driver.
# --------------------------------------------------------------------------- #
def process_task(
    task_dir: str,
    comp_path: str,
    agent: VLLMClient,
    judge: GeminiJudge,
    cache: DivergenceCache,
    args,
) -> Dict:
    comp = json.load(open(comp_path))
    segments = [seg["step_indices"] for seg in comp["segments"]]
    summaries = [seg["summary"] for seg in comp["segments"]]
    traj = load_trajectory(task_dir)

    cached_task = cache.load(traj.task_id)
    boundaries: List[Dict] = []
    # i = 1 .. num_segments-2 (needs an earlier prefix AND a following action).
    for i in range(1, len(segments) - 1):
        rec = process_boundary(
            traj, segments, summaries, i, agent, judge, cache, cached_task,
            max_next_actions=args.max_next_actions, budget=args.compression_budget,
            max_gen_tokens=args.max_gen_tokens, seed=args.seed, overwrite=args.overwrite,
        )
        boundaries.append(rec)
    return {"task_id": traj.task_id, "num_segments": len(segments), "boundaries": boundaries}


def find_pairs(trajectory_dir: str, compressions_dir: str):
    """Match task_<id> trajectory dirs to their cached compression JSONs."""
    pairs = []
    for fname in sorted(os.listdir(compressions_dir)):
        if not fname.endswith(".json"):
            continue
        task_id = fname[: -len(".json")]
        task_dir = os.path.join(trajectory_dir, task_id)
        if os.path.isdir(task_dir):
            pairs.append((task_id, task_dir, os.path.join(compressions_dir, fname)))
    return pairs


# --------------------------------------------------------------------------- #
# Aggregation.
# --------------------------------------------------------------------------- #
def aggregate(task_results: List[Dict]) -> Dict:
    """Mean judge score per condition, over all scored boundaries."""
    per_cond: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    n_boundaries = 0
    for t in task_results:
        for b in t.get("boundaries", []):
            n_boundaries += 1
            for c in CONDITIONS:
                v = b.get("conditions", {}).get(c, {}).get("score")
                if v is not None:
                    per_cond[c].append(float(v))
    summary = {
        "num_trajectories": len(task_results),
        "num_boundaries": n_boundaries,
        "scores": {},
    }
    for c in CONDITIONS:
        vals = per_cond[c]
        summary["scores"][c] = {
            "mean": round(mean(vals), 2) if vals else None,
            "std": round(pstdev(vals), 2) if len(vals) > 1 else None,
            "n": len(vals),
        }
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressions_dir", required=True,
                   help="Cached cumulative compressions dir (divergence/compressions/<tag>)")
    p.add_argument("--agent_model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"))
    p.add_argument("--agent_base_url", default=os.environ.get("VLLM_BASE_URL"))
    p.add_argument("--judge_model", default="gemini-3.5-flash")
    p.add_argument("--max_next_actions", type=int, default=5)
    p.add_argument("--compression_budget", type=int, default=2048,
                   help="Token budget for selection baselines (mask_obs/fifo/random)")
    p.add_argument("--max_gen_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=6, help="Parallel tasks")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Ignore cached plans/scores")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.agent_base_url:
        raise SystemExit("No agent endpoint: pass --agent_base_url or set VLLM_BASE_URL")

    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    judge = GeminiJudge(model=args.judge_model)

    tag = os.path.basename(args.compressions_dir.rstrip("/"))
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(__file__), "outputs", "divergence_cache", tag
    )
    cache = DivergenceCache(cache_dir)

    pairs = find_pairs(args.trajectory_dir, args.compressions_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[divergence] {len(pairs)} trajectories | conditions={CONDITIONS} "
          f"| next_actions={args.max_next_actions} | budget={args.compression_budget} "
          f"| judge={args.judge_model} | cache={cache_dir}")

    task_results: List[Optional[Dict]] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2idx = {
            ex.submit(process_task, td, cp, agent, judge, cache, args): (idx, tid)
            for idx, (tid, td, cp) in enumerate(pairs)
        }
        done = 0
        for fut in as_completed(fut2idx):
            idx, tid = fut2idx[fut]
            done += 1
            try:
                res = fut.result()
                task_results[idx] = res
                nb = len(res["boundaries"])
                got = {c: sum(1 for b in res["boundaries"]
                              if b.get("conditions", {}).get(c, {}).get("score") is not None)
                       for c in CONDITIONS}
                print(f"[{done}/{len(pairs)}] {tid}: {nb} boundaries scored={got}")
            except Exception as e:
                print(f"[{done}/{len(pairs)}] {tid}: ERROR {e}")
                traceback.print_exc()
                task_results[idx] = {"task_id": tid, "error": str(e), "boundaries": []}

    task_results = [t for t in task_results if t is not None]
    summary = aggregate(task_results)

    report = {
        "config": {
            "trajectory_dir": args.trajectory_dir,
            "compressions_dir": args.compressions_dir,
            "agent_model": args.agent_model,
            "judge_model": args.judge_model,
            "max_next_actions": args.max_next_actions,
            "compression_budget": args.compression_budget,
            "rollout": "single_shot_plan",
            "scheme": "preserve_recent_segment",
        },
        "summary": summary,
        "trajectories": task_results,
    }

    print("\n=== TRAJECTORY DIVERGENCE (mean judge similarity 0-100; higher = less divergence) ===")
    for c in CONDITIONS:
        s = summary["scores"][c]
        label = "self-consistency" if c == "self" else c
        print(f"  {label:18s}: {s['mean']}  (std {s['std']}, n={s['n']})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
