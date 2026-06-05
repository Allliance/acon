"""Trajectory divergence for the NAIVE selection baselines (no compressor LLM).

Conditions: ``self`` (full-context control) + ``fifo`` / ``random`` / ``mask_obs``.
Each selection baseline keeps/drops prefix turns to fit ``--compression_budget``
tokens (measured with the agent's own tokenizer), keeps the recent segment S_i
verbatim, then the agent produces a single-shot 5-action speculative plan that a
judge (gemini) compares to the agent's real next actions.

Because the agent's plan is what's sampled, this runner exposes the agent
decoding params so the same boundaries can be scored under different sampling
regimes (e.g. greedy temperature=0 vs temperature=0.6/top_p=0.95/top_k=20).

Example (greedy, the default):
    python -m divergence.run_naive \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
        --agent_base_url http://r818u33n08:8000/v1 \
        --tag greedy --output divergence/outputs/divergence_naive_greedy_b2048.json

Sampled:
    ... --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 \
        --presence_penalty 0.0 --repetition_penalty 1.0 \
        --tag sampled --output divergence/outputs/divergence_naive_sampled_b2048.json
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

from divergence.core import build_context, parse_plan, with_plan_instruction  # noqa: E402
from divergence.judge import GeminiJudge  # noqa: E402
from divergence.segmentation import load_trajectory  # noqa: E402
from divergence.vllm_client import VLLMClient  # noqa: E402

CONDITIONS = ["self", "fifo", "random", "mask_obs"]


class Cache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, tid):
        return os.path.join(self.cache_dir, f"{tid}.json")

    def load(self, tid):
        p = self._path(tid)
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return {}
        return {}

    def save(self, tid, data):
        with self._lock:
            with open(self._path(tid), "w") as f:
                json.dump(data, f, indent=2)


def _gen(agent, ctx, max_gen_tokens, temperature, seed, sampling):
    for attempt in range(4):
        try:
            return agent.chat(ctx, max_tokens=max_gen_tokens, temperature=temperature,
                              seed=seed, sampling=sampling)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt + 0.5)


def process_task(task_id, task_dir, comp_path, agent, judge, cache, args, sampling):
    comp = json.load(open(comp_path))
    segments = [seg["step_indices"] for seg in comp["segments"]]
    traj = load_trajectory(task_dir)
    cached = cache.load(traj.task_id)
    boundaries = []
    for i in range(1, len(segments) - 1):
        end_i = segments[i][-1]
        r = min(args.max_next_actions, len(traj.steps) - 1 - end_i)
        real = [traj.steps[end_i + k].action for k in range(1, r + 1)]
        key = str(i)
        rec = cached.get(key, {}) if not args.overwrite else {}
        rec.update({"verbatim_segment": i, "end_step": end_i, "num_real_actions": r, "real_actions": real})
        conds = rec.setdefault("conditions", {})
        for cond in CONDITIONS:
            ex = conds.get(cond)
            if ex and ex.get("score") is not None and not args.overwrite:
                continue
            try:
                ctx = build_context(traj, segments, i, cond, None, agent.count_tokens,
                                    budget=args.compression_budget, seed=args.seed)
                ctx = with_plan_instruction(ctx, args.max_next_actions)
                plan = _gen(agent, ctx, args.max_gen_tokens, args.temperature, args.seed, sampling)
                pred = parse_plan(plan, args.max_next_actions)[:r]
                v = judge.score(traj.task, real, pred)
                conds[cond] = {"score": v["score"], "reasoning": v["reasoning"],
                               "predicted": pred, "plan_text": plan}
            except Exception as e:
                conds[cond] = {"score": None, "error": f"{type(e).__name__}: {e}"}
        cached[key] = rec
        cache.save(traj.task_id, cached)
        boundaries.append(rec)
    return {"task_id": traj.task_id, "num_segments": len(segments), "boundaries": boundaries}


def find_pairs(trajectory_dir, compressions_dir):
    pairs = []
    for fname in sorted(os.listdir(compressions_dir)):
        if not fname.endswith(".json"):
            continue
        tid = fname[:-5]
        td = os.path.join(trajectory_dir, tid)
        if os.path.isdir(td):
            pairs.append((tid, td, os.path.join(compressions_dir, fname)))
    return pairs


def aggregate(task_results):
    per = {c: [] for c in CONDITIONS}
    nb = 0
    for t in task_results:
        for b in t.get("boundaries", []):
            nb += 1
            for c in CONDITIONS:
                v = b.get("conditions", {}).get(c, {}).get("score")
                if v is not None:
                    per[c].append(float(v))
    out = {"num_trajectories": len(task_results), "num_boundaries": nb, "scores": {}}
    for c in CONDITIONS:
        vals = per[c]
        out["scores"][c] = {"mean": round(mean(vals), 2) if vals else None,
                            "std": round(pstdev(vals), 2) if len(vals) > 1 else None, "n": len(vals)}
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressions_dir", required=True)
    p.add_argument("--agent_model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--agent_base_url", default="http://r818u33n08:8000/v1")
    p.add_argument("--judge_model", default="gemini-3.5-flash")
    p.add_argument("--compression_budget", type=int, default=2048)
    p.add_argument("--max_next_actions", type=int, default=5)
    p.add_argument("--max_gen_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    # Decoding params (greedy by default; extras only sent when set).
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_p", type=float, default=None)
    p.add_argument("--presence_penalty", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--tag", default="greedy", help="Label for cache dir / report (e.g. greedy, sampled)")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    judge = GeminiJudge(model=args.judge_model)

    sampling = {}
    for k in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
        v = getattr(args, k)
        if v is not None:
            sampling[k] = v

    comp_tag = os.path.basename(args.compressions_dir.rstrip("/"))
    cache_dir = os.path.join(os.path.dirname(__file__), "outputs",
                             "divergence_cache_naive", f"{comp_tag}__{args.tag}")
    cache = Cache(cache_dir)

    pairs = find_pairs(args.trajectory_dir, args.compressions_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[divergence-naive] {len(pairs)} trajectories | conditions={CONDITIONS} | tag={args.tag}")
    print(f"  agent={args.agent_model} @ {args.agent_base_url} | budget={args.compression_budget}")
    print(f"  decoding: temperature={args.temperature} extras={sampling or '(none)'} seed={args.seed}")
    print(f"  cache={cache_dir}")

    task_results: List[Optional[Dict]] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_task, tid, td, cp, agent, judge, cache, args, sampling): (idx, tid)
                for idx, (tid, td, cp) in enumerate(pairs)}
        done = 0
        for fut in as_completed(futs):
            idx, tid = futs[fut]
            done += 1
            try:
                res = fut.result()
                task_results[idx] = res
                got = {c: sum(1 for b in res["boundaries"]
                              if b.get("conditions", {}).get(c, {}).get("score") is not None)
                       for c in CONDITIONS}
                print(f"[{done}/{len(pairs)}] {tid}: {len(res['boundaries'])} boundaries {got}")
            except Exception as e:
                print(f"[{done}/{len(pairs)}] {tid}: ERROR {e}")
                traceback.print_exc()
                task_results[idx] = {"task_id": tid, "error": str(e), "boundaries": []}

    task_results = [t for t in task_results if t is not None]
    summary = aggregate(task_results)
    report = {
        "config": {
            "trajectory_dir": args.trajectory_dir, "agent_model": args.agent_model,
            "agent_base_url": args.agent_base_url, "judge_model": args.judge_model,
            "compression_budget": args.compression_budget, "tag": args.tag,
            "decoding": {"temperature": args.temperature, **sampling, "seed": args.seed},
            "rollout": "single_shot_plan", "scheme": "preserve_recent_segment",
        },
        "summary": summary, "trajectories": task_results,
    }
    print(f"\n=== NAIVE BASELINES — {args.tag} (next5; higher = less divergence) ===")
    for c in CONDITIONS:
        s = summary["scores"][c]
        print(f"  {c:10s}: {s['mean']}  (std {s['std']}, n={s['n']})")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
