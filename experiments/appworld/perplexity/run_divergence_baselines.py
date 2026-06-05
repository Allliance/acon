"""Trajectory divergence for FULL-MODE (non-recurrent) summary baselines.

Each baseline re-summarizes the WHOLE past prefix (segments 0..i-1) at every
boundary with a compressor LLM, keeps the recent segment S_i verbatim, then has
the agent (Qwen3.5-35B-A3B) produce a single-shot 5-action speculative plan that
a judge (gemini) compares against the agent's real next actions.

Default 4 baselines (compressor model x compression prompt):
  * 35b_default : Qwen3.5-35B-A3B + prompt_history_v2      (compressor @ agent endpoint)
  * 35b_acon    : Qwen3.5-35B-A3B + compression_prompt_ut  (compressor @ agent endpoint)
  * 27b_default : Qwen3.5-27B      + prompt_history_v2      (compressor @ --compressor_base_url)
  * 27b_acon    : Qwen3.5-27B      + compression_prompt_ut  (compressor @ --compressor_base_url)

plus the ``self`` full-context control. Summaries + plans + judge verdicts are
cached on disk (resumable). Reuses the cached segmentation from
``--compressions_dir`` so boundaries match the other divergence runs.

Example:
    cd experiments/appworld
    python -m perplexity.run_divergence_baselines \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressions_dir perplexity/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
        --agent_base_url http://r818u33n04:8000/v1 \
        --compressor_base_url http://r4519u01n01:8000/v1 \
        --workers 16 \
        --output perplexity/outputs/divergence_fullbaselines_b2048.json
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

from perplexity.compressor import make_compressor  # noqa: E402
from perplexity.divergence import (  # noqa: E402
    build_context,
    build_summary_context,
    parse_plan,
    with_plan_instruction,
)
from perplexity.judge import GeminiJudge  # noqa: E402
from perplexity.segmentation import load_trajectory  # noqa: E402
from perplexity.vllm_client import VLLMClient  # noqa: E402

# Baseline spec: name -> (config file, which endpoint serves the compressor).
# "agent" means the compressor shares the agent endpoint (35B-A3B); "compressor"
# means it uses --compressor_base_url (27B).
DEFAULT_BASELINES = [
    ("35b_default", "configs/context_opt/div_full_35b_default.yaml", "agent"),
    ("35b_acon", "configs/context_opt/div_full_35b_acon.yaml", "agent"),
    ("27b_default", "configs/context_opt/div_full_27b_default.yaml", "compressor"),
    ("27b_acon", "configs/context_opt/div_full_27b_acon.yaml", "compressor"),
    ("9b_default", "configs/context_opt/div_full_9b_default.yaml", "9b"),
    ("9b_acon", "configs/context_opt/div_full_9b_acon.yaml", "9b"),
    ("4b_default", "configs/context_opt/div_full_4b_default.yaml", "4b"),
    ("4b_acon", "configs/context_opt/div_full_4b_acon.yaml", "4b"),
    ("120b_default", "configs/context_opt/div_full_120b_default.yaml", "120b"),
    ("120b_acon", "configs/context_opt/div_full_120b_acon.yaml", "120b"),
]


class Cache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, task_id):
        return os.path.join(self.cache_dir, f"{task_id}.json")

    def load(self, task_id):
        p = self._path(task_id)
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                return {}
        return {}

    def save(self, task_id, data):
        with self._lock:
            with open(self._path(task_id), "w") as f:
                json.dump(data, f, indent=2)


def _gen_with_retry(agent, ctx, max_gen_tokens, seed):
    for attempt in range(4):
        try:
            return agent.chat(ctx, max_tokens=max_gen_tokens, temperature=0.0, seed=seed)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt + 0.5)


def process_boundary(traj, segments, i, agent, compressors, judge, cache, cached_task,
                     conditions, max_next_actions, max_gen_tokens, seed, overwrite):
    end_i = segments[i][-1]
    n_steps = len(traj.steps)
    r = min(max_next_actions, n_steps - 1 - end_i)
    real_actions = [traj.steps[end_i + k].action for k in range(1, r + 1)]

    key = str(i)
    rec = cached_task.get(key, {}) if not overwrite else {}
    rec.update({"verbatim_segment": i, "end_step": end_i, "num_real_actions": r,
                "real_actions": real_actions})
    conds = rec.setdefault("conditions", {})

    for cond in conditions:
        existing = conds.get(cond)
        if existing and existing.get("score") is not None and not overwrite:
            continue
        try:
            summary = None
            if cond == "self":
                ctx = build_context(traj, segments, i, "self", None, agent.count_tokens)
            else:
                ctx, summary = build_summary_context(
                    traj, segments, i, compressors[cond], agent.count_tokens
                )
            ctx = with_plan_instruction(ctx, max_next_actions)
            plan_text = _gen_with_retry(agent, ctx, max_gen_tokens, seed)
            predicted = parse_plan(plan_text, max_next_actions)[:r]
            verdict = judge.score(traj.task, real_actions, predicted)
            entry = {
                "score": verdict["score"],
                "reasoning": verdict["reasoning"],
                "predicted": predicted,
                "plan_text": plan_text,
            }
            if summary is not None:
                entry["summary"] = summary
                entry["summary_tokens"] = agent.count_tokens(summary)
            conds[cond] = entry
        except Exception as e:
            conds[cond] = {"score": None, "error": f"{type(e).__name__}: {e}"}

    cached_task[key] = rec
    cache.save(traj.task_id, cached_task)
    return rec


def process_task(task_id, task_dir, comp_path, agent, compressors, judge, cache, conditions, args):
    comp = json.load(open(comp_path))
    segments = [seg["step_indices"] for seg in comp["segments"]]
    traj = load_trajectory(task_dir)
    cached_task = cache.load(traj.task_id)
    boundaries = []
    for i in range(1, len(segments) - 1):
        boundaries.append(process_boundary(
            traj, segments, i, agent, compressors, judge, cache, cached_task,
            conditions, args.max_next_actions, args.max_gen_tokens, args.seed, args.overwrite,
        ))
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


def aggregate(task_results, conditions):
    per = {c: [] for c in conditions}
    nb = 0
    for t in task_results:
        for b in t.get("boundaries", []):
            nb += 1
            for c in conditions:
                v = b.get("conditions", {}).get(c, {}).get("score")
                if v is not None:
                    per[c].append(float(v))
    out = {"num_trajectories": len(task_results), "num_boundaries": nb, "scores": {}}
    for c in conditions:
        vals = per[c]
        out["scores"][c] = {
            "mean": round(mean(vals), 2) if vals else None,
            "std": round(pstdev(vals), 2) if len(vals) > 1 else None,
            "n": len(vals),
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressions_dir", required=True)
    p.add_argument("--agent_model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--agent_base_url", default=os.environ.get("VLLM_BASE_URL", "http://r818u33n04:8000/v1"))
    p.add_argument("--compressor_base_url", default=os.environ.get("VLLM_COMPRESSOR_BASE_URL", "http://r4519u01n01:8000/v1"),
                   help="Endpoint for the 27B compressor baselines")
    p.add_argument("--base_9b", default="http://r818u29n04:8000/v1", help="Endpoint for Qwen3.5-9B compressor")
    p.add_argument("--base_4b", default="http://r818u29n04:8001/v1", help="Endpoint for Qwen3.5-4B compressor")
    p.add_argument("--base_120b", default="http://r818u33n04:8000/v1", help="Endpoint for gpt-oss-120b compressor")
    p.add_argument("--judge_model", default="gemini-3.5-flash")
    p.add_argument("--max_next_actions", type=int, default=5)
    p.add_argument("--max_gen_tokens", type=int, default=2048)
    p.add_argument("--max_segment_tokens", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    judge = GeminiJudge(model=args.judge_model)

    # Build one compressor per baseline (shared across threads).
    endpoints = {"agent": args.agent_base_url, "compressor": args.compressor_base_url,
                 "9b": args.base_9b, "4b": args.base_4b, "120b": args.base_120b}
    compressors = {}
    for name, cfg, which in DEFAULT_BASELINES:
        compressors[name] = make_compressor(
            config_path=cfg, max_segment_tokens=args.max_segment_tokens,
            compressor_base_url=endpoints[which], count_tokens=agent.count_tokens,
        )
    conditions = ["self"] + [b[0] for b in DEFAULT_BASELINES]

    tag = os.path.basename(args.compressions_dir.rstrip("/"))
    cache_dir = args.cache_dir or os.path.join(
        os.path.dirname(__file__), "outputs", "divergence_cache_fullbaselines", tag
    )
    cache = Cache(cache_dir)

    pairs = find_pairs(args.trajectory_dir, args.compressions_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[divergence-baselines] {len(pairs)} trajectories | conditions={conditions}")
    print(f"  agent      : {args.agent_model} @ {args.agent_base_url}")
    print(f"  35b compr. : @ {args.agent_base_url}")
    print(f"  27b compr. : @ {args.compressor_base_url}")
    print(f"  9b  compr. : @ {args.base_9b}")
    print(f"  4b  compr. : @ {args.base_4b}")
    print(f"  judge={args.judge_model} | workers={args.workers} | cache={cache_dir}")

    task_results: List[Optional[Dict]] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2idx = {
            ex.submit(process_task, tid, td, cp, agent, compressors, judge, cache, conditions, args): (idx, tid)
            for idx, (tid, td, cp) in enumerate(pairs)
        }
        done = 0
        for fut in as_completed(fut2idx):
            idx, tid = fut2idx[fut]
            done += 1
            try:
                res = fut.result()
                task_results[idx] = res
                got = {c: sum(1 for b in res["boundaries"]
                              if b.get("conditions", {}).get(c, {}).get("score") is not None)
                       for c in conditions}
                print(f"[{done}/{len(pairs)}] {tid}: {len(res['boundaries'])} boundaries {got}")
            except Exception as e:
                print(f"[{done}/{len(pairs)}] {tid}: ERROR {e}")
                traceback.print_exc()
                task_results[idx] = {"task_id": tid, "error": str(e), "boundaries": []}

    task_results = [t for t in task_results if t is not None]
    summary = aggregate(task_results, conditions)

    report = {
        "config": {
            "trajectory_dir": args.trajectory_dir,
            "agent_model": args.agent_model,
            "agent_base_url": args.agent_base_url,
            "compressor_base_url": args.compressor_base_url,
            "judge_model": args.judge_model,
            "baselines": {b[0]: b[1] for b in DEFAULT_BASELINES},
            "summary_mode": "full",
            "max_next_actions": args.max_next_actions,
            "compression_budget": 2048,
            "rollout": "single_shot_plan",
            "scheme": "preserve_recent_segment",
        },
        "summary": summary,
        "trajectories": task_results,
    }

    print("\n=== TRAJECTORY DIVERGENCE — full-mode baselines (next5; higher = less divergence) ===")
    for c in conditions:
        s = summary["scores"][c]
        label = "self-consistency" if c == "self" else c
        print(f"  {label:16s}: {s['mean']}  (std {s['std']}, n={s['n']})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
