"""Re-score divergence for the LLM-based (summary) compressors under a DIFFERENT
agent decoding regime, reusing the CACHED compressions.

The compressor summaries are already cached per cell in a source run (e.g.
``divergence_cache_fullbaselines/<tag>``). This runner does NOT call any
compressor server: for each cached cell it rebuilds the context from the stored
summary (``self`` is rebuilt from the full trajectory), regenerates only the
agent's single-shot 5-action plan under the requested decoding params, and
re-judges it. So only the agent endpoint + the judge are needed.

Example (sampled regime):
    python -m divergence.run_resample \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressions_dir divergence/compressions/qwen35a3b_self_cumulative_b2048_t6000 \
        --source_cache_dir divergence/outputs/divergence_cache_fullbaselines/qwen35a3b_self_cumulative_b2048_t6000 \
        --agent_base_url http://r818u33n08:8000/v1 \
        --temperature 0.6 --top_p 0.95 --top_k 20 --min_p 0.0 \
        --presence_penalty 0.0 --repetition_penalty 1.0 \
        --tag sampled --output divergence/outputs/divergence_fullbaselines_sampled_b2048.json
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
    build_context,
    build_context_from_summary,
    parse_plan,
    with_plan_instruction,
)
from divergence.judge import GeminiJudge  # noqa: E402
from divergence.segmentation import load_trajectory  # noqa: E402
from divergence.vllm_client import VLLMClient  # noqa: E402


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


def process_task(task_id, task_dir, comp_path, src_cache_dir, agent, judge, out_cache, args, sampling):
    comp = json.load(open(comp_path))
    segments = [seg["step_indices"] for seg in comp["segments"]]
    traj = load_trajectory(task_dir)

    src_path = os.path.join(src_cache_dir, f"{traj.task_id}.json")
    if not os.path.exists(src_path):
        return {"task_id": traj.task_id, "boundaries": []}
    src = json.load(open(src_path))
    out = out_cache.load(traj.task_id)

    boundaries = []
    for key, srec in src.items():
        if not key.isdigit():
            continue
        i = int(key)
        end_i = segments[i][-1]
        r = min(args.max_next_actions, len(traj.steps) - 1 - end_i)
        real = [traj.steps[end_i + k].action for k in range(1, r + 1)]
        rec = out.get(key, {}) if not args.overwrite else {}
        rec.update({"verbatim_segment": i, "end_step": end_i, "num_real_actions": r, "real_actions": real})
        conds = rec.setdefault("conditions", {})

        for cond, scell in srec.get("conditions", {}).items():
            ex = conds.get(cond)
            if ex and ex.get("score") is not None and not args.overwrite:
                continue
            # Need the context: self -> rebuild from trajectory; others -> cached summary.
            if cond == "self":
                ctx_base = build_context(traj, segments, i, "self", None, agent.count_tokens)
                summary = None
            else:
                summary = scell.get("summary")
                if summary is None:
                    conds[cond] = {"score": None, "error": "no_cached_summary"}
                    continue
                ctx_base = build_context_from_summary(traj, segments, i, summary)
            try:
                ctx = with_plan_instruction(ctx_base, args.max_next_actions)
                plan = _gen(agent, ctx, args.max_gen_tokens, args.temperature, args.seed, sampling)
                pred = parse_plan(plan, args.max_next_actions)[:r]
                v = judge.score(traj.task, real, pred)
                entry = {"score": v["score"], "reasoning": v["reasoning"], "predicted": pred, "plan_text": plan}
                if summary is not None:
                    entry["summary_tokens"] = scell.get("summary_tokens")
                conds[cond] = entry
            except Exception as e:
                conds[cond] = {"score": None, "error": f"{type(e).__name__}: {e}"}
        out[key] = rec
        out_cache.save(traj.task_id, out)
        boundaries.append(rec)
    return {"task_id": traj.task_id, "boundaries": boundaries}


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
    conds, out = [], {}
    nb = 0
    for t in task_results:
        for b in t.get("boundaries", []):
            nb += 1
            for c, cc in b.get("conditions", {}).items():
                if c not in out:
                    out[c] = []
                    conds.append(c)
                if cc.get("score") is not None:
                    out[c].append(float(cc["score"]))
    summary = {"num_boundaries": nb, "scores": {}}
    ordered = (["self"] if "self" in conds else []) + [c for c in conds if c != "self"]
    for c in ordered:
        vals = out[c]
        summary["scores"][c] = {"mean": round(mean(vals), 2) if vals else None,
                                "std": round(pstdev(vals), 2) if len(vals) > 1 else None, "n": len(vals)}
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressions_dir", required=True)
    p.add_argument("--source_cache_dir", required=True, help="Greedy fullbaselines cache to reuse summaries from")
    p.add_argument("--agent_model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--agent_base_url", default="http://r818u33n08:8000/v1")
    p.add_argument("--judge_model", default="gemini-3.5-flash")
    p.add_argument("--max_next_actions", type=int, default=5)
    p.add_argument("--max_gen_tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_p", type=float, default=None)
    p.add_argument("--presence_penalty", type=float, default=None)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--tag", default="sampled")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    judge = GeminiJudge(model=args.judge_model)
    sampling = {k: getattr(args, k) for k in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
                if getattr(args, k) is not None}

    comp_tag = os.path.basename(args.compressions_dir.rstrip("/"))
    out_cache_dir = os.path.join(os.path.dirname(__file__), "outputs",
                                 "divergence_cache_fullbaselines_resampled", f"{comp_tag}__{args.tag}")
    out_cache = Cache(out_cache_dir)

    pairs = find_pairs(args.trajectory_dir, args.compressions_dir)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[resample] {len(pairs)} trajectories | tag={args.tag} | reuse summaries from {args.source_cache_dir}")
    print(f"  agent={args.agent_model} @ {args.agent_base_url}")
    print(f"  decoding: temperature={args.temperature} extras={sampling or '(none)'} seed={args.seed}")
    print(f"  out_cache={out_cache_dir}")

    task_results: List[Optional[Dict]] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_task, tid, td, cp, args.source_cache_dir, agent, judge, out_cache, args, sampling): (idx, tid)
                for idx, (tid, td, cp) in enumerate(pairs)}
        done = 0
        for fut in as_completed(futs):
            idx, tid = futs[fut]
            done += 1
            try:
                res = fut.result()
                task_results[idx] = res
                print(f"[{done}/{len(pairs)}] {tid}: {len(res['boundaries'])} boundaries")
            except Exception as e:
                print(f"[{done}/{len(pairs)}] {tid}: ERROR {e}")
                traceback.print_exc()
                task_results[idx] = {"task_id": tid, "error": str(e), "boundaries": []}

    task_results = [t for t in task_results if t is not None]
    summary = aggregate(task_results)
    report = {
        "config": {
            "trajectory_dir": args.trajectory_dir, "agent_model": args.agent_model,
            "source_cache_dir": args.source_cache_dir, "judge_model": args.judge_model,
            "tag": args.tag, "decoding": {"temperature": args.temperature, **sampling, "seed": args.seed},
            "rollout": "single_shot_plan", "scheme": "preserve_recent_segment",
            "note": "compressor summaries reused from source_cache_dir; only agent plan re-sampled",
        },
        "summary": summary, "trajectories": task_results,
    }
    print(f"\n=== LLM-COMPRESSOR BASELINES — {args.tag} (next5; higher = less divergence) ===")
    for c, s in summary["scores"].items():
        label = "self-consistency" if c == "self" else c
        print(f"  {label:16s}: {s['mean']}  (std {s['std']}, n={s['n']})")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
