"""Generate and cache cumulative LLM compressions for a directory of trajectories.

This is the reusable artifact step: it walks each trajectory's segments
recurrently (summary_i = LLM(prev_summary, segment_i)) and writes the full
summary chain to ``cache_dir/<task_id>.json``. The perplexity scorer
(`perplexity.run`) reuses the exact same cache, so compression is only paid for
once.

Example:
    cd experiments/appworld
    python -m perplexity.compress \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressor_config configs/context_opt/qwen35a3b_self_cumulative_b2048.yaml \
        --agent_base_url http://r4519u01n01:8000/v1

Run inside the `smolagents` conda env. The compressor uses the same vLLM
endpoint as the agent unless --compressor_base_url is given.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity.compressor import make_compressor  # noqa: E402
from perplexity.segmentation import load_trajectory, segment_trajectory  # noqa: E402
from perplexity.vllm_client import VLLMClient  # noqa: E402


def find_task_dirs(trajectory_dir: str):
    entries = sorted(
        d for d in os.listdir(trajectory_dir)
        if d.startswith("task_") and os.path.isdir(os.path.join(trajectory_dir, d))
    )
    return [os.path.join(trajectory_dir, d) for d in entries]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--compressor_config", required=True)
    p.add_argument("--agent_model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"),
                   help="Tokenizer model used for segmentation (the agent model)")
    p.add_argument("--agent_base_url", default=os.environ.get("VLLM_BASE_URL"),
                   help="vLLM endpoint for tokenization (and the compressor, unless overridden)")
    p.add_argument("--compressor_base_url", default=os.environ.get("VLLM_COMPRESSOR_BASE_URL"),
                   help="Compressor vLLM endpoint (defaults to --agent_base_url)")
    p.add_argument("--cache_dir", default=None, help="Where to write <task_id>.json (default under perplexity/compressions/<tag>)")
    p.add_argument("--max_segment_tokens", type=int, default=6000)
    p.add_argument("--compression_budget", type=int, default=None, help="Override config budget (tokens)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent trajectories (the chain within a trajectory stays sequential)")
    p.add_argument("--overwrite", action="store_true", help="Regenerate even if a cache file exists")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.agent_base_url:
        raise SystemExit("No endpoint: pass --agent_base_url or set VLLM_BASE_URL")

    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    compressor = make_compressor(
        config_path=args.compressor_config,
        max_segment_tokens=args.max_segment_tokens,
        compression_budget=args.compression_budget,
        compressor_base_url=args.compressor_base_url or args.agent_base_url,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
        count_tokens=agent.count_tokens,
        debug=args.debug,
    )
    if compressor.strategy != "summary_cumulative":
        raise SystemExit(
            f"{compressor.name} is '{compressor.strategy}', not a cumulative compressor. "
            "Set summary_mode: cumulative in the config."
        )

    task_dirs = find_task_dirs(args.trajectory_dir)
    if args.limit:
        task_dirs = task_dirs[: args.limit]
    print(f"[compress] {len(task_dirs)} trajectories | {compressor.name} | budget={compressor.budget} "
          f"| workers={args.workers} | cache_dir={compressor.cache_dir}", flush=True)

    counter = {"done": 0, "ok": 0}
    lock = threading.Lock()

    def work(task_dir):
        # compute_chain/generate_trajectory are thread-safe (no shared instance
        # state); only independent LLM HTTP requests are issued.
        traj = load_trajectory(task_dir)
        segments = segment_trajectory(traj, agent.count_tokens, args.max_segment_tokens)
        was_cached = compressor._load_cache(traj, segments) is not None and not args.overwrite
        summaries = compressor.generate_trajectory(traj, segments)
        return len(segments), len(summaries), was_cached

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, td): os.path.basename(td) for td in task_dirs}
        for fut in as_completed(futures):
            name = futures[fut]
            with lock:
                counter["done"] += 1
                i = counter["done"]
            try:
                n_seg, n_sum, was_cached = fut.result()
                counter["ok"] += 1
                print(f"[{i}/{len(task_dirs)}] {name}: {n_seg} segments -> {n_sum} summaries"
                      f"{' (cached)' if was_cached else ''}", flush=True)
            except Exception as e:
                print(f"[{i}/{len(task_dirs)}] {name}: ERROR {e}", flush=True)
                traceback.print_exc()

    print(f"\n[compress] done: {counter['ok']}/{len(task_dirs)} trajectories cached in {compressor.cache_dir}")


if __name__ == "__main__":
    main()
