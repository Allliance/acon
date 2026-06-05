"""Measure compressor-induced perplexity over a directory of trajectories.

Example:
    cd experiments/appworld
    export VLLM_BASE_URL=http://<agent-host>:8000/v1
    export VLLM_COMPRESSOR_BASE_URL=http://<compressor-host>:8000/v1
    python -m perplexity.run \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev \
        --compressor_config configs/context_opt/qwen3p5_27b_prompting_t6k_b2k.yaml \
        --output perplexity/outputs/qwen27b_t6k.json

Must run inside the `smolagents` conda env (provides yaml/openai/transformers
and `productive_agents`). See perplexity/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Ensure sibling-package imports work whether invoked as `-m perplexity.run`
# or `python perplexity/run.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity.compressor import make_compressor  # noqa: E402
from perplexity.pipeline import FullContextCache, aggregate, score_trajectory  # noqa: E402
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
    p.add_argument("--trajectory_dir", required=True,
                   help="Directory containing task_<id> subdirs (each with llm_history.json)")
    p.add_argument("--compressor_config", required=True,
                   help="YAML compressor config under configs/context_opt/")
    p.add_argument("--agent_model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-35B-A3B"),
                   help="Served agent model name (the model whose perplexity we measure)")
    p.add_argument("--agent_base_url", default=os.environ.get("VLLM_BASE_URL"),
                   help="Agent vLLM endpoint (defaults to $VLLM_BASE_URL)")
    p.add_argument("--compressor_base_url", default=os.environ.get("VLLM_COMPRESSOR_BASE_URL"),
                   help="Compressor vLLM endpoint (defaults to $VLLM_COMPRESSOR_BASE_URL or config)")
    p.add_argument("--prompt_dir", default=None,
                   help="Compression prompt-template dir (default: appworld prompts/context_opt)")
    p.add_argument("--max_segment_tokens", type=int, default=6000,
                   help="Max tokens per segment (default 6000)")
    p.add_argument("--max_next_actions", type=int, default=5,
                   help="Score the next up-to-N actions after the verbatim segment (default 5)")
    p.add_argument("--full_cache_dir", default=None,
                   help="Dir to cache full-context action scores (shared across compressors). "
                        "Default: perplexity/outputs/full_cache/<trajectory_dir>")
    p.add_argument("--compression_budget", type=int, default=None,
                   help="Override the config's compression budget (tokens)")
    p.add_argument("--cache_dir", default=None,
                   help="Cumulative-compression cache dir (default under perplexity/compressions/<tag>)")
    p.add_argument("--overwrite_cache", action="store_true",
                   help="Regenerate cumulative compressions even if cached")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N task dirs")
    p.add_argument("--output", default=None, help="Where to write the JSON report")
    p.add_argument("--debug", action="store_true", help="Verbose compressor output")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.agent_base_url:
        raise SystemExit("No agent endpoint: pass --agent_base_url or set VLLM_BASE_URL")

    agent = VLLMClient(base_url=args.agent_base_url, model=args.agent_model)
    compressor = make_compressor(
        config_path=args.compressor_config,
        max_segment_tokens=args.max_segment_tokens,
        compression_budget=args.compression_budget,
        prompt_dir=args.prompt_dir,
        # Compressor shares the agent endpoint unless explicitly overridden.
        compressor_base_url=args.compressor_base_url or args.agent_base_url,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite_cache,
        count_tokens=agent.count_tokens,
        debug=args.debug,
    )

    # Full-context action scores are compressor-independent — share one cache.
    full_cache_dir = args.full_cache_dir or os.path.join(
        os.path.dirname(__file__), "outputs", "full_cache",
        args.trajectory_dir.strip("/").replace("/", "_"),
    )
    full_cache = FullContextCache(agent, cache_dir=full_cache_dir)

    task_dirs = find_task_dirs(args.trajectory_dir)
    if args.limit:
        task_dirs = task_dirs[: args.limit]
    print(f"[perplexity] {len(task_dirs)} trajectories | compressor={compressor.name} "
          f"| max_segment_tokens={args.max_segment_tokens} | next_actions={args.max_next_actions}")

    trajectory_scores = []
    for i, task_dir in enumerate(task_dirs, 1):
        name = os.path.basename(task_dir)
        try:
            traj = load_trajectory(task_dir)
            segments = segment_trajectory(traj, agent.count_tokens, args.max_segment_tokens)
            score = score_trajectory(
                traj, segments, agent, compressor, full_cache,
                max_next_actions=args.max_next_actions,
            )
            trajectory_scores.append(score)
            print(f"[{i}/{len(task_dirs)}] {name}: {score['num_segments']} segs, "
                  f"{score['num_scored_boundaries']} scored boundaries, "
                  f"next1_nll={score['mean_next1_nll_diff']}, next5_nll={score['mean_next5_nll_diff']}")
        except Exception as e:  # keep going; record the failure
            print(f"[{i}/{len(task_dirs)}] {name}: ERROR {e}")
            traceback.print_exc()
            trajectory_scores.append({"task_id": name, "error": str(e),
                                      "mean_next1_nll_diff": None, "mean_next5_nll_diff": None})

    summary = aggregate(trajectory_scores)
    report = {
        "config": {
            "trajectory_dir": args.trajectory_dir,
            "compressor_config": args.compressor_config,
            "compressor_name": compressor.name,
            "compressor_strategy": compressor.strategy,
            "compression_budget": compressor.budget,
            "agent_model": args.agent_model,
            "max_segment_tokens": args.max_segment_tokens,
            "max_next_actions": args.max_next_actions,
            "scheme": "preserve_recent_segment",
        },
        "summary": summary,
        "trajectories": trajectory_scores,
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
