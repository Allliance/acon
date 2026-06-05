"""Offline smoke test: exercises parsing/segmentation/pipeline wiring without a
live vLLM server, using stub clients. Run:

    cd experiments/appworld && python -m perplexity.selftest \
        --trajectory_dir trajectory_qa/Qwen3.5_35B_A3B_dev_full/dev
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perplexity.pipeline import aggregate, score_trajectory  # noqa: E402
from perplexity.segmentation import load_trajectory, segment_trajectory  # noqa: E402


class StubAgent:
    """Deterministic fake: ~4 chars/token; nll grows with context length so the
    compressed (shorter) context yields lower perplexity — just to verify plumbing.
    """

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def tokenize(self, *, text=None, messages=None, **kw):
        payload = text if text is not None else "".join(m["content"] for m in messages)
        return list(range(self.count_tokens(payload)))

    def action_nll(self, context_messages, action_text, action_token_ids=None):
        ctx_len = sum(self.count_tokens(m["content"]) for m in context_messages)
        n = len(action_token_ids) if action_token_ids else self.count_tokens(action_text)
        nll = 1.0 + ctx_len / 50000.0
        return {"nll": nll, "ppl": math.exp(nll), "n_tokens": n}


class StubCompressor:
    name = "stub"
    strategy = "summary"
    budget = 2048

    def begin_trajectory(self, traj, segments):
        pass

    def build_context(self, system, task, prefix_steps, count_tokens, boundary=None):
        summary = f"[summary of {len(prefix_steps)} steps]"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{task}\n\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>"},
        ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory_dir", required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--max_segment_tokens", type=int, default=6000)
    args = p.parse_args()

    from perplexity.pipeline import FullContextCache

    agent, compressor = StubAgent(), StubCompressor()
    full_cache = FullContextCache(agent, cache_dir=None)
    task_dirs = sorted(
        os.path.join(args.trajectory_dir, d)
        for d in os.listdir(args.trajectory_dir)
        if d.startswith("task_")
    )[: args.limit]

    scores = []
    for task_dir in task_dirs:
        traj = load_trajectory(task_dir)
        segments = segment_trajectory(traj, agent.count_tokens, args.max_segment_tokens)
        # Invariant: segments partition all steps in order.
        flat = [i for seg in segments for i in seg]
        assert flat == list(range(len(traj.steps))), "segments must partition steps"
        # Invariant: each segment within budget unless it is a single step.
        for seg in segments:
            tot = sum(traj.steps[i].n_tokens for i in seg)
            assert len(seg) == 1 or tot <= args.max_segment_tokens, (seg, tot)
        score = score_trajectory(traj, segments, agent, compressor, full_cache)
        scores.append(score)
        n1 = score["mean_next1_nll_diff"]
        n5 = score["mean_next5_nll_diff"]
        print(f"{traj.task_id}: {len(traj.steps)} steps -> {score['num_segments']} segs, "
              f"{score['num_scored_boundaries']} scored boundaries, "
              f"next1_nll={n1 if n1 is None else round(n1,3)}, next5_nll={n5 if n5 is None else round(n5,3)}")

    print("\nAggregate:", aggregate(scores))
    print("OK: parsing, segmentation, and pipeline wiring pass.")


if __name__ == "__main__":
    main()
