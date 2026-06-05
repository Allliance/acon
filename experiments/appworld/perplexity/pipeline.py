"""Core scoring: compressor-induced perplexity with a preserved recent segment.

For a trajectory split into segments S_0, S_1, ..., S_{n-1} we sweep the
**verbatim segment** index ``i`` from 1 (the second segment) to ``n-2``. At each:

  * segments ``S_0..S_{i-1}`` are compressed under the budget (the *old* prefix),
  * segment ``S_i`` is kept **verbatim** (not counted against the budget),
  * we score the agent's perplexity on the next actions that follow segment
    ``S_i`` in the real trajectory.

Two contexts per scored action ``a`` (at trajectory step ``s``):

  * full       : system + task + every real step before ``s`` (verbatim)
  * compressed : system + (task + compressed old prefix) + verbatim S_i
                 + the real steps between S_i and ``s`` (verbatim)

Only the *old* prefix differs between the two; segment ``S_i`` and everything
after it are identical/verbatim, so the difference isolates the effect of
compressing the old history on near-future action prediction.

Metrics (induced = compressed − full):
  * next-1 : the first action after S_i.
  * next-5 : the next up-to-5 actions, token-weighted into one ppl/nll.

The full-context action scores depend only on the trajectory + step index (not on
the compressor), so they are computed once and cached on disk for reuse across
compressors.
"""

from __future__ import annotations

import json
import math
import os
from statistics import mean
from typing import Dict, List, Optional, Tuple

from .compressor import BaseCompressor
from .segmentation import Trajectory, step_messages
from .vllm_client import VLLMClient


# --------------------------------------------------------------------------- #
# Full-context action scores: compressor-independent, cached on disk.
# --------------------------------------------------------------------------- #
class FullContextCache:
    """Caches the full-context (uncompressed) action NLL per (task, step).

    The full-context score for the action at step ``s`` is the same regardless of
    which compressor we evaluate, so all compressors share this cache.
    """

    def __init__(self, agent: VLLMClient, cache_dir: Optional[str] = None):
        self.agent = agent
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._mem: Dict[str, Dict[int, Optional[Dict]]] = {}

    def _path(self, task_id: str) -> Optional[str]:
        return os.path.join(self.cache_dir, f"{task_id}.json") if self.cache_dir else None

    def _load(self, task_id: str) -> Dict[int, Optional[Dict]]:
        if task_id in self._mem:
            return self._mem[task_id]
        data: Dict[int, Optional[Dict]] = {}
        path = self._path(task_id)
        if path and os.path.exists(path):
            try:
                raw = json.load(open(path))
                data = {int(k): v for k, v in raw.items()}
            except Exception:
                data = {}
        self._mem[task_id] = data
        return data

    def _save(self, task_id: str) -> None:
        path = self._path(task_id)
        if not path:
            return
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self._mem[task_id].items()}, f)

    def action_nll(
        self, traj: Trajectory, step_index: int, action_token_ids: List[int]
    ) -> Optional[Dict]:
        """Return {'nll','ppl','n_tokens'} for the action at ``step_index`` under
        the full real context, or ``None`` if it cannot be scored (e.g. context
        window overflow). Result (including a None for un-scorable steps) is cached.
        """
        cache = self._load(traj.task_id)
        if step_index in cache:
            return cache[step_index]

        ctx = [
            {"role": "system", "content": traj.system},
            {"role": "user", "content": traj.task},
        ]
        for j in range(step_index):
            ctx += step_messages(traj.steps[j])
        try:
            result = self.agent.action_nll(
                ctx, traj.steps[step_index].action, action_token_ids=action_token_ids
            )
        except Exception:
            result = None
        cache[step_index] = result
        self._save(traj.task_id)
        return result


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def _steps_to_messages(traj: Trajectory, indices: range | List[int]) -> List[Dict]:
    msgs: List[Dict] = []
    for j in indices:
        msgs += step_messages(traj.steps[j])
    return msgs


def _token_weighted(actions: List[Dict], full_key: str, comp_key: str) -> Dict:
    """Token-weighted nll/ppl aggregate across a list of scored actions."""
    n = sum(a["n_action_tokens"] for a in actions)
    full_nll = sum(a[full_key] * a["n_action_tokens"] for a in actions) / n
    comp_nll = sum(a[comp_key] * a["n_action_tokens"] for a in actions) / n
    return {
        "n_actions": len(actions),
        "n_tokens": n,
        "full_nll": full_nll,
        "compressed_nll": comp_nll,
        "full_ppl": math.exp(full_nll),
        "compressed_ppl": math.exp(comp_nll),
        "nll_diff": comp_nll - full_nll,
        "ppl_diff": math.exp(comp_nll) - math.exp(full_nll),
    }


def score_trajectory(
    traj: Trajectory,
    segments: List[List[int]],
    agent: VLLMClient,
    compressor: BaseCompressor,
    full_cache: FullContextCache,
    max_next_actions: int = 5,
) -> Dict:
    """Score one trajectory under the preserved-recent-segment scheme."""
    compressor.begin_trajectory(traj, segments)

    boundaries: List[Dict] = []
    n_steps = len(traj.steps)

    # Verbatim segment i: 1 (second segment) .. n-2 (needs a following action).
    for i in range(1, len(segments) - 1):
        old_idxs = [j for seg in segments[:i] for j in seg]   # segments 0..i-1
        old_steps = [traj.steps[j] for j in old_idxs]
        seg_i = segments[i]
        end_i = seg_i[-1]

        try:
            # Compressed *old* prefix (system + task[+summary] [+selected turns]).
            # For the cumulative compressor, summary of segments 0..i-1 == summary[i-1].
            prefix_ctx = compressor.build_context(
                traj.system, traj.task, old_steps, agent.count_tokens, boundary=i - 1
            )
            # Append the verbatim recent segment S_i.
            base_comp = prefix_ctx + _steps_to_messages(traj, seg_i)
            prefix_tokens = sum(agent.count_tokens(m["content"]) for m in prefix_ctx)
        except Exception as e:
            boundaries.append({"verbatim_segment": i, "error": f"prefix: {e}", "actions": []})
            continue

        actions: List[Dict] = []
        for k in range(1, max_next_actions + 1):
            s = end_i + k
            if s >= n_steps:
                break
            action_text = traj.steps[s].action
            action_ids = agent.tokenize(text=action_text, add_special_tokens=False)
            if not action_ids:
                continue

            full = full_cache.action_nll(traj, s, action_ids)
            if full is None:
                continue  # full context overflowed the window; skip this action
            try:
                # Compressed context = base + verbatim intervening steps (S_i .. s-1).
                comp_ctx = base_comp + _steps_to_messages(traj, range(end_i + 1, s))
                comp = agent.action_nll(comp_ctx, action_text, action_token_ids=action_ids)
            except Exception:
                continue

            actions.append({
                "k": k,
                "step": s,
                "n_action_tokens": full["n_tokens"],
                "full_nll": full["nll"],
                "full_ppl": full["ppl"],
                "compressed_nll": comp["nll"],
                "compressed_ppl": comp["ppl"],
            })

        if not actions:
            boundaries.append({"verbatim_segment": i, "prefix_tokens": prefix_tokens, "actions": []})
            continue

        next1 = next((a for a in actions if a["k"] == 1), None)
        boundaries.append({
            "verbatim_segment": i,
            "old_segments": list(range(i)),
            "prefix_tokens": prefix_tokens,
            "actions": actions,
            "next1": {
                "full_nll": next1["full_nll"], "compressed_nll": next1["compressed_nll"],
                "full_ppl": next1["full_ppl"], "compressed_ppl": next1["compressed_ppl"],
                "nll_diff": next1["compressed_nll"] - next1["full_nll"],
                "ppl_diff": next1["compressed_ppl"] - next1["full_ppl"],
            } if next1 else None,
            "next5": _token_weighted(actions, "full_nll", "compressed_nll"),
        })

    # Trajectory-level means over boundaries that produced each metric.
    def _traj_mean(metric: str, field: str) -> Optional[float]:
        vals = [b[metric][field] for b in boundaries if b.get(metric)]
        return mean(vals) if vals else None

    return {
        "task_id": traj.task_id,
        "num_segments": len(segments),
        "num_boundaries": len(boundaries),
        "num_scored_boundaries": sum(1 for b in boundaries if b.get("actions")),
        "boundaries": boundaries,
        "mean_next1_nll_diff": _traj_mean("next1", "nll_diff"),
        "mean_next1_ppl_diff": _traj_mean("next1", "ppl_diff"),
        "mean_next5_nll_diff": _traj_mean("next5", "nll_diff"),
        "mean_next5_ppl_diff": _traj_mean("next5", "ppl_diff"),
    }


def aggregate(trajectory_scores: List[Dict]) -> Dict:
    """Average per-trajectory means into the final compressor scores."""
    out = {"num_trajectories": len(trajectory_scores)}
    for metric in ("next1_nll_diff", "next1_ppl_diff", "next5_nll_diff", "next5_ppl_diff"):
        key = f"mean_{metric}"
        vals = [s[key] for s in trajectory_scores if s.get(key) is not None]
        out[f"compressor_{metric}"] = mean(vals) if vals else None
    out["num_trajectories_scored"] = sum(
        1 for s in trajectory_scores if s.get("mean_next1_nll_diff") is not None
    )
    return out
