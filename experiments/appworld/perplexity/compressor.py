"""Compressors that turn a prefix of trajectory steps into a compressed context.

Families, behind one interface (``begin_trajectory`` + ``build_context``):

* :class:`SelectionCompressor` — naive no-LLM baselines (``fifo``, ``mask_obs``,
  ``mask_action``, ``random``). Prefix turns are selected/masked to fit
  ``compression_budget`` tokens and inserted verbatim after the task message.

* :class:`SummaryCompressor` — LLM / LLMLingua summary. At each boundary the
  *whole* prefix is re-summarized.

* :class:`CumulativeSummaryCompressor` — LLM summary built **recurrently**:
  ``summary_0 = LLM(segment_0)`` and ``summary_i = LLM(prev=summary_{i-1},
  new=segment_i)``. The full chain is generated once per trajectory and cached
  to disk (``cache_dir/<task_id>.json``) so it can be reused later without
  re-calling the LLM.

All token budgets are measured with the agent's own tokenizer (``count_tokens``)
so "2k tokens" means 2k tokens of the context the agent sees. ``make_compressor``
selects the family from the config (``baseline_strategy`` / ``summary_mode``).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import yaml

from productive_agents.ctxopt.history_optimizer import HistoryOptimizer
from productive_agents.ctxopt.selection_strategies import apply_selection_strategy

from .segmentation import Step, Trajectory, step_messages

Message = Dict[str, str]
TokenCounter = Callable[[str], int]
SELECTION_STRATEGIES = {"fifo", "mask_obs", "mask_action", "random"}

# Repo root: experiments/appworld/perplexity/ -> acon/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_PROMPT_DIR = os.path.join(_REPO_ROOT, "experiments", "appworld", "prompts", "context_opt")
# Where cumulative compression caches live (inside the perplexity dir).
COMPRESSIONS_ROOT = os.path.join(os.path.dirname(__file__), "compressions")


def _segment_history_text(traj: Trajectory, step_indices: List[int]) -> str:
    """Flatten a group of steps into the ``USER:/ASSISTANT:`` text the templates expect."""
    parts: List[str] = []
    for i in step_indices:
        s = traj.steps[i]
        parts.append(f"ASSISTANT:\n{s.action}")
        if s.observation is not None:
            parts.append(f"USER:\n{s.observation}")
    return "\n\n".join(parts)


def _summary_user_message(task: str, summary: str) -> Message:
    return {"role": "user", "content": f"{task}\n\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>"}


class BaseCompressor:
    name: str
    strategy: str
    budget: Optional[int]

    def begin_trajectory(self, traj: Trajectory, segments: List[List[int]]) -> None:
        """Optional per-trajectory setup (e.g. build/load a cumulative chain)."""

    def build_context(
        self,
        system: str,
        task: str,
        prefix_steps: List[Step],
        count_tokens: TokenCounter,
        boundary: Optional[int] = None,
    ) -> List[Message]:
        """Return the compressed context messages (system + task + compressed prefix).

        ``boundary`` is the segment-boundary index (prefix covers segments
        ``0..boundary``); only stateful compressors need it.
        """
        raise NotImplementedError


class SelectionCompressor(BaseCompressor):
    def __init__(self, name: str, strategy: str, budget: int, seed: int = 0):
        self.name = name
        self.strategy = strategy
        self.budget = budget
        self.seed = seed

    def build_context(self, system, task, prefix_steps, count_tokens, boundary=None):
        history_msgs: List[Message] = []
        for s in prefix_steps:
            history_msgs += step_messages(s)
        selected = apply_selection_strategy(
            self.strategy, history_msgs, self.budget, count_tokens, seed=self.seed
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ] + selected


class SummaryCompressor(BaseCompressor):
    """Re-summarize the whole prefix at each boundary (non-recurrent)."""

    def __init__(self, name: str, optimizer: HistoryOptimizer, budget: Optional[int]):
        self.name = name
        self.strategy = "summary"
        self.budget = budget
        self.optimizer = optimizer

    def build_context(self, system, task, prefix_steps, count_tokens, boundary=None):
        parts: List[str] = []
        for s in prefix_steps:
            parts.append(f"ASSISTANT:\n{s.action}")
            if s.observation is not None:
                parts.append(f"USER:\n{s.observation}")
        summary = self.optimizer.process(
            task=task, history="\n\n".join(parts), prev_history_summary=None, raw_history=[]
        )
        summary = (summary or "").strip()
        return [{"role": "system", "content": system}, _summary_user_message(task, summary)]


class CumulativeSummaryCompressor(BaseCompressor):
    """Recurrent summary chain, generated once per trajectory and cached to disk.

    summary_0 = LLM(segment_0); summary_i = LLM(prev=summary_{i-1}, new=segment_i).
    At boundary ``b`` (prefix = segments 0..b) the context uses ``summary_b``.
    """

    def __init__(
        self,
        name: str,
        optimizer: HistoryOptimizer,
        budget: Optional[int],
        cache_dir: str,
        segmentation_meta: Dict,
        overwrite: bool = False,
        count_tokens: Optional[TokenCounter] = None,
    ):
        self.name = name
        self.strategy = "summary_cumulative"
        self.budget = budget
        self.optimizer = optimizer
        self.cache_dir = cache_dir
        self.segmentation_meta = segmentation_meta
        self.overwrite = overwrite
        self._count_tokens = count_tokens
        self._summaries: Optional[List[str]] = None
        self._task_id: Optional[str] = None
        os.makedirs(cache_dir, exist_ok=True)

    # -- caching --------------------------------------------------------------
    def _cache_path(self, task_id: str) -> str:
        return os.path.join(self.cache_dir, f"{task_id}.json")

    def _ct(self, text: Optional[str]) -> int:
        if not text:
            return 0
        if self._count_tokens is not None:
            return self._count_tokens(text)
        return self.optimizer.count_tokens(text)

    def _load_cache(self, traj: Trajectory, segments: List[List[int]]) -> Optional[List[str]]:
        path = self._cache_path(traj.task_id)
        if not os.path.exists(path) or self.overwrite:
            return None
        try:
            data = json.load(open(path))
        except Exception:
            return None
        cached_segments = [seg["step_indices"] for seg in data.get("segments", [])]
        same_segmentation = cached_segments == [list(s) for s in segments]
        same_budget = data.get("compressor", {}).get("budget") == self.budget
        if same_segmentation and same_budget:
            return [seg["summary"] for seg in data["segments"]]
        return None

    def compute_chain(self, traj: Trajectory, segments: List[List[int]]) -> List[Dict]:
        """Build the recurrent summary chain (pure: no instance-state mutation).

        Safe to call concurrently for different trajectories — it only issues
        independent LLM requests and reads ``self``. Returns the per-segment
        records (also what gets written to the cache).
        """
        records: List[Dict] = []
        prev: Optional[str] = None
        for idx, seg in enumerate(segments):
            seg_text = _segment_history_text(traj, seg)
            summary = self.optimizer.process(
                task=traj.task, history=seg_text, prev_history_summary=prev, raw_history=[]
            )
            summary = (summary or "").strip()
            records.append({
                "index": idx,
                "step_indices": list(seg),
                "prev_summary_index": idx - 1 if idx > 0 else None,
                "input_tokens": self._ct(seg_text),
                "prev_summary_tokens": self._ct(prev),
                "summary_tokens": self._ct(summary),
                "summary": summary,
            })
            prev = summary
        return records

    def write_cache(self, traj: Trajectory, segments: List[List[int]], records: List[Dict]) -> None:
        data = {
            "task_id": traj.task_id,
            "compressor": {
                "name": self.name,
                "model": self.optimizer.model_name,
                "mode": "cumulative",
                "budget": self.budget,
                "prompt_template": self.optimizer.history_template,
            },
            "segmentation": self.segmentation_meta,
            "num_segments": len(segments),
            "segments": records,
        }
        with open(self._cache_path(traj.task_id), "w") as f:
            json.dump(data, f, indent=2)

    def generate_trajectory(self, traj: Trajectory, segments: List[List[int]]) -> List[str]:
        """Load from cache, or compute + cache. Returns the summary chain. Thread-safe."""
        cached = self._load_cache(traj, segments)
        if cached is not None:
            return cached
        records = self.compute_chain(traj, segments)
        self.write_cache(traj, segments, records)
        return [r["summary"] for r in records]

    # -- interface ------------------------------------------------------------
    def begin_trajectory(self, traj, segments):
        self._summaries = self.generate_trajectory(traj, segments)
        self._task_id = traj.task_id

    def build_context(self, system, task, prefix_steps, count_tokens, boundary=None):
        if self._summaries is None:
            raise RuntimeError("begin_trajectory() must run before build_context()")
        if boundary is None:
            raise ValueError("CumulativeSummaryCompressor needs a boundary index")
        summary = self._summaries[boundary]
        return [{"role": "system", "content": system}, _summary_user_message(task, summary)]


def default_cache_tag(config_name: str, budget: Optional[int], max_segment_tokens: int) -> str:
    # The config name already encodes the budget (e.g. ..._b2048); only the
    # segmentation needs to be added. Budget correctness is enforced on cache
    # load via the stored compressor.budget (see _load_cache).
    return f"{config_name}_t{max_segment_tokens}"


def make_compressor(
    config_path: str,
    max_segment_tokens: int,
    compression_budget: Optional[int] = None,
    prompt_dir: Optional[str] = None,
    compressor_base_url: Optional[str] = None,
    cache_dir: Optional[str] = None,
    overwrite: bool = False,
    count_tokens: Optional[TokenCounter] = None,
    debug: bool = False,
) -> BaseCompressor:
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    if compression_budget is not None:
        config["compression_budget"] = compression_budget

    name = os.path.splitext(os.path.basename(config_path))[0]
    strategy = config.get("baseline_strategy", "none")
    budget = config.get("compression_budget")

    if strategy in SELECTION_STRATEGIES:
        if not budget:
            raise ValueError(f"{name}: selection baseline needs a compression_budget")
        return SelectionCompressor(name, strategy, int(budget), seed=int(config.get("random_seed", 0)))

    # LLM / LLMLingua summary compressor.
    config["history_prompt_dir"] = prompt_dir or config.get("history_prompt_dir", _DEFAULT_PROMPT_DIR)
    if compressor_base_url:
        config["compressor_base_url"] = compressor_base_url
    optimizer = HistoryOptimizer(config, debug_mode=debug)

    summary_mode = config.get("summary_mode", "full")
    if summary_mode == "cumulative":
        tag = default_cache_tag(name, budget, max_segment_tokens)
        resolved_cache_dir = cache_dir or os.path.join(COMPRESSIONS_ROOT, tag)
        return CumulativeSummaryCompressor(
            name, optimizer, budget, resolved_cache_dir,
            segmentation_meta={"max_segment_tokens": max_segment_tokens, "tokenizer": optimizer.model_name},
            overwrite=overwrite, count_tokens=count_tokens,
        )
    return SummaryCompressor(name, optimizer, budget)
