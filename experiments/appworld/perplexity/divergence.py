"""Trajectory divergence: does the agent's speculative next-5-action *plan*
diverge when conditioned on a **compressed** prefix instead of the **full**
history?

For each preserve-recent-segment boundary ``i`` (segments ``0..i-1`` are
compressed, segment ``S_i`` is kept verbatim) we prompt the agent to lay out its
next 5 Python code steps *in one shot* (single-shot speculative plan, no
intervening execution outputs), then ask a judge (gemini) how similar that plan
is to the actions the agent **actually** took in the real trajectory.

Conditions scored per boundary (each vs the same real next-5 actions):

  * ``self``        — plan from the FULL real prefix          (self-consistency)
  * ``cumulative``  — plan from the cumulative LLM summary    (cached)
  * ``mask_obs`` / ``fifo`` / ``random`` — plan from a selection-compressed prefix

The boundary structure, segmentation, and cumulative summaries are read from the
cached perplexity artifacts (``perplexity/compressions/<tag>/<task>.json``), so
no compression-LLM calls are made here. Only agent generations (speculative
plans) and judge calls are new, and both are cached on disk for cheap re-runs.

Reported numbers (mean judge similarity 0-100 over boundaries):
  self, cumulative, mask_obs, fifo, random
"""

from __future__ import annotations

import json
import os
import re
from statistics import mean
from typing import Callable, Dict, List, Optional

from productive_agents.ctxopt.selection_strategies import apply_selection_strategy

from .segmentation import Trajectory, step_messages

Message = Dict[str, str]
TokenCounter = Callable[[str], int]

SELECTION_CONDITIONS = ("mask_obs", "fifo", "random")
CONDITIONS = ("self", "cumulative") + SELECTION_CONDITIONS

# Instruction appended to the context to elicit a single-shot 5-action plan.
PLAN_INSTRUCTION = """[META-INSTRUCTION] Do not wait for any execution output. Predict the next {n} Python code blocks you would run next to continue the task, in order — i.e. forecast your own next {n} steps.

Write each block as the real code you would execute (API calls, computation, control flow). Do NOT fabricate, hardcode, or guess the contents of any output you have not seen yet. If a step uses a previous step's result, just write code that operates on that result (e.g. iterate over / index into the returned object).

Reply using EXACTLY this template and nothing else (no preamble, no prose outside the blocks). All {n} "### NEXT ACTION k" headers are REQUIRED — emit every one even if you are unsure about the later steps:

### NEXT ACTION 1
<the python code you would run at step 1>
### NEXT ACTION 2
<the python code you would run at step 2>
...
### NEXT ACTION {n}
<the python code you would run at step {n}>"""

_ACTION_SPLIT_RE = re.compile(r"^\s*#{1,6}\s*NEXT ACTION\s*\d+\s*$", re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"^\s*```(?:python|py)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


# --------------------------------------------------------------------------- #
# Context construction (mirrors perplexity/pipeline.py + compressor.py).
# --------------------------------------------------------------------------- #
def _summary_user_message(task: str, summary: str) -> Message:
    return {"role": "user", "content": f"{task}\n\n<HISTORY_SUMMARY>\n{summary}\n</HISTORY_SUMMARY>"}


def _steps_to_messages(traj: Trajectory, indices) -> List[Message]:
    msgs: List[Message] = []
    for j in indices:
        msgs += step_messages(traj.steps[j])
    return msgs


def build_context(
    traj: Trajectory,
    segments: List[List[int]],
    i: int,
    condition: str,
    summaries: Optional[List[str]],
    count_tokens: TokenCounter,
    budget: int = 2048,
    seed: int = 0,
) -> List[Message]:
    """Build the agent context up to (and including) the end of segment ``S_i``.

    ``self``        : system + task + every real step in segments 0..i (verbatim).
    ``cumulative``  : system + (task + <HISTORY_SUMMARY> summaries[i-1]) + verbatim S_i.
    selection       : system + task + selection(old prefix turns, budget) + verbatim S_i.
    """
    seg_i = segments[i]
    if condition == "self":
        all_idxs = [j for seg in segments[: i + 1] for j in seg]
        return [
            {"role": "system", "content": traj.system},
            {"role": "user", "content": traj.task},
        ] + _steps_to_messages(traj, all_idxs)

    if condition == "cumulative":
        if summaries is None:
            raise ValueError("cumulative condition needs cached summaries")
        # Prefix = segments 0..i-1, whose cumulative summary is summaries[i-1].
        prefix = [
            {"role": "system", "content": traj.system},
            _summary_user_message(traj.task, summaries[i - 1]),
        ]
        return prefix + _steps_to_messages(traj, seg_i)

    if condition in SELECTION_CONDITIONS:
        old_idxs = [j for seg in segments[:i] for j in seg]
        history_msgs = _steps_to_messages(traj, old_idxs)
        selected = apply_selection_strategy(condition, history_msgs, budget, count_tokens, seed=seed)
        prefix = [
            {"role": "system", "content": traj.system},
            {"role": "user", "content": traj.task},
        ] + selected
        return prefix + _steps_to_messages(traj, seg_i)

    raise ValueError(f"unknown condition: {condition}")


def build_summary_context(traj, segments, i, compressor, count_tokens):
    """Context for a *full-mode* (non-recurrent) summary compressor.

    The whole old prefix (segments ``0..i-1``) is re-summarized in one LLM call
    by ``compressor`` (a SummaryCompressor); segment ``S_i`` is then appended
    verbatim. Returns ``(messages, summary_text)``.
    """
    old_steps = [traj.steps[j] for seg in segments[:i] for j in seg]
    prefix = compressor.build_context(traj.system, traj.task, old_steps, count_tokens)
    summary = ""
    if prefix and prefix[-1]["role"] == "user" and "<HISTORY_SUMMARY>" in prefix[-1]["content"]:
        summary = prefix[-1]["content"].split("<HISTORY_SUMMARY>", 1)[1].rsplit("</HISTORY_SUMMARY>", 1)[0].strip()
    messages = prefix + _steps_to_messages(traj, segments[i])
    return messages, summary


def build_context_from_summary(traj, segments, i, summary):
    """Rebuild a summary-compressor context from an already-cached summary string
    (no compressor LLM call): system + (task + <HISTORY_SUMMARY>) + verbatim S_i."""
    return [
        {"role": "system", "content": traj.system},
        _summary_user_message(traj.task, summary),
    ] + _steps_to_messages(traj, segments[i])


def with_plan_instruction(messages: List[Message], n: int) -> List[Message]:
    """Append the plan instruction, merging into a trailing user turn if present."""
    instr = PLAN_INSTRUCTION.format(n=n)
    messages = [dict(m) for m in messages]
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] = messages[-1]["content"] + "\n\n" + instr
    else:
        messages.append({"role": "user", "content": instr})
    return messages


# --------------------------------------------------------------------------- #
# Parsing the speculative plan.
# --------------------------------------------------------------------------- #
def parse_plan(text: str, n: int) -> List[str]:
    """Split a generated plan into up to ``n`` code blocks on the NEXT ACTION
    delimiters. Strips markdown fences. Falls back to the whole text as a single
    action if no delimiters are found."""
    if not text:
        return []
    parts = _ACTION_SPLIT_RE.split(text)
    # Everything before the first delimiter is preamble; drop it.
    blocks = [p.strip() for p in parts[1:]] if len(parts) > 1 else [text.strip()]
    cleaned: List[str] = []
    for b in blocks:
        if not b:
            continue
        m = _FENCE_RE.match(b)
        cleaned.append((m.group(1) if m else b).strip())
    return [c for c in cleaned if c][:n]
