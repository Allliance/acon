"""Load a saved full-context trajectory and split it into token-bounded segments.

A trajectory is the agent's ``llm_history.json`` — a chat transcript of the form

    [system, user(task), assistant(a1), user(o1), assistant(a2), user(o2), ...]

We reconstruct ``(action, observation)`` *steps* and greedily pack consecutive
steps into segments of at most ``max_tokens`` tokens (measured with the agent's
own tokenizer, so segment sizes match what the model actually sees).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class Step:
    action: str               # assistant turn content
    observation: Optional[str]  # following environment/user turn, if any
    n_tokens: int = 0          # token cost of action(+observation), filled in later


@dataclass
class Trajectory:
    task_id: str
    system: str
    task: str
    steps: List[Step]


def load_trajectory(task_dir: str) -> Trajectory:
    """Read ``<task_dir>/llm_history.json`` into a :class:`Trajectory`.

    ``llm_history.json`` is a list of sessions; a full-context run has a single
    session. If several are present we take the longest (the uncompressed one).
    """
    with open(os.path.join(task_dir, "llm_history.json")) as f:
        sessions = json.load(f)
    if not sessions:
        raise ValueError(f"empty llm_history.json in {task_dir}")
    messages = max(sessions, key=len) if isinstance(sessions[0], list) else sessions

    if messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"unexpected message layout in {task_dir}")
    system = messages[0]["content"]
    task = messages[1]["content"]

    steps: List[Step] = []
    i = 2
    while i < len(messages):
        if messages[i]["role"] != "assistant":
            i += 1
            continue
        action = messages[i]["content"]
        observation = None
        if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
            observation = messages[i + 1]["content"]
        steps.append(Step(action=action, observation=observation))
        i += 2

    return Trajectory(task_id=os.path.basename(task_dir.rstrip("/")), system=system, task=task, steps=steps)


def step_text(step: Step) -> str:
    """Flatten a step into the text used for token counting / compression."""
    parts = [f"ASSISTANT:\n{step.action}"]
    if step.observation is not None:
        parts.append(f"USER:\n{step.observation}")
    return "\n\n".join(parts)


def step_messages(step: Step) -> List[Dict[str, str]]:
    """The chat messages a step contributes to the agent's context."""
    msgs = [{"role": "assistant", "content": step.action}]
    if step.observation is not None:
        msgs.append({"role": "user", "content": step.observation})
    return msgs


def segment_trajectory(
    traj: Trajectory,
    count_tokens: Callable[[str], int],
    max_tokens: int = 6000,
) -> List[List[int]]:
    """Greedily pack steps into segments of at most ``max_tokens`` tokens.

    Returns a list of segments, each a list of step indices. A single step that
    exceeds ``max_tokens`` on its own becomes its own segment.
    """
    for step in traj.steps:
        step.n_tokens = count_tokens(step_text(step))

    segments: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0
    for idx, step in enumerate(traj.steps):
        if current and current_tokens + step.n_tokens > max_tokens:
            segments.append(current)
            current, current_tokens = [], 0
        current.append(idx)
        current_tokens += step.n_tokens
    if current:
        segments.append(current)
    return segments
