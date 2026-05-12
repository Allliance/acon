"""Selection-based history-compression baselines.

These strategies don't call any LLM; they pick / mask turns from prior history
to fit a fixed token budget. All operate on the segment of the session that
appears *before* the last-k preserved turns (same input as the LLM-based
HistoryOptimizer would receive).

Each strategy returns a list of messages (alternating assistant/user) that will
be inserted into the rebuilt session right before the preserved last-k turns.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List

Message = Dict[str, str]
TokenCounter = Callable[[str], int]


MASK_OBS_PLACEHOLDER = "[OBSERVATION MASKED]"
MASK_ACTION_PLACEHOLDER = "[ACTION MASKED]"


def _pair_turns(history: List[Message]) -> List[List[Message]]:
    """Group a flat assistant/user message list into (assistant, user) pairs.

    Stray messages that don't fit a clean pair (e.g. trailing assistant with no
    user, or leading user) are emitted as singleton groups so nothing is lost.
    """
    pairs: List[List[Message]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        if msg["role"] == "assistant" and i + 1 < len(history) and history[i + 1]["role"] == "user":
            pairs.append([msg, history[i + 1]])
            i += 2
        else:
            pairs.append([msg])
            i += 1
    return pairs


def _pair_tokens(pair: List[Message], count_tokens: TokenCounter) -> int:
    return sum(count_tokens(m["content"]) for m in pair)


def fifo_select(history: List[Message], budget: int, count_tokens: TokenCounter) -> List[Message]:
    """Keep the most recent turns that fit in `budget` tokens (FIFO discards the oldest)."""
    pairs = _pair_turns(history)
    kept: List[List[Message]] = []
    used = 0
    for pair in reversed(pairs):
        cost = _pair_tokens(pair, count_tokens)
        if used + cost > budget:
            break
        kept.append(pair)
        used += cost
    kept.reverse()
    out: List[Message] = []
    for pair in kept:
        out.extend(pair)
    return out


def _mask_select(
    history: List[Message],
    budget: int,
    count_tokens: TokenCounter,
    mask_role: str,
    placeholder: str,
) -> List[Message]:
    """Shared masking: keep recent pairs (within budget after masking) and replace
    the masked role's content with a placeholder."""
    pairs = _pair_turns(history)

    def _mask_pair(pair: List[Message]) -> List[Message]:
        return [
            {"role": m["role"], "content": placeholder if m["role"] == mask_role else m["content"]}
            for m in pair
        ]

    kept: List[List[Message]] = []
    used = 0
    for pair in reversed(pairs):
        masked = _mask_pair(pair)
        cost = _pair_tokens(masked, count_tokens)
        if used + cost > budget:
            break
        kept.append(masked)
        used += cost
    kept.reverse()
    out: List[Message] = []
    for pair in kept:
        out.extend(pair)
    return out


def mask_obs_select(history: List[Message], budget: int, count_tokens: TokenCounter) -> List[Message]:
    """Keep recent turns within budget; replace observation (user) content with a placeholder."""
    return _mask_select(history, budget, count_tokens, mask_role="user", placeholder=MASK_OBS_PLACEHOLDER)


def mask_action_select(history: List[Message], budget: int, count_tokens: TokenCounter) -> List[Message]:
    """Keep recent turns within budget; replace action (assistant) content with a placeholder."""
    return _mask_select(history, budget, count_tokens, mask_role="assistant", placeholder=MASK_ACTION_PLACEHOLDER)


def random_select(
    history: List[Message],
    budget: int,
    count_tokens: TokenCounter,
    seed: int = 0,
) -> List[Message]:
    """Random subsample of pairs that fit within `budget` tokens.

    Pairs are sampled without replacement in random order; selected pairs are
    re-sorted to preserve chronological order in the rebuilt session.
    """
    pairs = _pair_turns(history)
    indexed = list(enumerate(pairs))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    selected: List[int] = []
    used = 0
    for idx, pair in indexed:
        cost = _pair_tokens(pair, count_tokens)
        if used + cost > budget:
            continue
        selected.append(idx)
        used += cost
    selected.sort()
    out: List[Message] = []
    for idx in selected:
        out.extend(pairs[idx])
    return out


SELECTION_STRATEGIES = {
    "fifo": fifo_select,
    "mask_obs": mask_obs_select,
    "mask_action": mask_action_select,
    "random": random_select,
}


def apply_selection_strategy(
    name: str,
    history: List[Message],
    budget: int,
    count_tokens: TokenCounter,
    seed: int = 0,
) -> List[Message]:
    if name not in SELECTION_STRATEGIES:
        raise ValueError(f"Unknown selection strategy: {name!r}. Known: {list(SELECTION_STRATEGIES)}")
    fn = SELECTION_STRATEGIES[name]
    if name == "random":
        return fn(history, budget, count_tokens, seed=seed)
    return fn(history, budget, count_tokens)
