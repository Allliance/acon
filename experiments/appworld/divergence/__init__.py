"""Trajectory-divergence metric for long-horizon agents.

Measures how much an agent's near-future plan changes when it is conditioned on
a *compressed* prefix instead of the *full* history. At each preserve-recent
boundary the agent emits a single-shot N-action speculative plan; a judge
(gemini) scores 0-100 how close that plan is to the actions the agent actually
took. Lower score = more divergence induced by the compression.

Public API (see ``core.py``):
    build_context, build_summary_context, build_context_from_summary,
    with_plan_instruction, parse_plan, CONDITIONS, SELECTION_CONDITIONS

Runners (``python -m divergence.<name>``):
    run            self + cumulative + selection baselines (uses cached summaries)
    run_baselines  full-mode (non-recurrent) LLM-summary compressors
    run_naive      selection-only baselines (fifo/random/mask_obs), decoding sweep
    run_resample   re-decode LLM compressors reusing cached summaries
    score_next1    add a next-1-action score to an existing cache
"""

from .core import (  # noqa: F401
    CONDITIONS,
    SELECTION_CONDITIONS,
    build_context,
    build_context_from_summary,
    build_summary_context,
    parse_plan,
    with_plan_instruction,
)
