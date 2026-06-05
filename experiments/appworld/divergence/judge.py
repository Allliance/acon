"""Gemini judge: score how similar a *predicted* next-action sequence is to the
*reference* (real) one, on a 0-100 scale.

Uses the ``google-genai`` SDK with the ``gemini_key`` from
``configs/private_config.yaml``. The judge sees the task, the reference action
sequence (what the agent actually did), and the predicted sequence (the
speculative plan), and returns strict JSON ``{"score": int, "reasoning": str}``.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional

import yaml
from google import genai
from google.genai import types

# experiments/appworld/divergence/ -> acon/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PRIVATE_CONFIG = os.path.join(_REPO_ROOT, "configs", "private_config.yaml")

_SYSTEM = """You are a strict evaluator measuring TRAJECTORY DIVERGENCE for an AI coding agent.

The agent solves a task by writing Python code that calls task APIs (e.g. apis.spotify.*, apis.venmo.*), one code block ("action") per step, reacting to each execution output.

You are given:
  - the TASK,
  - a REFERENCE sequence: the actions the agent ACTUALLY took next,
  - a PREDICTED sequence: the actions a (possibly differently-conditioned) agent speculatively planned to take next.

Score how closely the PREDICTED sequence reproduces the REFERENCE sequence on a 0-100 scale:
  - 100 = functionally identical plan: same APIs/operations, same key arguments, same sub-goals, in the same order.
  - 70-99 = same overall intent and most of the same API calls, with minor differences in order, arguments, or granularity.
  - 40-69 = partially aligned: some shared steps or a related approach, but meaningful divergence in what is done.
  - 1-39 = largely different approach; only superficial overlap.
  - 0 = completely unrelated.

Judge by SEMANTICS, not surface text: ignore comments, variable names, print formatting, and harmless extra exploratory calls. What matters is which operations the agent performs, with what arguments, toward what sub-goal, in what order. Align the sequences position-by-position but do not over-penalize a one-step offset.

Respond with ONLY a JSON object: {"score": <integer 0-100>, "reasoning": "<one concise sentence>"}."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_gemini_key(path: str = _PRIVATE_CONFIG) -> str:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    key = cfg.get("gemini_key") or cfg.get("google_key")
    if not key:
        raise ValueError(f"no gemini_key in {path}")
    return key


def _format_actions(actions: List[str]) -> str:
    if not actions:
        return "(none)"
    return "\n\n".join(f"--- ACTION {k} ---\n{a}" for k, a in enumerate(actions, 1))


class GeminiJudge:
    def __init__(self, model: str = "gemini-3.5-flash", api_key: Optional[str] = None,
                 max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key or load_gemini_key())

    def score(self, task: str, reference: List[str], predicted: List[str]) -> Dict:
        """Return {'score': int|None, 'reasoning': str, 'raw': str}."""
        user = (
            f"TASK:\n{task}\n\n"
            f"REFERENCE ACTIONS (what the agent actually did next):\n{_format_actions(reference)}\n\n"
            f"PREDICTED ACTIONS (the speculative plan to evaluate):\n{_format_actions(predicted)}\n\n"
            "Score the PREDICTED sequence against the REFERENCE."
        )
        cfg = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
        )
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=cfg
                )
                return self._parse((resp.text or "").strip())
            except Exception as e:  # rate limit / transient: backoff and retry
                last_err = e
                time.sleep(2 ** attempt + 0.5)
        return {"score": None, "reasoning": f"judge_error: {last_err}", "raw": ""}

    @staticmethod
    def _parse(raw: str) -> Dict:
        obj = None
        for candidate in (raw, *(m.group(1) for m in _FENCE_RE.finditer(raw))):
            try:
                obj = json.loads(candidate)
                break
            except Exception:
                continue
        if obj is None:
            m = _OBJ_RE.search(raw)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    obj = None
        if not isinstance(obj, dict) or "score" not in obj:
            return {"score": None, "reasoning": "unparseable", "raw": raw}
        try:
            score = max(0, min(100, int(round(float(obj["score"])))))
        except Exception:
            return {"score": None, "reasoning": str(obj.get("reasoning", "")), "raw": raw}
        return {"score": score, "reasoning": str(obj.get("reasoning", "")), "raw": raw}
