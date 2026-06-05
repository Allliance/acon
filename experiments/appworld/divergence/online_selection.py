"""Online best-of-N compression selection for a *live* AppWorld agent run.

When the history optimizer is about to compress, the agent generates N candidate
compressions (summaries) instead of one, and this module picks the candidate that
perturbs the agent's near-future plan the least.

Method (``scorer="divergence"``, the one the user runs)
-------------------------------------------------------
At the compression boundary we form two kinds of context for the *agent* model:

  reference  : the UNCOMPRESSED session right before this compression (earlier
               compressions stay applied — only *this* compression is withheld).
  candidate  : the session that *would* be installed if a given candidate
               summary were used (system + rebuilt user prompt with the
               candidate's <HISTORY_SUMMARY> + the preserved recent turns).

For each, the agent forecasts its next ``n_actions`` Python code blocks in one
shot (no execution feedback), exactly as in the offline trajectory-divergence
study (``divergence.core.with_plan_instruction`` / ``parse_plan``). A judge then
scores each candidate plan against the *reference* plan (0-100, higher = less
divergence). The highest-scoring candidate summary is installed.

This is the online analogue of ``divergence/run.py``. The only difference is the
reference: mid-run we have no ground-truth future actions, so the reference is
the agent's own plan conditioned on the uncompressed history (what the agent
*would* do if it never compressed this turn).

Method (``scorer="rubric"``, implemented for comparison, not run yet)
--------------------------------------------------------------------
No plan generation. A judge scores each candidate summary directly against a
context-compression rubric tuned for AppWorld agentic tasks (state/variable
fidelity, progress & next-step continuity, faithfulness, error retention,
conciseness). See :class:`RubricJudge`.

Both scorers share the same Gemini backend and key handling as
``divergence.judge``.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Dict, List, Optional

from .core import parse_plan, with_plan_instruction
from .judge import GeminiJudge, load_gemini_key
from .vllm_client import VLLMClient

Message = Dict[str, str]


# --------------------------------------------------------------------------- #
# Rubric judge (scorer="rubric").
# --------------------------------------------------------------------------- #
_RUBRIC_SYSTEM = """You are a strict evaluator of HISTORY COMPRESSIONS for a long-horizon AI coding agent.

The agent solves a task by writing Python that calls task APIs (e.g. apis.spotify.*, apis.venmo.*, apis.phone.*), one code block ("action") per step, reacting to each execution output. Because its context window is bounded, its earlier history is periodically replaced by a compact <HISTORY_SUMMARY>. The agent then continues the task from that summary plus a few recent verbatim turns. A GOOD summary lets the agent finish the task as if it still had the full history; a BAD summary makes it repeat work, lose state, or take wrong actions.

You are given:
  - the TASK,
  - the ORIGINAL HISTORY that is being compressed (the ground truth),
  - a CANDIDATE SUMMARY produced for that history.

Score the CANDIDATE SUMMARY on these rubric dimensions (each scored on its own scale; judge by what the agent will NEED next, not by surface wording):

1. STATE_FIDELITY (0-25): Does it preserve the concrete runtime values the next steps must reuse — ids, access tokens, account names, phone numbers, emails, amounts, file paths, list contents, last page_index/page_limit — accurately and without inventing or corrupting any value? Missing or wrong critical values score low.
2. CONTINUITY (0-25): Can the agent recover the current sub-goal, what is already done, what remains (pending TODOs), and the correct immediate next action from this summary alone? Reward an accurate, ordered account of progress and remaining work.
3. FAITHFULNESS (0-20): Is everything consistent with the ORIGINAL HISTORY, with nothing fabricated, altered, or contradictory?
4. ERROR_RETENTION (0-15): Does it keep the lessons that prevent repeated mistakes — failed approaches, error causes, and learned constraints (e.g. "login needs phone number not email", "paginate until empty page")?
5. CONCISENESS (0-15): Is it focused and within budget — no raw API/log dumps, no redundancy, no filler — while keeping everything essential? Penalize bloat AND over-compression that drops needed facts.

OVERALL must equal the sum of the five dimension scores (0-100).

Respond with ONLY a JSON object:
{"state_fidelity": <0-25>, "continuity": <0-25>, "faithfulness": <0-20>, "error_retention": <0-15>, "conciseness": <0-15>, "overall": <0-100>, "reasoning": "<one concise sentence>"}"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_DIMS = ("state_fidelity", "continuity", "faithfulness", "error_retention", "conciseness")
_DIM_MAX = {"state_fidelity": 25, "continuity": 25, "faithfulness": 20, "error_retention": 15, "conciseness": 15}


class RubricJudge:
    """Gemini judge that scores a single compression against the rubric above."""

    def __init__(self, model: str = "gemini-3.5-flash", api_key: Optional[str] = None,
                 max_retries: int = 4, thinking_budget: Optional[int] = None):
        from google import genai  # local import: only needed for the judge

        self.model = model
        self.max_retries = max_retries
        # thinking_budget: None -> model default (3.5-flash thinks; flash-lite does
        # not). 0 -> force off. -1 -> dynamic (let the model decide). >0 -> fixed cap.
        self.thinking_budget = thinking_budget
        self._genai = genai
        self.client = genai.Client(api_key=api_key or load_gemini_key())

    def score(self, task: str, history_text: str, summary: str) -> Dict:
        """Return {'score': int|None (overall 0-100), 'dimensions': {...}, 'reasoning', 'raw'}."""
        from google.genai import types

        user = (
            f"TASK:\n{task}\n\n"
            f"ORIGINAL HISTORY (being compressed):\n{history_text}\n\n"
            f"CANDIDATE SUMMARY (to evaluate):\n{summary}\n\n"
            "Score the CANDIDATE SUMMARY against the rubric."
        )
        cfg_kwargs = dict(
            system_instruction=_RUBRIC_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
        )
        if self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        cfg = types.GenerateContentConfig(**cfg_kwargs)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=cfg
                )
                return self._parse((resp.text or "").strip())
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt + 0.5)
        return {"score": None, "dimensions": {}, "reasoning": f"judge_error: {last_err}", "raw": ""}

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
        if not isinstance(obj, dict):
            return {"score": None, "dimensions": {}, "reasoning": "unparseable", "raw": raw}
        dims: Dict[str, Optional[int]] = {}
        for d in _DIMS:
            try:
                dims[d] = max(0, min(_DIM_MAX[d], int(round(float(obj[d])))))
            except Exception:
                dims[d] = None
        # Prefer the model's OVERALL; fall back to the sum of present dims.
        overall = obj.get("overall")
        try:
            overall = max(0, min(100, int(round(float(overall)))))
        except Exception:
            present = [v for v in dims.values() if v is not None]
            overall = sum(present) if len(present) == len(_DIMS) else None
        return {
            "score": overall,
            "dimensions": dims,
            "reasoning": str(obj.get("reasoning", "")),
            "raw": raw,
        }


# --------------------------------------------------------------------------- #
# Selector.
# --------------------------------------------------------------------------- #
class CompressionSelector:
    """Picks the least-divergent (or highest-rubric) of N candidate compressions.

    Parameters
    ----------
    agent_base_url, agent_model : the agent vLLM endpoint used to forecast plans
        (divergence scorer only). Resolved from ``VLLM_BASE_URL`` / the agent
        config by the caller.
    scorer : "divergence" | "rubric".
    n_actions : how many next code blocks the agent forecasts per plan.
    max_gen_tokens : decode cap for a forecasted plan.
    agent_temperature : decode temperature for plan forecasting (0.0 = greedy,
        matching the offline divergence study).
    agent_enable_thinking : Qwen3 hidden-thinking toggle for plan forecasting
        (False matches the offline study and yields a clean, parseable plan).
    judge_model : Gemini model id for both judges.
    """

    def __init__(
        self,
        agent_base_url: Optional[str] = None,
        agent_model: Optional[str] = None,
        *,
        scorer: str = "divergence",
        n_actions: int = 5,
        max_gen_tokens: int = 2048,
        agent_temperature: float = 0.0,
        agent_enable_thinking: bool = False,
        judge_model: str = "gemini-3.5-flash",
        judge_thinking_budget: Optional[int] = None,
        seed: int = 42,
        agent: Optional[VLLMClient] = None,
        judge: Optional[object] = None,
    ):
        if scorer not in ("divergence", "rubric"):
            raise ValueError(f"unknown scorer: {scorer!r}")
        self.scorer = scorer
        self.n_actions = n_actions
        self.max_gen_tokens = max_gen_tokens
        self.agent_temperature = agent_temperature
        self.agent_enable_thinking = agent_enable_thinking
        self.seed = seed

        if scorer == "divergence":
            self.agent = agent or VLLMClient(base_url=agent_base_url, model=agent_model)
            self.judge = judge or GeminiJudge(model=judge_model)
            self.rubric_judge = None
        else:
            self.agent = agent  # may be None for rubric mode
            self.judge = None
            self.rubric_judge = judge or RubricJudge(
                model=judge_model, thinking_budget=judge_thinking_budget
            )

    # -- plan forecasting -----------------------------------------------------
    def _forecast_plan(self, messages: List[Message]) -> Dict:
        """Agent forecasts its next ``n_actions`` code blocks given ``messages``.

        Returns {'predicted': List[str], 'plan_text': str}. Retries transient
        endpoint errors with backoff (mirrors divergence/run.py).
        """
        ctx = with_plan_instruction([dict(m) for m in messages], self.n_actions)
        plan_text = ""
        last_err = None
        for attempt in range(4):
            try:
                plan_text = self.agent.chat(
                    ctx,
                    max_tokens=self.max_gen_tokens,
                    temperature=self.agent_temperature,
                    enable_thinking=self.agent_enable_thinking,
                    seed=self.seed,
                )
                last_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt + 0.5)
        if last_err is not None:
            raise last_err
        return {"predicted": parse_plan(plan_text, self.n_actions), "plan_text": plan_text}

    # -- public API -----------------------------------------------------------
    def select(
        self,
        task: str,
        reference_messages: List[Message],
        history_text: str,
        candidates: List[Dict],
    ) -> Dict:
        """Score ``candidates`` and return a selection record.

        ``candidates`` is a list of ``{"summary": str, "messages": List[Message]}``.
        ``reference_messages`` is the uncompressed session (divergence scorer).
        ``history_text`` is the plain-text history being compressed (rubric scorer).

        The returned dict always has ``best_index`` (int) and ``scores``
        (list aligned with ``candidates``); plus scorer-specific detail.
        """
        if not candidates:
            raise ValueError("no candidates to select from")
        if self.scorer == "divergence":
            return self._select_divergence(task, reference_messages, candidates)
        return self._select_rubric(task, history_text, candidates)

    @staticmethod
    def _argmax(scores: List[Optional[float]]) -> int:
        """First index of the max score; ``None`` scores count as -inf."""
        best_i, best_v = 0, float("-inf")
        for i, s in enumerate(scores):
            v = float(s) if s is not None else float("-inf")
            if v > best_v:
                best_i, best_v = i, v
        return best_i

    def _select_divergence(self, task, reference_messages, candidates) -> Dict:
        ref = self._forecast_plan(reference_messages)
        ref_plan = ref["predicted"]

        cand_records: List[Dict] = []
        scores: List[Optional[float]] = []
        for idx, cand in enumerate(candidates):
            rec: Dict[str, object] = {"index": idx}
            try:
                plan = self._forecast_plan(cand["messages"])
                verdict = self.judge.score(task, reference=ref_plan, predicted=plan["predicted"])
                rec["predicted"] = plan["predicted"]
                rec["score"] = verdict["score"]
                rec["judge_reasoning"] = verdict["reasoning"]
            except Exception as e:  # noqa: BLE001
                rec["score"] = None
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["summary"] = cand.get("summary", "")
            scores.append(rec.get("score"))
            cand_records.append(rec)

        best_index = self._argmax(scores)
        return {
            "scorer": "divergence",
            "best_index": best_index,
            "scores": scores,
            "n_actions": self.n_actions,
            "reference_plan": ref_plan,
            "reference_plan_text": ref["plan_text"],
            "candidates": cand_records,
        }

    def _select_rubric(self, task, history_text, candidates) -> Dict:
        cand_records: List[Dict] = []
        scores: List[Optional[float]] = []
        for idx, cand in enumerate(candidates):
            rec: Dict[str, object] = {"index": idx, "summary": cand.get("summary", "")}
            try:
                verdict = self.rubric_judge.score(task, history_text, cand["summary"])
                rec["score"] = verdict["score"]
                rec["dimensions"] = verdict["dimensions"]
                rec["judge_reasoning"] = verdict["reasoning"]
            except Exception as e:  # noqa: BLE001
                rec["score"] = None
                rec["error"] = f"{type(e).__name__}: {e}"
            scores.append(rec.get("score"))
            cand_records.append(rec)

        best_index = self._argmax(scores)
        return {
            "scorer": "rubric",
            "best_index": best_index,
            "scores": scores,
            "candidates": cand_records,
        }
