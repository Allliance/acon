"""Thin HTTP client for a vLLM OpenAI-compatible server.

Only two server capabilities are needed:

1. ``/tokenize`` — apply the model's chat template server-side and return token
   ids. This avoids depending on a local copy of the tokenizer / chat template
   (the agent model is only available as a served endpoint).
2. ``/v1/completions`` with ``echo=True`` — score the log-probability of a
   *given* token sequence (teacher forcing); used for perplexity-style scoring.

Everything goes through ``requests`` so we don't depend on a particular version
of the ``openai`` client supporting token-id prompts.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import requests


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "token-abc",
        timeout: int = 600,
    ):
        base = (base_url or "").strip().rstrip("/")
        if not base:
            raise ValueError("vLLM base_url is empty (set --agent_base_url or VLLM_BASE_URL)")
        if not base.startswith("http"):
            base = "http://" + base
        # Normalise into a server root (for /tokenize) and an api base (for /v1/...).
        if base.endswith("/v1"):
            self.api_base = base
            self.root = base[: -len("/v1")].rstrip("/")
        else:
            self.root = base
            self.api_base = base + "/v1"
        self.model = model
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self._count_cache: Dict[str, int] = {}

    # -- tokenization ---------------------------------------------------------
    def tokenize(
        self,
        *,
        text: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        add_generation_prompt: bool = False,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Return the model's token ids for either raw ``text`` or ``messages``.

        When ``messages`` is given the server applies the chat template; with
        ``add_generation_prompt=True`` the returned ids end exactly where the
        assistant turn would begin.
        """
        payload: Dict = {"model": self.model}
        if messages is not None:
            payload["messages"] = messages
            payload["add_generation_prompt"] = add_generation_prompt
        elif text is not None:
            payload["prompt"] = text
            payload["add_special_tokens"] = add_special_tokens
        else:
            raise ValueError("tokenize() needs either text= or messages=")
        r = requests.post(
            f"{self.root}/tokenize", json=payload, headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["tokens"]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        cached = self._count_cache.get(text)
        if cached is not None:
            return cached
        n = len(self.tokenize(text=text, add_special_tokens=False))
        self._count_cache[text] = n
        return n

    # -- generation -----------------------------------------------------------
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        seed: int = 42,
        sampling: Optional[Dict] = None,
    ) -> str:
        """Generate an assistant turn for ``messages`` (chat completions).

        ``enable_thinking=False`` disables Qwen3's hidden ``<think>`` block so the
        returned ``content`` is the direct answer (clean to parse). Returns the
        message ``content`` (falls back to ``reasoning_content`` if content is
        empty, which can happen when thinking is on).

        ``sampling`` may carry extra decoding params (``top_p``, ``top_k``,
        ``min_p``, ``presence_penalty``, ``repetition_penalty``). vLLM accepts the
        non-OpenAI ones (top_k/min_p/repetition_penalty) directly in the body.
        If ``sampling`` sets ``temperature`` it overrides the positional arg.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if sampling:
            payload.update(sampling)
        r = requests.post(
            f"{self.api_base}/chat/completions", json=payload, headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()

    # -- scoring --------------------------------------------------------------
    def _echo_token_logprobs(self, token_ids: List[int]) -> List[Optional[float]]:
        """Teacher-force ``token_ids`` and return per-token log-probs.

        Uses ``echo=True`` so the returned ``token_logprobs`` cover the prompt
        tokens (the first entry is ``None`` — no context for the first token).
        ``max_tokens=1`` keeps us compatible with vLLM builds that reject
        ``max_tokens=0``; the single generated token is sliced off by the
        caller via absolute indexing.
        """
        payload = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": 1,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
        }
        r = requests.post(
            f"{self.api_base}/completions", json=payload, headers=self.headers, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["logprobs"]["token_logprobs"]

    def action_nll(
        self,
        context_messages: List[Dict[str, str]],
        action_text: str,
        action_token_ids: Optional[List[int]] = None,
    ) -> Dict:
        """Mean per-token negative log-likelihood of ``action_text`` given context.

        The context is rendered with the chat template up to the assistant
        generation point, the action tokens are appended, and only the action
        tokens are scored. ``action_token_ids`` can be passed to guarantee the
        identical action tokenization across the full and compressed conditions.

        Returns a dict with ``nll`` (mean per-token), ``ppl`` and ``n_tokens``.
        """
        prompt_ids = self.tokenize(messages=context_messages, add_generation_prompt=True)
        if action_token_ids is None:
            action_token_ids = self.tokenize(text=action_text, add_special_tokens=False)
        if not action_token_ids:
            raise ValueError("action has no tokens to score")

        full_ids = prompt_ids + action_token_ids
        token_logprobs = self._echo_token_logprobs(full_ids)

        # The action tokens are the last len(action_token_ids) tokens of the
        # prompt portion (absolute indices, so a trailing generated token from
        # max_tokens=1 is excluded).
        n = len(action_token_ids)
        action_lps = token_logprobs[len(full_ids) - n : len(full_ids)]
        action_lps = [lp for lp in action_lps if lp is not None]
        if not action_lps:
            raise RuntimeError("server returned no log-probs for action tokens")

        nll = -sum(action_lps) / len(action_lps)
        return {"nll": nll, "ppl": math.exp(nll), "n_tokens": len(action_lps)}
