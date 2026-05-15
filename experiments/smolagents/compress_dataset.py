"""Apply every compression baseline to the segments in compressor_eval_data/dataset.jsonl.

Outputs one JSONL per baseline at
    compressor_eval_data/../compressors_generations/<baseline>/compressions.jsonl

Each output row mirrors the input segment id and stores the compressed text:
    {"sample_id", "segment_start", "segment_end", "compressor", "input_tokens",
     "output_tokens", "compressed"}

Baselines:
    - Selection (no LLM): fifo, mask_obs, mask_action, random  (token budget = 2048)
    - LLM prompting    : qwen3.5-35b-a3b, qwen3.5-27b, qwen3.5-9b, qwen3.5-4b,
                          gpt-4.1-mini  (word budget ~ 1024 = budget/2)
    - Discard          : keep_last_k=5 (no LLM, no token budget)

LLMLingua is in a sibling script (uses a different env / GPU node).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List

import tiktoken
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers shared by every baseline
# ---------------------------------------------------------------------------

def load_env(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        out[k.strip()] = v.strip()
    return out


_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


def segment_to_messages(content: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten the (step, user, assistant) list into role/content message turns."""
    msgs: List[Dict[str, str]] = []
    for step in content:
        msgs.append({"role": "user", "content": step["user"]})
        msgs.append({"role": "assistant", "content": step["assistant"]})
    return msgs


def messages_to_text(msgs: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in msgs:
        parts.append(f"{m['role'].upper()}:\n{m['content']}")
    return "\n\n".join(parts)


def task_text(content: List[Dict[str, Any]], sample_id: str, gens_dir: Path) -> str:
    """The user task is the first user message of step 1. If the segment doesn't
    start at step 1, re-read it from the original llm_history.json."""
    if content and content[0]["step"] == 1:
        return content[0]["user"]
    hist_path = gens_dir / sample_id / "llm_history.json"
    if hist_path.exists():
        conv = json.loads(hist_path.read_text())[0]
        if len(conv) > 1 and conv[1]["role"] == "user":
            return conv[1]["content"]
    return ""


# ---------------------------------------------------------------------------
# selection baselines (vendored from ctxopt/selection_strategies.py; tweaked
# to operate on (user, assistant) step pairs already produced by the dataset
# build, where each step is one user + one assistant turn).
# ---------------------------------------------------------------------------

MASK_OBS = "[OBSERVATION MASKED]"
MASK_ACT = "[ACTION MASKED]"


def _pair_cost(pair: Dict[str, Any]) -> int:
    return count_tokens(pair["user"]) + count_tokens(pair["assistant"])


def fifo_select(content: List[Dict[str, Any]], budget: int, **_) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    used = 0
    for p in reversed(content):
        c = _pair_cost(p)
        if used + c > budget:
            break
        kept.append(p)
        used += c
    kept.reverse()
    return kept


def mask_obs_select(content: List[Dict[str, Any]], budget: int, **_) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    used = 0
    for p in reversed(content):
        masked = {"step": p["step"], "user": MASK_OBS, "assistant": p["assistant"]}
        c = _pair_cost(masked)
        if used + c > budget:
            break
        kept.append(masked)
        used += c
    kept.reverse()
    return kept


def mask_action_select(content: List[Dict[str, Any]], budget: int, **_) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    used = 0
    for p in reversed(content):
        masked = {"step": p["step"], "user": p["user"], "assistant": MASK_ACT}
        c = _pair_cost(masked)
        if used + c > budget:
            break
        kept.append(masked)
        used += c
    kept.reverse()
    return kept


def random_select(content: List[Dict[str, Any]], budget: int, seed: int = 0) -> List[Dict[str, Any]]:
    import random as _r
    indexed = list(enumerate(content))
    rng = _r.Random(seed)
    rng.shuffle(indexed)
    used = 0
    chosen: List[int] = []
    for idx, p in indexed:
        c = _pair_cost(p)
        if used + c > budget:
            continue
        chosen.append(idx)
        used += c
    chosen.sort()
    return [content[i] for i in chosen]


def discard_last_k(content: List[Dict[str, Any]], k: int, **_) -> List[Dict[str, Any]]:
    return content[-k:] if k > 0 else content


SELECTION_BASELINES: Dict[str, Callable] = {
    "fifo_b2048":        lambda c: fifo_select(c, budget=2048),
    "mask_obs_b2048":    lambda c: mask_obs_select(c, budget=2048),
    "mask_action_b2048": lambda c: mask_action_select(c, budget=2048),
    "random_b2048":      lambda c: random_select(c, budget=2048, seed=0),
    "discard_keep5":     lambda c: discard_last_k(c, k=5),
}


# ---------------------------------------------------------------------------
# LLM prompting baselines
# ---------------------------------------------------------------------------

# (baseline_name, model_id, base_url-or-None, kind)   kind in {"openai", "vllm"}
LLM_BASELINES = [
    ("gpt-4.1-mini",     "gpt-4.1-mini",         None,                "openai"),
    ("qwen3.5-35b-a3b",  "Qwen/Qwen3.5-35B-A3B", "http://r818u33n04:8000/v1",  "vllm"),
    ("qwen3.5-27b",      "Qwen/Qwen3.5-27B",     "http://r4519u04n01:8000/v1", "vllm"),
    ("qwen3.5-9b",       "Qwen/Qwen3.5-9B",      "http://r4519u04n01:8001/v1", "vllm"),
    ("qwen3.5-4b",       "Qwen/Qwen3.5-4B",      "http://r817u23n05:8000/v1",  "vllm"),
]


def render_prompt(template_dir: Path, task: str, history_text: str, word_budget: int) -> tuple[str, str]:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    system = env.get_template("system_prompt.jinja").render()
    user = env.get_template("prompt_history_v2.jinja").render(
        task=task,
        prev_summary="",
        history=history_text,
        word_budget=word_budget,
        budget=word_budget * 2,
    )
    return system, user


def call_llm(client, model: str, system: str, user: str, max_tokens: int = 4096, retries: int = 3, disable_thinking: bool = False) -> str:
    last_err = None
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    if disable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return f"__ERROR__: {last_err}"


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def run_selection_baselines(rows: List[Dict[str, Any]], out_root: Path):
    for name, fn in SELECTION_BASELINES.items():
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "compressions.jsonl"
        with out_path.open("w") as f:
            for r in tqdm(rows, desc=f"[selection] {name}"):
                kept = fn(r["segment"]["content"])
                compressed_msgs = segment_to_messages(kept)
                compressed_text = messages_to_text(compressed_msgs)
                f.write(json.dumps({
                    "sample_id": r["sample_id"],
                    "segment_start": r["segment"]["start_step"],
                    "segment_end": r["segment"]["end_step"],
                    "compressor": name,
                    "input_tokens": r["segment"]["token_count"],
                    "output_tokens": count_tokens(compressed_text),
                    "compressed": compressed_text,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")


def run_llm_baseline(
    name: str, model: str, base_url: str | None, kind: str,
    rows: List[Dict[str, Any]], out_root: Path,
    api_key: str, gens_dir: Path, prompt_dir: Path,
    num_workers: int, word_budget: int, resume: bool,
):
    from openai import OpenAI
    if kind == "vllm":
        client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=300.0)
    else:
        client = OpenAI(api_key=api_key, timeout=300.0)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compressions.jsonl"

    # Resume: skip rows already written, indexed by (sample_id, start, end)
    done: set = set()
    if resume and out_path.exists():
        for ln in out_path.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add((d["sample_id"], d["segment_start"], d["segment_end"]))
            except Exception:
                continue
    todo = [
        r for r in rows
        if (r["sample_id"], r["segment"]["start_step"], r["segment"]["end_step"]) not in done
    ]
    print(f"[{name}] {len(todo)}/{len(rows)} to compress (resume: {len(done)} cached)")

    def work(r):
        task = task_text(r["segment"]["content"], r["sample_id"], gens_dir)
        msgs = segment_to_messages(r["segment"]["content"])
        history_text = messages_to_text(msgs)
        system, user = render_prompt(prompt_dir, task, history_text, word_budget)
        text = call_llm(client, model, system, user, disable_thinking=(kind == "vllm"))
        return r, text

    with out_path.open("a") as f_out, ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"[llm] {name}"):
            r, text = fut.result()
            f_out.write(json.dumps({
                "sample_id": r["sample_id"],
                "segment_start": r["segment"]["start_step"],
                "segment_end": r["segment"]["end_step"],
                "compressor": name,
                "input_tokens": r["segment"]["token_count"],
                "output_tokens": count_tokens(text),
                "compressed": text,
            }, ensure_ascii=False) + "\n")
            f_out.flush()
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="compressor_eval_data/dataset.jsonl")
    ap.add_argument("--gens_dir",
                    default="outputs/Qwen-Qwen3.5-35B-A3B_qwen3p5_35b_a3b_causalgen/test/samples")
    ap.add_argument("--out_root", default="compressors_generations")
    ap.add_argument("--prompt_dir", default="prompts/context_opt")
    ap.add_argument("--env_file", default=".env")
    ap.add_argument("--word_budget", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to listed baseline names")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="Baselines to skip")
    ap.add_argument("--no_resume", action="store_true",
                    help="Disable resume (overwrite existing LLM output files)")
    args = ap.parse_args()

    env = load_env(Path(args.env_file))
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(f"OPENAI_API_KEY missing in {args.env_file}")

    rows = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l]
    print(f"loaded {len(rows)} segments from {args.dataset}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    want = set(args.only) if args.only else None
    skip = set(args.skip) if args.skip else set()

    # --- selection baselines (cheap, single-threaded)
    if not args.only or any(n in want for n in SELECTION_BASELINES):
        sel_rows = rows
        for name in list(SELECTION_BASELINES):
            if (want and name not in want) or name in skip:
                continue
            out_dir = out_root / name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "compressions.jsonl"
            if out_path.exists() and not args.no_resume:
                print(f"[selection] {name}: exists, skipping (use --no_resume to redo)")
                continue
            with out_path.open("w") as f:
                for r in tqdm(sel_rows, desc=f"[selection] {name}"):
                    kept = SELECTION_BASELINES[name](r["segment"]["content"])
                    compressed_msgs = segment_to_messages(kept)
                    compressed_text = messages_to_text(compressed_msgs)
                    f.write(json.dumps({
                        "sample_id": r["sample_id"],
                        "segment_start": r["segment"]["start_step"],
                        "segment_end": r["segment"]["end_step"],
                        "compressor": name,
                        "input_tokens": r["segment"]["token_count"],
                        "output_tokens": count_tokens(compressed_text),
                        "compressed": compressed_text,
                    }, ensure_ascii=False) + "\n")
            print(f"wrote {out_path}")

    # --- LLM baselines
    for name, model, base_url, kind in LLM_BASELINES:
        if (want and name not in want) or name in skip:
            continue
        run_llm_baseline(
            name, model, base_url, kind, rows, out_root,
            api_key=api_key, gens_dir=Path(args.gens_dir),
            prompt_dir=Path(args.prompt_dir),
            num_workers=args.num_workers, word_budget=args.word_budget,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
