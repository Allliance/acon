"""Evaluate each compressor baseline by probing the agent with the questions
attached to every trajectory segment.

For each baseline directory under `compressors_generations/`:

  1) For every segment, look up the questions from the eval dataset.
  2) Ask the agent (Qwen3.5-35B-A3B on the configured vLLM endpoint) to answer
     each question given only the compressed context. Concurrency: 64.
  3) Judge each answer with gpt-4.1-mini against the gold `info` string.
  4) Write `answers.jsonl` (one row per question, judge verdict augmented in
     place) and `summary.json` (accuracy + stats).

Resumable: existing rows in answers.jsonl are kept if both the agent answer
and judge verdict are non-empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken
from tqdm import tqdm


AGENT_MODEL = "Qwen/Qwen3.5-35B-A3B"
AGENT_BASE_URL = "http://r818u33n04:8000/v1"
JUDGE_MODEL_DEFAULT = "gpt-4.1-mini"


_enc = tiktoken.get_encoding("cl100k_base")


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


def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM = (
    "You answer questions strictly from the provided compressed agent context. "
    "If the context does not contain the answer, reply exactly: I don't know. "
    "Otherwise, answer in one short sentence or phrase. Do not invent facts."
)


def agent_user_prompt(context: str, question: str) -> str:
    # Truncate extremely large contexts to keep input below model limit.
    if count_tokens(context) > 20000:
        toks = _enc.encode(context)
        context = _enc.decode(toks[:20000])
    return (
        "## Compressed Context\n"
        f"{context}\n\n"
        "## Question\n"
        f"{question}\n\n"
        "## Answer"
    )


JUDGE_SYSTEM = (
    "You are a strict evaluator. Decide whether the candidate answer correctly "
    "captures the gold fact for the given question. Be lenient about phrasing "
    "and minor surface differences (synonyms, partial names, equivalent dates) "
    "as long as the key fact is the same. Reply on a single line in the form: "
    "VERDICT | brief reason. VERDICT must be exactly 'correct' or 'incorrect'."
)


def judge_user_prompt(question: str, gold_info: str, candidate: str) -> str:
    return (
        f"Question: {question}\n"
        f"Gold fact: {gold_info}\n"
        f"Candidate answer: {candidate}\n\n"
        "Reply on one line: VERDICT | reason"
    )


# ---------------------------------------------------------------------------
# OpenAI clients
# ---------------------------------------------------------------------------

def make_clients(api_key: str):
    from openai import OpenAI
    agent_client = OpenAI(api_key="EMPTY", base_url=AGENT_BASE_URL, timeout=300.0)
    judge_client = OpenAI(api_key=api_key, timeout=120.0)
    return agent_client, judge_client


def call_chat(
    client, model: str, system: str, user: str,
    max_tokens: int, retries: int = 3, disable_thinking: bool = False,
) -> str:
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
    last = None
    for i in range(retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    return f"__ERROR__: {last}"


# ---------------------------------------------------------------------------
# dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(dataset_path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    out = {}
    for ln in dataset_path.read_text().splitlines():
        if not ln:
            continue
        r = json.loads(ln)
        key = (r["sample_id"], r["segment"]["start_step"], r["segment"]["end_step"])
        out[key] = r
    return out


def load_existing_answers(path: Path) -> Dict[Tuple[str, int, int, str], Dict[str, Any]]:
    """Index: (sample_id, start, end, question) -> answer row."""
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if not ln:
            continue
        d = json.loads(ln)
        out[(d["sample_id"], d["segment_start"], d["segment_end"], d["question"])] = d
    return out


# ---------------------------------------------------------------------------
# per-baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(
    baseline_dir: Path,
    dataset_index: Dict[Tuple[str, int, int], Dict[str, Any]],
    agent_client, judge_client, judge_model: str,
    agent_workers: int, judge_workers: int,
    skip_empty_questions: bool = True,
) -> Dict[str, Any]:
    name = baseline_dir.name
    comp_path = baseline_dir / "compressions.jsonl"
    ans_path = baseline_dir / "answers.jsonl"
    summary_path = baseline_dir / "summary.json"

    if not comp_path.exists():
        return {"baseline": name, "skipped": "no compressions.jsonl"}

    # Build the work list: (compression_row, question_dict) pairs
    compressions = [json.loads(ln) for ln in comp_path.read_text().splitlines() if ln]
    existing = load_existing_answers(ans_path)
    work: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for c in compressions:
        key = (c["sample_id"], c["segment_start"], c["segment_end"])
        ds_row = dataset_index.get(key)
        if not ds_row:
            continue
        for q in ds_row["questions"]:
            if skip_empty_questions and not q.get("question", "").strip():
                continue
            ans_key = (c["sample_id"], c["segment_start"], c["segment_end"], q["question"])
            prior = existing.get(ans_key)
            if prior and prior.get("agent_answer") and prior.get("judge_verdict"):
                continue
            work.append((c, q))

    print(f"[{name}] {len(work)} questions to evaluate ({len(existing)} cached)")
    if not work and not existing:
        return {"baseline": name, "skipped": "no questions"}

    # ----- agent pass
    def agent_work(item):
        comp, q = item
        prompt = agent_user_prompt(comp["compressed"], q["question"])
        ans = call_chat(
            agent_client, AGENT_MODEL, AGENT_SYSTEM, prompt,
            max_tokens=256, disable_thinking=True,
        )
        return comp, q, ans

    agent_results: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    if work:
        with ThreadPoolExecutor(max_workers=agent_workers) as ex:
            futs = [ex.submit(agent_work, w) for w in work]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"[{name}] agent"):
                agent_results.append(f.result())

    # ----- judge pass
    def judge_work(triple):
        comp, q, ans = triple
        prompt = judge_user_prompt(q["question"], q["info"], ans)
        verdict_raw = call_chat(
            judge_client, judge_model, JUDGE_SYSTEM, prompt,
            max_tokens=120, disable_thinking=False,
        )
        # Parse "VERDICT | reason"
        line = verdict_raw.split("\n", 1)[0].strip()
        verdict = "incorrect"
        reason = verdict_raw
        if "|" in line:
            head, tail = line.split("|", 1)
            v = head.strip().lower()
            if v in {"correct", "incorrect"}:
                verdict = v
                reason = tail.strip()
        else:
            low = line.lower()
            if low.startswith("correct"):
                verdict = "correct"
            elif low.startswith("incorrect"):
                verdict = "incorrect"
            reason = line
        return comp, q, ans, verdict, reason

    judged: List[Dict[str, Any]] = []
    if agent_results:
        with ThreadPoolExecutor(max_workers=judge_workers) as ex:
            futs = [ex.submit(judge_work, t) for t in agent_results]
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"[{name}] judge"):
                comp, q, ans, verdict, reason = f.result()
                judged.append({
                    "sample_id": comp["sample_id"],
                    "segment_start": comp["segment_start"],
                    "segment_end": comp["segment_end"],
                    "compressor": name,
                    "prior_step": q["prior_step"],
                    "posterior_step": q["posterior_step"],
                    "info": q["info"],
                    "question": q["question"],
                    "compressed_tokens": comp.get("output_tokens"),
                    "agent_answer": ans,
                    "judge_verdict": verdict,
                    "judge_reason": reason,
                })

    # ----- merge existing + new, write atomic
    merged: Dict[Tuple[str, int, int, str], Dict[str, Any]] = dict(existing)
    for row in judged:
        merged[(row["sample_id"], row["segment_start"], row["segment_end"], row["question"])] = row
    rows = list(merged.values())
    with ans_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- summary
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("judge_verdict") == "correct")
    n_idk = sum(
        1 for r in rows
        if (r.get("agent_answer") or "").strip().lower().startswith("i don't know")
    )
    by_compressed_tok_buckets: Dict[str, Dict[str, int]] = {}
    for r in rows:
        ot = r.get("compressed_tokens") or 0
        bucket = (
            "0-512" if ot <= 512 else
            "513-1024" if ot <= 1024 else
            "1025-2048" if ot <= 2048 else
            "2049-4096" if ot <= 4096 else
            "4097+"
        )
        b = by_compressed_tok_buckets.setdefault(bucket, {"n": 0, "correct": 0})
        b["n"] += 1
        if r.get("judge_verdict") == "correct":
            b["correct"] += 1

    summary = {
        "baseline": name,
        "judge_model": judge_model,
        "agent_model": AGENT_MODEL,
        "agent_endpoint": AGENT_BASE_URL,
        "num_questions": n,
        "num_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "num_idk": n_idk,
        "idk_rate": (n_idk / n) if n else 0.0,
        "accuracy_by_compressed_tokens": by_compressed_tok_buckets,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[{name}] accuracy = {summary['accuracy']:.3f}  ({n_correct}/{n}, idk={n_idk})")
    return summary


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="compressor_eval_data/dataset.jsonl")
    ap.add_argument("--root", default="compressors_generations")
    ap.add_argument("--env_file", default=".env")
    ap.add_argument("--only", nargs="*", default=None, help="baseline names to evaluate")
    ap.add_argument("--skip", nargs="*", default=None)
    ap.add_argument("--agent_workers", type=int, default=64)
    ap.add_argument("--judge_workers", type=int, default=32)
    ap.add_argument("--judge_model", default=None)
    args = ap.parse_args()

    env = load_env(Path(args.env_file))
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(f"OPENAI_API_KEY not found in {args.env_file}")
    judge_model = args.judge_model or env.get("QUESTION_EXTRACTION_MODEL") or JUDGE_MODEL_DEFAULT

    agent_client, judge_client = make_clients(api_key)
    dataset_index = load_dataset(Path(args.dataset))
    print(f"dataset: {len(dataset_index)} segments")

    root = Path(args.root)
    baselines = sorted([p for p in root.iterdir() if p.is_dir()])
    want = set(args.only) if args.only else None
    skip = set(args.skip or [])

    overall: List[Dict[str, Any]] = []
    for bdir in baselines:
        if (want and bdir.name not in want) or bdir.name in skip:
            continue
        s = evaluate_baseline(
            bdir, dataset_index, agent_client, judge_client, judge_model,
            args.agent_workers, args.judge_workers,
        )
        overall.append(s)

    print("\n=== Accuracy summary ===")
    for s in sorted(overall, key=lambda x: -x.get("accuracy", 0.0)):
        if "accuracy" not in s:
            print(f"  {s['baseline']}: SKIPPED ({s.get('skipped')})")
            continue
        print(f"  {s['baseline']:24s} acc={s['accuracy']:.3f}  "
              f"({s['num_correct']}/{s['num_questions']}, idk={s['num_idk']})")

    (root / "_overall_summary.json").write_text(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
