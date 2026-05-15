"""Build a compressor-evaluation dataset from causal-generation rollouts.

For each rollout we sweep a fixed-length sliding window over the cleaned
trajectory and, at each window position, ask: "of the causal recalls whose
prior is inside this window, which ones reference information needed AFTER
the window ends?" Those are the questions that probe whether a compressor
retained the right facts.

Segmentation method (`fixed_moving_window`):
  - Window holds as many consecutive steps as fit under `window_tokens`
    (default 8192).
  - To advance, drop steps from the left until at least `shift_tokens`
    (default 4096) worth of tokens have been removed, then extend right
    again under the cap. Repeat until the trajectory ends.

Each "step" bundles the user turn and the assistant turn for that step
(conv[2k-1] + conv[2k] in llm_history). Text is cleaned of the causal-gen
instrumentation (`[Step N — your action]`, `[Observation from step N]`,
`Recalls: ...` lines, `<recall>` tags) so what the compressor sees matches
what a non-causal agent would have produced.

Each output row: {sample_id, segment: {start, end, method, token_count,
content (list of {step, role, content})}, questions: [...]}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken
from tqdm import tqdm


_RECALL_TAG_RE = re.compile(
    r"<recall\s+step\s*=\s*['\"]?(\d+)['\"]?\s*>(.*?)</recall>",
    re.DOTALL | re.IGNORECASE,
)
_RECALLS_HEADER_RE = re.compile(
    r"^[ \t]*Recalls:[ \t]*(\(none\))?[ \t]*(?:\n|$)",
    re.MULTILINE | re.IGNORECASE,
)
_STEP_HEADER_RE = re.compile(
    r"^\s*\[Step\s+\d+\s+—\s+your action\]\s*\n?", re.MULTILINE
)
_OBS_HEADER_RE = re.compile(
    r"^\s*\[Observation from step\s+\d+\]\s*\n?", re.MULTILINE
)


def clean_causal_artifacts(text: str) -> str:
    """Strip causal-generation instrumentation so the segment matches what a
    vanilla agent would see."""
    if not text:
        return text
    text = _STEP_HEADER_RE.sub("", text)
    text = _OBS_HEADER_RE.sub("", text)
    # Remove the inline <recall> tags first, then the Recalls: header line(s).
    text = _RECALL_TAG_RE.sub("", text)
    # Catch unbalanced/malformed recall tags the model occasionally emits.
    text = re.sub(r"<recall\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</recall\s*>", "", text, flags=re.IGNORECASE)
    text = _RECALLS_HEADER_RE.sub("", text)
    return text.lstrip("\n")


def load_env(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def extract_steps(conv: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Pair each user turn with the assistant turn that follows it.

    conv layout (causal-gen runs):
        0: system, 1: user(task), 2: assistant step1, 3: user obs step1, ...
    Step k therefore lives at conv[2k-1] (user) + conv[2k] (assistant).
    """
    steps: List[Dict[str, Any]] = []
    k = 1
    while True:
        u_idx = 2 * k - 1
        a_idx = 2 * k
        if a_idx >= len(conv):
            break
        if conv[u_idx]["role"] != "user" or conv[a_idx]["role"] != "assistant":
            break
        user_clean = clean_causal_artifacts(conv[u_idx]["content"])
        asst_clean = clean_causal_artifacts(conv[a_idx]["content"])
        steps.append({"step": k, "user": user_clean, "assistant": asst_clean})
        k += 1
    return steps


def annotate_token_counts(steps: List[Dict[str, Any]], enc) -> None:
    for s in steps:
        s["user_tokens"] = len(enc.encode(s["user"]))
        s["asst_tokens"] = len(enc.encode(s["assistant"]))
        s["tokens"] = s["user_tokens"] + s["asst_tokens"]


def fixed_moving_window(
    step_tokens: List[int], window_tokens: int, shift_tokens: int
) -> List[Tuple[int, int]]:
    """Return list of (start_step, end_step) 1-indexed inclusive segments."""
    N = len(step_tokens)
    if N == 0:
        return []
    segments: List[Tuple[int, int]] = []
    start = 1  # 1-indexed
    while start <= N:
        end = start - 1
        total = 0
        # extend right under the cap
        while end + 1 <= N:
            t = step_tokens[end]  # step_tokens is 0-indexed, end+1 step -> index `end`
            if end >= start and total + t > window_tokens:
                break
            end += 1
            total += t
            if end >= N:
                break
        if end < start:
            # the single step at `start` exceeds the window; include it alone
            end = start
            total = step_tokens[start - 1]
        segments.append((start, end))
        if end >= N:
            break
        # advance start by >= shift_tokens of removed step tokens
        removed = 0
        new_start = start
        while new_start <= end and removed < shift_tokens:
            removed += step_tokens[new_start - 1]
            new_start += 1
        if new_start == start:
            new_start = start + 1
        if new_start > N:
            break
        start = new_start
    return segments


QUESTION_SYS = (
    "You convert a short factual statement into a single natural-language question "
    "whose correct answer is exactly that fact. The question must be self-contained "
    "(no pronouns like 'this' or 'it' referring to context the reader cannot see) and "
    "answerable using only the information in the statement. Output the question only, "
    "no quotes, no preamble."
)


def info_to_question(client, model: str, info: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUESTION_SYS},
            {"role": "user", "content": f"Statement: {info}\n\nQuestion:"},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generations_dir",
        default="outputs/Qwen-Qwen3.5-35B-A3B_qwen3p5_35b_a3b_causalgen/test/samples",
    )
    ap.add_argument(
        "--output_path", default="compressor_eval_data/dataset.jsonl"
    )
    ap.add_argument(
        "--questions_cache",
        default="compressor_eval_data/questions.json",
        help="info->question cache so re-runs cost nothing for old infos.",
    )
    ap.add_argument("--env_file", default=".env")
    ap.add_argument("--model", default=None, help="Override QUESTION_EXTRACTION_MODEL")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--window_tokens", type=int, default=8192)
    ap.add_argument("--shift_tokens", type=int, default=4096)
    ap.add_argument(
        "--tokenizer", default="cl100k_base", help="tiktoken encoding name"
    )
    ap.add_argument("--limit_samples", type=int, default=None)
    args = ap.parse_args()

    env = load_env(Path(args.env_file))
    api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(f"OPENAI_API_KEY not found in {args.env_file} or environment")
    model = args.model or env.get("QUESTION_EXTRACTION_MODEL") or "gpt-4.1-mini"

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    enc = tiktoken.get_encoding(args.tokenizer)

    gen_dir = Path(args.generations_dir)
    sample_dirs = sorted(p for p in gen_dir.iterdir() if p.is_dir())
    if args.limit_samples:
        sample_dirs = sample_dirs[: args.limit_samples]

    # ----- pass 1: load each sample, build steps + segments, collect distinct infos
    per_sample: List[Dict[str, Any]] = []
    all_infos: set = set()

    for sd in sample_dirs:
        cp_path = sd / "causal_pairs.json"
        hist_path = sd / "llm_history.json"
        if not cp_path.exists() or not hist_path.exists():
            continue
        pairs = json.loads(cp_path.read_text())
        conv_list = json.loads(hist_path.read_text())
        if not conv_list:
            continue
        conv = conv_list[0]
        steps = extract_steps(conv)
        if not steps:
            continue
        annotate_token_counts(steps, enc)
        step_tokens = [s["tokens"] for s in steps]
        segments = fixed_moving_window(step_tokens, args.window_tokens, args.shift_tokens)
        per_sample.append({
            "sample_dir": sd,
            "sample_id": sd.name,
            "pairs": pairs,
            "steps": steps,
            "segments": segments,
        })
        for _, _post, _prior, info in pairs:
            all_infos.add(info.strip())

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qcache_path = Path(args.questions_cache)
    qcache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, str] = {}
    if qcache_path.exists():
        cache = json.loads(qcache_path.read_text())
    missing = [s for s in all_infos if s not in cache]
    print(
        f"Samples processed: {len(per_sample)} | "
        f"Distinct info strings: {len(all_infos)} | Missing in cache: {len(missing)}"
    )

    if missing:
        def worker(info: str) -> Tuple[str, str]:
            try:
                return info, info_to_question(client, model, info)
            except Exception as e:
                return info, f"__ERROR__: {e}"

        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = [ex.submit(worker, info) for info in missing]
            for f in tqdm(as_completed(futs), total=len(futs), desc="info->question"):
                info, q = f.result()
                cache[info] = q
                if len(cache) % 100 == 0:
                    qcache_path.write_text(
                        json.dumps(cache, ensure_ascii=False, indent=2)
                    )
        qcache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    # ----- pass 2: emit one row per segment; track "uncounted" pairs
    n_examples = 0
    n_questions = 0
    n_pairs_total = 0
    n_pairs_uncounted = 0  # pair never crosses any segment boundary
    n_pairs_no_segment = 0  # prior step not inside any segment at all
    sum_window_tokens = 0

    with out_path.open("w") as f_out:
        for s in per_sample:
            steps = s["steps"]
            segments = s["segments"]
            pairs = s["pairs"]
            sample_id = s["sample_id"]

            # For each pair, find segments that contain its prior step.
            # A pair becomes a question for segment seg iff prior in seg AND posterior > seg_end.
            # A pair is "uncounted" if every segment containing its prior also contains its posterior.
            pair_status = []  # parallel to pairs: "crossed" / "inside" / "out_of_range"
            for _sid, post, prior, _info in pairs:
                segs_with_prior = [
                    (a, b) for (a, b) in segments if a <= prior <= b
                ]
                if not segs_with_prior:
                    pair_status.append("out_of_range")
                    continue
                if any(post > b for (_a, b) in segs_with_prior):
                    pair_status.append("crossed")
                else:
                    pair_status.append("inside")
            n_pairs_total += len(pairs)
            n_pairs_uncounted += sum(1 for st in pair_status if st == "inside")
            n_pairs_no_segment += sum(1 for st in pair_status if st == "out_of_range")

            for (seg_start, seg_end) in segments:
                # collect questions: pairs whose prior is in this segment and posterior past seg_end
                crossing = []
                seen = set()
                for (p, status) in zip(pairs, pair_status):
                    _sid, post, prior, info = p
                    if not (seg_start <= prior <= seg_end):
                        continue
                    if post <= seg_end:
                        continue
                    key = (prior, info.strip())
                    if key in seen:
                        continue
                    seen.add(key)
                    crossing.append({
                        "prior_step": prior,
                        "posterior_step": post,
                        "info": info,
                        "question": cache.get(info.strip(), ""),
                    })

                # build segment content as a list of cleaned step dicts
                content = []
                tok = 0
                for st in steps[seg_start - 1: seg_end]:
                    content.append({
                        "step": st["step"],
                        "user": st["user"],
                        "assistant": st["assistant"],
                    })
                    tok += st["tokens"]
                sum_window_tokens += tok

                ex = {
                    "sample_id": sample_id,
                    "segment": {
                        "method": "fixed_moving_window",
                        "window_tokens": args.window_tokens,
                        "shift_tokens": args.shift_tokens,
                        "start_step": seg_start,
                        "end_step": seg_end,
                        "token_count": tok,
                        "content": content,
                    },
                    "questions": crossing,
                }
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_examples += 1
                n_questions += len(crossing)

    # ----- report
    avg_tok = sum_window_tokens / n_examples if n_examples else 0
    report = (
        f"Compressor-eval dataset report\n"
        f"==============================\n"
        f"generations_dir   : {args.generations_dir}\n"
        f"segmentation      : fixed_moving_window (window={args.window_tokens}, shift={args.shift_tokens})\n"
        f"tokenizer         : {args.tokenizer}\n"
        f"\n"
        f"samples processed : {len(per_sample)}\n"
        f"segments emitted  : {n_examples}\n"
        f"avg segment tokens: {avg_tok:.1f}\n"
        f"questions emitted : {n_questions}\n"
        f"\n"
        f"causal pairs total                 : {n_pairs_total}\n"
        f"  -> uncounted (prior+posterior in same segment): {n_pairs_uncounted}\n"
        f"  -> prior not contained in any segment         : {n_pairs_no_segment}\n"
        f"  -> contributed at least one question          : {n_pairs_total - n_pairs_uncounted - n_pairs_no_segment}\n"
    )
    print(report)
    report_path = out_path.with_suffix(".report.txt")
    report_path.write_text(report)
    print(f"Wrote dataset to {out_path}")
    print(f"Wrote report to  {report_path}")
    print(f"Question cache:  {qcache_path}")


if __name__ == "__main__":
    main()
