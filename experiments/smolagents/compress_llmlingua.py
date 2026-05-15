"""Run LLMLingua compression over every segment in the eval dataset.

Runs locally on a GPU node (e.g. r4516u05n01). Loads the default LLMLingua
small model and compresses each segment's history at a fixed rate (0.2 like
the existing ACON llmlingua_history.yaml).

Output: compressors_generations/llmlingua/compressions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken
from tqdm import tqdm

from llmlingua import PromptCompressor


_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


def messages_to_text(content):
    parts = []
    for step in content:
        parts.append(f"USER:\n{step['user']}\n\nASSISTANT:\n{step['assistant']}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="compressor_eval_data/dataset.jsonl")
    ap.add_argument("--out", default="compressors_generations/llmlingua/compressions.jsonl")
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument(
        "--model",
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        help="LLMLingua-2 model is fast; original llmlingua needs a small LM.",
    )
    ap.add_argument("--use_llmlingua2", action="store_true", default=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            try:
                d = json.loads(ln)
                done.add((d["sample_id"], d["segment_start"], d["segment_end"]))
            except Exception:
                pass

    rows = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l]
    todo = [
        r for r in rows
        if (r["sample_id"], r["segment"]["start_step"], r["segment"]["end_step"]) not in done
    ]
    print(f"loaded {len(rows)} segments; {len(todo)} to compress; {len(done)} cached")

    compressor = PromptCompressor(
        model_name=args.model,
        use_llmlingua2=args.use_llmlingua2,
    )

    with out_path.open("a") as f:
        for r in tqdm(todo, desc="llmlingua"):
            history_text = messages_to_text(r["segment"]["content"])
            try:
                result = compressor.compress_prompt(history_text, rate=args.rate)
                compressed = result.get("compressed_prompt", "")
            except Exception as e:
                compressed = f"__ERROR__: {e}"
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "segment_start": r["segment"]["start_step"],
                "segment_end": r["segment"]["end_step"],
                "compressor": "llmlingua",
                "input_tokens": r["segment"]["token_count"],
                "output_tokens": count_tokens(compressed),
                "compressed": compressed,
            }, ensure_ascii=False) + "\n")
            f.flush()
    print("done:", out_path)


if __name__ == "__main__":
    main()
