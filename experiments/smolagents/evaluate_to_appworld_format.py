#!/usr/bin/env python3
"""
Convert Smolagents predictions.jsonl into an AppWorld-style evaluation report.

Input:
- predictions.jsonl files written by experiments/smolagents/run.py

Output:
- evaluations/dev.json next to predictions.jsonl, with structure:
  {
    "aggregate": {
      "task_goal_completion": <avg EM percent>,
      "scenario_goal_completion": <avg F1 percent>
    },
    "individual": {
      <id>: {
        "success": <bool>,          # EM==1.0
        "difficulty": 1,            # Unknown for MuSiQue; default to 1
        "num_tests": <int>,         # Number of sub-questions (len(answer)) if available else 1
        "passes": [ {"requirement": "assert answers match.", "label": "no_op_pass"} ]
        "failures": [ {"requirement": "assert answers match.", "trace": "pred vs gold", "label": "no_op_fail"} ]
      },
      ...
    }
  }

Usage examples:
  python evaluate_to_appworld_format.py \
    --file experiments/smolagents/outputs/<run>/<fold>/predictions.jsonl

  python evaluate_to_appworld_format.py --outputs-root experiments/smolagents/outputs --all
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}: {e}\nLine: {line[:200]}")
    return rows


def _to_eval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"aggregate": {"exact_match": 0.0, "f1": 0.0}, "individual": {}}

    # Aggregate
    ems = [float(r.get("em", 0.0)) for r in rows]
    f1s = [float(r.get("f1", 0.0)) for r in rows]
    avg_em = sum(ems) / len(ems) if ems else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    aggregate = {
        # Map: task_goal_completion = EM%, scenario_goal_completion = F1%
        "exact_match": round(avg_em * 100.0, 3),
        "f1": round(avg_f1 * 100.0, 3),
    }

    # Individuals
    individual: Dict[str, Any] = {}
    for r in rows:
        rid = str(r.get("id", ""))
        # answer is typically a list of lists (acceptable answers per sub-question)
        ans = r.get("answer")
        num_tests = 1
        if isinstance(ans, list):
            try:
                num_tests = len(ans)
            except Exception:
                num_tests = 1
        em = float(r.get("em", 0.0))
        pred = r.get("prediction")

        success = bool(em >= 0.9999)

        # Compose simple pass/fail evidence centered around answer match requirement
        passes: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []
        if success:
            passes.append({"requirement": "assert answers match.", "label": "no_op_pass"})
        else:
            # Trace is a compact string, avoid dumping huge content
            trace_lines: List[str] = []
            try:
                # Only include up to first 3 predicted and gold answers to keep concise
                pred_preview = pred if isinstance(pred, list) else [str(pred)]
                pred_preview = [str(x) for x in pred_preview][:3]
                gold_preview: List[Any]
                if isinstance(ans, list):
                    # If answers are list-of-list, take the first candidate of each sub-q for preview
                    gold_preview = [(a[0] if isinstance(a, list) and a else a) for a in ans][:3]
                else:
                    gold_preview = [ans]
                trace_lines.append(f"pred={pred_preview}")
                trace_lines.append(f"gold={gold_preview}")
            except Exception:
                trace_lines.append("unavailable")
            failures.append({
                "requirement": "assert answers match.",
                "trace": " | ".join(trace_lines),
                "label": "no_op_fail",
            })

        individual[rid] = {
            "f1": float(r.get("f1", 0.0)),
            "difficulty": int(r.get("difficulty", 1) or 1),  # default to 1
            "num_tests": int(num_tests),
            "passes": passes,
            "failures": failures,
        }

    return {"aggregate": aggregate, "individual": individual}


def _write_eval(eval_obj: Dict[str, Any], out_dir: str) -> str:
    # Save alongside predictions.jsonl as evaluations.json (no subfolder)
    out_path = os.path.join(out_dir, "evaluations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_obj, f, indent=4, ensure_ascii=False)
    return out_path


def process_predictions_file(pred_path: str) -> str:
    rows = _read_jsonl(pred_path)
    eval_obj = _to_eval(rows)
    out_dir = os.path.dirname(pred_path)
    out_path = _write_eval(eval_obj, out_dir)
    return out_path


def find_all_predictions(outputs_root: str) -> List[str]:
    matches: List[str] = []
    for root, _dirs, files in os.walk(outputs_root):
        if "predictions.jsonl" in files:
            matches.append(os.path.join(root, "predictions.jsonl"))
    return matches


def main():
    parser = argparse.ArgumentParser(description="Build AppWorld-style evaluation JSON from Smolagents predictions.jsonl")
    parser.add_argument("--file", type=str, default=None, help="Path to a predictions.jsonl to convert")
    parser.add_argument("--outputs-root", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"), help="Root folder to scan for predictions.jsonl when --all is used")
    parser.add_argument("--all", action="store_true", help="Process all predictions.jsonl under --outputs-root")
    args = parser.parse_args()

    written: List[Tuple[str, str]] = []
    if args.file:
        out = process_predictions_file(args.file)
        written.append((args.file, out))
    elif args.all:
        preds = find_all_predictions(args.outputs_root)
        if not preds:
            print(f"No predictions.jsonl found under {args.outputs_root}")
            return
        for p in preds:
            out = process_predictions_file(p)
            written.append((p, out))
    else:
        parser.error("Provide --file or --all")

    for src, dst in written:
        print(f"Wrote evaluation for {src} -> {dst}")


if __name__ == "__main__":
    main()
