import os
import json
import argparse
from typing import List, Dict, Any, Optional
from data_utils import (
    load_task_results,
    process_history_file,
    get_train_passed_paths,
)

# -----------------------------
# Smolagents-specific helpers
# -----------------------------

def load_smolagents_predictions(predictions_path: str) -> Dict[str, Dict[str, Any]]:
    """Load per-sample metrics from smolagents predictions.jsonl.

    Returns mapping sample_id -> {em, f1, success, _raw}.
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    if not predictions_path or not os.path.exists(predictions_path):
        print(f"[warn] predictions file not found: {predictions_path}")
        return metrics
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _id = obj.get("id")
            if not _id:
                continue
            metrics[_id] = {
                "em": obj.get("em"),
                "f1": obj.get("f1"),
                "success": obj.get("success"),
                "_raw": obj,
            }
    print(f"Loaded metrics for {len(metrics)} samples")
    return metrics


def smolagents_sample_passes(m: Optional[Dict[str, Any]], min_f1: Optional[float], min_em: Optional[float], require_success: bool) -> bool:
    if m is None:
        return False
    if require_success and not m.get("success"):
        return False
    if min_f1 is not None and (m.get("f1") is None or m.get("f1") < min_f1):
        return False
    if min_em is not None and (m.get("em") is None or m.get("em") < min_em):
        return False
    return True


def smolagents_gather_history_paths(train_dir: str, file_type: str) -> List[str]:
    """Return list of history file paths under train/samples/<id>/<file_type>.json.

    train_dir should point to the split directory (e.g., .../train)
    """
    samples_dir = os.path.join(train_dir, "samples")
    if not os.path.isdir(samples_dir):
        return []
    paths: List[str] = []
    for entry in os.listdir(samples_dir):
        candidate = os.path.join(samples_dir, entry, f"{file_type}.json")
        if os.path.isfile(candidate):
            paths.append(candidate)
    return sorted(paths)


def save_trajectories_to_jsonl(
    task: str,
    base_dir: str,
    file_types: List[str],
    output_path: str,
    result_path: Optional[str] = None,
    replace_jinja_path: Optional[str] = None,
    # smolagents-specific
    predictions_path: Optional[str] = None,
    min_f1: Optional[float] = None,
    min_em: Optional[float] = None,
    require_success: bool = True,
):
    """Aggregate trajectories into a JSONL dataset (supports appworld / officebench / smolagents)."""
    all_conversations: List[Dict[str, Any]] = []

    if task == "smolagents":
        # Process smolagents format: base_dir is the split dir (e.g., .../train)
        metrics_map = load_smolagents_predictions(predictions_path or "")
        for file_type in file_types:
            print(f"\n[smolagents] Processing {file_type} files...")
            history_paths = smolagents_gather_history_paths(base_dir, file_type)
            print(f"Found {len(history_paths)} history files under samples/")
            kept = 0
            for hp in history_paths:
                sample_id = os.path.basename(os.path.dirname(hp))
                m = metrics_map.get(sample_id)
                if not smolagents_sample_passes(m, min_f1, min_em, require_success):
                    continue
                conversations = process_history_file(hp, file_type, replace_jinja_path)
                for c in conversations:
                    c.setdefault("metadata", {}).update({
                        "file_type": file_type,
                        "sample_id": sample_id,
                        "em": m.get("em") if m else None,
                        "f1": m.get("f1") if m else None,
                        "success": m.get("success") if m else None,
                        "source_file": hp,
                    })
                all_conversations.extend(conversations)
                kept += 1
            print(f"  kept {kept}/{len(history_paths)} samples for {file_type}")
    else:
        # appworld / officebench
        task_id_to_result = load_task_results(result_path, task) if (result_path and os.path.exists(result_path)) else {}
        for file_type in file_types:
            print(f"\nProcessing {file_type} files...")
            train_passed_paths = get_train_passed_paths(task, base_dir, file_type, task_id_to_result)
            print(f"Found {len(train_passed_paths)} passing files")
            for file_path in train_passed_paths:
                conversations = process_history_file(file_path, file_type, replace_jinja_path)
                print(f"  {file_path}: {len(conversations)} conversations")
                all_conversations.extend(conversations)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in all_conversations:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")

    print(f"\nSaved {len(all_conversations)} conversations -> {output_path}")
    counts: Dict[str, int] = {}
    for conv in all_conversations:
        ft = conv.get("metadata", {}).get("file_type", "unknown")
        counts[ft] = counts.get(ft, 0) + 1
    print("Distribution:", counts)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Save trajectories (appworld / officebench / smolagents)")
    p.add_argument("--task", choices=["appworld", "officebench", "smolagents"], required=True)
    p.add_argument("--folders", required=True, help="Comma separated experiment folder names under outputs/")
    p.add_argument("--file-types", default="history_optimizer_history,obs_optimizer_history,llm_history")
    p.add_argument("--outputs-root", default="dataset", help="Root output directory; files saved under <root>/<task>/<file_type>/exp_split.jsonl")
    p.add_argument("--replace-jinja", default=None, help="Optional Jinja template to render user prompts when extra args present in history")
    p.add_argument("--split", default="train")
    # appworld/officebench
    p.add_argument("--results-filename", default=None, help="Override results filename (appworld/officebench)")
    # smolagents
    p.add_argument("--predictions-filename", default="predictions.jsonl", help="Filename of predictions.jsonl inside split dir (smolagents)")
    p.add_argument("--min-f1", type=float, default=None, help="Keep samples with f1 >= this (smolagents)")
    p.add_argument("--min-em", type=float, default=None, help="Keep samples with em >= this (smolagents)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--require-success", dest="require_success", action="store_true", help="Require success==True (smolagents)")
    group.add_argument("--no-require-success", dest="require_success", action="store_false", help="Do not require success (smolagents)")
    p.set_defaults(require_success=True)
    return p


def main():
    args = build_arg_parser().parse_args()
    task = args.task
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    file_types = [f.strip() for f in args.file_types.split(",") if f.strip()]
    split = args.split

    results_filename = args.results_filename
    if results_filename is None:
        results_filename = "train.json" if task == "appworld" else "train.jsonl"

    for folder in folders:
        if task == "appworld":
            base_dir = f"../{task}/outputs/{folder}"
            result_path = f"../{task}/experiments/outputs/{folder}/evaluations/{results_filename}"
            predictions_path = None
        elif task == "officebench":
            base_dir = f"../officebench/outputs/{folder}"
            result_path = f"../officebench/outputs/{folder}/{split}/{folder}_{split}_result_overall.json"
            predictions_path = None
        else:  # smolagents
            base_dir = f"../smolagents/outputs/{folder}"
            # predictions live under the split dir
            predictions_path = os.path.join(base_dir, split, args.predictions_filename)
            result_path = None
        experiment_name = os.path.basename(base_dir)
        for ft in file_types:
            out_path = f"{args.outputs_root}/{task}/{ft}/{experiment_name}_{split}.jsonl"
            print(f"Processing folder={folder} file_type={ft} split={split}")
            split_dir = os.path.join(base_dir, split)
            save_trajectories_to_jsonl(
                task,
                split_dir,
                [ft],
                out_path,
                result_path=result_path,
                replace_jinja_path=args.replace_jinja,
                predictions_path=predictions_path,
                min_f1=args.min_f1,
                min_em=args.min_em,
                require_success=args.require_success,
            )
            print("-" * 60)


if __name__ == "__main__":
    main()
