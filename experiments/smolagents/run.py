#!/usr/bin/env python3
"""
Minimal Smolagents + MuSiQue runner

- Loads minimal MuSiQue samples
- Builds SmolagentsEnv and SmolagentsAgent
- Evaluates with exact match on final answer
"""

import json
import os
import yaml
from dataclasses import asdict
from types import SimpleNamespace
from datetime import datetime
from typing import Any, Dict, Optional
from eval_utils import exact_match, f1_max

from productive_agents.env.smolagents.env import SmolagentsEnv
from productive_agents.env.smolagents.config import SmolagentsEnvConfig
from productive_agents.agents.smolagents.agent import create_smolagents_agent
from productive_agents.agents.unified_agent import merge_configs
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

try:
    # When running as a module: experiments.smolagents.run
    from .dataset import QALoader, QAExample
except Exception:
    # When running as a script from this folder: python run.py
    from dataset import QALoader, QAExample


def _sanitize_for_path(name: str) -> str:
    # Keep alnum, dash, underscore, dot; replace others with '-'
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in name)

def run_sample(
    ex,
    model_name: str,
    max_iter: int,
    debug: bool,
    output_base: str,
    lora_name: Optional[str] = None,
    model_ctxopt: Optional[Any] = None,
    co_config: Optional[Dict[str, Any]] = None,
    experiment_name: str = "smolagents_musique",
):
    os.makedirs(output_base, exist_ok=True)

    env_cfg = SmolagentsEnvConfig(
        experiment_name=experiment_name,
        max_interactions=max_iter,
        debug_mode=debug,
        local_workdir=os.path.join(output_base, "workdir"),
    )
    env = SmolagentsEnv(config=env_cfg)

    # Task: we pass the question as the instruction
    env.reset(seed=0, task=ex.question)

    # Build exp config as attributes (MemoryManager expects attribute access)
    agent_cfg = SimpleNamespace(
        debug_mode=debug,
        max_iter=max_iter,
        co_config=co_config,
    )

    task_cfg = {
        "task_id": ex.id,
        "question": ex.question,
    }

    agent = create_smolagents_agent(
        model_name=model_name,
        key="",
        env=env,
        task_config=task_cfg,
    exp_config=agent_cfg,
        lora_name=lora_name,
    model_ctxopt=model_ctxopt,
        debug_mode=debug,
    )

    result = agent.run(env, max_iter=max_iter)
    # Try to pull final answer from info
    info = result.get("info", {}) or {}
    pred = info.get("final_answer_raw") or info.get("final_answer") or ""

    # Persist traces
    sample_dir = os.path.join(output_base, ex.id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, "sample.json"), "w", encoding="utf-8") as f:
        json.dump({
            "id": ex.id,
            "question": ex.question,
            "answer": ex.answer,
            "prediction": pred,
            "result": result,
        }, f, indent=2, ensure_ascii=False)

    # Save histories
    agent.dump_history(sample_dir)
    env.dump_history(sample_dir)

    return pred, result


def main(
    split: str = "dev",
    output_dir: str = "outputs/smolagents_musique",
    model_name: str = "gpt-4o-mini",
    max_iter: int = 8,
    limit: Optional[int] = None,
    debug: bool = False,
    lora_name: Optional[str] = None,
    tag: Optional[str] = None,
    data_folder: Optional[str] = None,
    co_config_path: Optional[str] = None,
    id_list_file: Optional[str] = None,
):
    # Resolve dataset path from split/data_folder if provided.
    # Desired: choose file by split (train/test), reading from data_folder.
    if data_folder:
        f = (split or "test").lower()
        # Map common aliases
        if f in {"dev", "validation", "val"}:
            f = "test"
        elif f not in {"train", "test"}:
            f = "test"
        root = data_folder or "data/4hop_hf"
        this_dir = os.path.dirname(__file__)
        root_abs = root if os.path.isabs(root) else os.path.abspath(os.path.join(this_dir, root))
        fname = "train.jsonl" if f == "train" else "test.jsonl"
        resolved_data_path = os.path.join(root_abs, fname)
        # Normalize split to resolved fold alias for downstream naming
        split = f

    # Load context optimization config if provided
    co_config = None
    model_ctxopt = None
    if co_config_path:
        if os.path.exists(co_config_path):
            try:
                with open(co_config_path, "r") as f:
                    co_config = yaml.safe_load(f)
            except Exception:
                co_config = None
        else:
            print(f"Warning: co_config_path {co_config_path} not found; continuing without ctxopt.")

    # Initialize local ctxopt model if requested
    if co_config and co_config.get("model_type") == "local":
        try:
            from productive_agents.llm import vLLMLocal
            model_ctxopt = vLLMLocal(co_config["model"], lora_path=co_config.get("lora_name"))
        except Exception as e:
            print(f"Warning: failed to initialize local ctxopt model: {e}")
            model_ctxopt = None

    # Optional filtering: load ID list if provided
    filter_ids = None
    if id_list_file:
        if os.path.exists(id_list_file):
            with open(id_list_file, 'r', encoding='utf-8') as f:
                filter_ids = {line.strip() for line in f if line.strip() and not line.strip().startswith('#')}
            if not filter_ids:
                print(f"Warning: id_list_file {id_list_file} contained no usable IDs; proceeding without filtering.")
                filter_ids = None
        else:
            print(f"Warning: id_list_file {id_list_file} not found; proceeding without filtering.")
            filter_ids = None

    # Dataset: file-backed if provided (resolved or explicit), otherwise a tiny inline demo set
    if 'resolved_data_path' in locals() and resolved_data_path:
        loader = QALoader(resolved_data_path)
        # If filtering, materialize and filter for accurate counts; else use streaming iterator
        if filter_ids is not None:
            materialized = [ex for ex in loader.iter(limit=None) if ex.id in filter_ids]
            if limit is not None:
                materialized = materialized[:limit]
            iterator = materialized
            total_count = len(materialized)
        else:
            iterator = loader.iter(limit=limit)
            # Best-effort total for ETA
            try:
                total_count = loader_count = None
                if hasattr(loader, 'count'):
                    loader_count = loader.count(limit=limit)
                total_count = loader_count
            except Exception:
                total_count = None
    else:  # demo data path
        demo = [
            QAExample(id="demo1", question="Where did the leader of the largest European country after the collapse of the country that denied anything more than an advisory role in the Korean war die?", answer="Moscow"),
            QAExample(id="demo2", question="What is the capital of France?", answer="Paris"),
        ]
        if filter_ids is not None:
            demo = [ex for ex in demo if ex.id in filter_ids]
        demo_list = demo[: limit or len(demo)]
        iterator = demo_list
        total_count = len(demo_list)

    # Save under outputs/{model_name}_{tag}/{fold}
    model_part = _sanitize_for_path(model_name)
    tag_part = _sanitize_for_path(tag) if tag else "notag"
    split_part = _sanitize_for_path((split or "test").lower())
    outputs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    output_dir = os.path.join(outputs_root, f"{model_part}_{tag_part}", split_part)
    os.makedirs(output_dir, exist_ok=True)

    experiment_name = f"smolagents_musique_{split_part}" + (f"_{tag_part}" if tag_part else "")

    n = 0
    correct = 0
    f1_sum = 0.0
    all_rows = []

    # Progress bar with elapsed and ETA
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task_id = progress.add_task("Running MuSiQue", total=total_count)
        try:
            for ex in iterator:
                pred, result = run_sample(
                    ex=ex,
                    model_name=model_name,
                    max_iter=max_iter,
                    debug=debug,
                    output_base=os.path.join(output_dir, "samples"),
                    lora_name=lora_name,
                    model_ctxopt=model_ctxopt,
                    co_config=co_config,
                    experiment_name=experiment_name,
                )

                pred = [p.strip() for p in pred.split(";")]
                # each _answer is the list
                em_list = [exact_match(_pred, _answer) for _pred, _answer in zip(pred, ex.answer)]
                f1_list = [f1_max(_pred, _answer) for _pred, _answer in zip(pred, ex.answer)]

                em_score = sum(em_list) / len(ex.answer)
                f1_score = sum(f1_list) / len(ex.answer)

                correct += em_score
                f1_sum += f1_score
                n += 1
                all_rows.append({
                    "id": ex.id,
                    "question": ex.question,
                    "answer": ex.answer,
                    "prediction": pred,
                    "em": em_score,
                    "f1": f1_score,
                    "iterations": result.get("iterations", 0),
                    "success": result.get("success", False),
                })

                # advance progress
                progress.advance(task_id, 1)
        except KeyboardInterrupt:
            progress.console.print("\nInterrupted by user (Ctrl+C). Finishing up...")

    summary = {
        "total": n,
        "avg_em": (correct / n) if n else 0.0,
        "avg_f1": (f1_sum / n) if n else 0.0,
        "model": model_name,
        "split": split,
        "tag": tag,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "limit": limit,
        "max_iter": max_iter,
        "co_config_path": co_config_path,
        "id_list_file": id_list_file,
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))

    # Pretty summary table with rich
    console = Console()
    table = Table(title="Smolagents MuSiQue Results", show_lines=False)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    em_rate = (correct / n * 100.0) if n else 0.0
    avg_f1 = (f1_sum / n) if n else 0.0
    table.add_row("Model", model_name)
    table.add_row("Tag", str(tag) if tag else "-")
    table.add_row("Split", str(split))
    table.add_row("Total", str(n))
    table.add_row("EM", f"{correct} ({em_rate:.1f}%)")
    table.add_row("Avg F1", f"{avg_f1:.3f}")
    table.add_row("Max Iter", str(max_iter))
    table.add_row("Limit", str(limit))
    table.add_row("Output Dir", output_dir)
    try:
        # resolved_data_path may not exist in some branches
        table.add_row("Data", resolved_data_path)
    except NameError:
        pass
    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run minimal Smolagents MuSiQue eval")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--output_dir", type=str, default="outputs/smolagents_multi_8")
    parser.add_argument("--model_name", type=str, default="gpt-4.1")
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lora_name", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None, help="Tag name to group outputs (saved under output_dir/tag)")
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/nq_multi_8",
        help="Folder containing dataset files (expects train_4hop.jsonl/test_4hop.jsonl)",
    )
    parser.add_argument("--co_config_path", type=str, default=None, help="Context optimization config file path")
    parser.add_argument("--id_list_file", type=str, default=None, help="Optional file containing example IDs (one per line) to restrict the run")

    args = parser.parse_args()

    main(
        split=args.split,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_iter=args.max_iter,
        limit=args.limit,
        debug=args.debug,
        lora_name=args.lora_name,
        tag=args.tag,
        data_folder=args.data_folder,
        co_config_path=args.co_config_path,
        id_list_file=args.id_list_file,
    )
