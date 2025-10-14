"""Path and evaluation file helpers shared across prompt optimizer scripts."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any


def read_eval_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Eval file not found: {path}")
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:  # pragma: no cover - just defensive
        raise RuntimeError(f"Evaluation file not valid JSON: {path}: {e}")


def _experiments_root() -> Path:
    return Path(__file__).resolve().parent.parent  # prompt_optimizer/ -> experiments/prompt_optimizer


def infer_paths_appworld(run_id: str, split: str) -> Dict[str, Path]:
    experiments_root = _experiments_root().parent  # experiments/
    appworld_root = experiments_root / 'appworld'
    model_output_dir = appworld_root / 'outputs' / run_id / split
    eval_dir = appworld_root / 'experiments' / 'outputs' / run_id / 'evaluations'
    eval_file_json = eval_dir / f'{split}.json'
    eval_file_txt = eval_dir / f'{split}.txt'
    if eval_file_json.exists():
        eval_file = eval_file_json
    elif eval_file_txt.exists():
        eval_file = eval_file_txt
    else:
        raise FileNotFoundError(f"No eval file found for run {run_id} at {eval_file_json} or {eval_file_txt}")
    if not model_output_dir.exists():
        raise FileNotFoundError(f"Model output dir not found for run {run_id}: {model_output_dir}")
    return {
        'model_output_dir': model_output_dir.resolve(),
        'eval_output_file': eval_file.resolve(),
    }


def infer_paths_smolagents(run_id: str, split: str) -> Dict[str, Path]:
    experiments_root = _experiments_root().parent  # experiments/
    smol_root = experiments_root / 'smolagents' / 'outputs' / run_id / split
    eval_file = smol_root / 'evaluations.json'
    if not eval_file.exists():
        raise FileNotFoundError(f"Smolagents eval file not found: {eval_file}")
    if not smol_root.exists():
        raise FileNotFoundError(f"Smolagents model output dir not found for run {run_id}: {smol_root}")
    return {
        'model_output_dir': smol_root.resolve(),
        'eval_output_file': eval_file.resolve(),
    }


def infer_paths_officebench(run_id: str, split: str) -> Dict[str, Path]:
    """Infer OfficeBench model output dir and evaluation file.

    New format (preferred):
      {run_id}_{split}_result_overall.json
    Legacy fallback:
      experiment_summary.json
    """
    experiments_root = _experiments_root().parent  # experiments/
    office_root = experiments_root / 'officebench' / 'outputs' / run_id / split
    if not office_root.exists():
        raise FileNotFoundError(f"OfficeBench model output dir not found: {office_root}")
    new_file = office_root / f"{run_id}_{split}_result_overall.json"
    legacy_file = office_root / 'experiment_summary.json'
    if new_file.exists():
        eval_file = new_file
    elif legacy_file.exists():
        eval_file = legacy_file
    else:
        raise FileNotFoundError(f"No OfficeBench eval file found: {new_file} or {legacy_file}")
    return {'model_output_dir': office_root.resolve(), 'eval_output_file': eval_file.resolve()}
