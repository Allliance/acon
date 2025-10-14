#!/usr/bin/env python3
"""Unified Observation Context Reduction -> Prompt Update Pipeline

Phases:
  analysis : Run observation context reduction on an observation-optimized run.
  update   : Sample analyses to produce length-optimized observation prompt templates.
  both     : (default) analysis then update.

Update-only requires --analysis-dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2  # noqa: F401
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from productive_agents.llm import AzureOpenAIServerModel, ChatGPT  # noqa: E402
from common.llm import JinjaLLMTemplate  # type: ignore
from common.config import load_config_file_and_merge  # type: ignore


def setup_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Unified Observation Context Reduction + Prompt Update')
    p.add_argument('--phase', choices=['both', 'analysis', 'update'], default='both')
    # Analysis args
    p.add_argument('--optimized-run', type=str)
    p.add_argument('--optimized-model-output-dir', type=Path)
    p.add_argument('--optimized-eval-output-file', type=Path)
    p.add_argument('--task-split', type=str, default='train')
    p.add_argument('--benchmark', type=str, choices=['appworld', 'smolagents'], default='appworld')
    p.add_argument('--analysis-model', type=str, default='o3')
    default_analysis_prompt = Path(__file__).parent / 'prompts' / 'observation_context_reducer_analysis_prompt.jinja'
    p.add_argument('--analysis-prompt-template', type=Path, default=default_analysis_prompt)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--skip-existing-analysis', action='store_true')
    p.add_argument('--min-keep-ratio', type=float, default=0.35)
    p.add_argument('--smolagents-min-f1', type=float, default=0.6)
    # Update args
    p.add_argument('--analysis-dir', type=Path)
    p.add_argument('--base-prompt-template', type=Path)
    p.add_argument('--optimizer-template', type=Path, default=Path('prompts/observation_context_prompt_optimizer_prompt_by_samples.jinja'))
    p.add_argument('--update-model', type=str, default='o3')
    p.add_argument('--num-prompts', type=int, default=5)
    p.add_argument('--samples-per-prompt', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    # LLM Backend options
    p.add_argument('--llm-backend', type=str, choices=['azure', 'openai'], default='azure',
                   help='LLM backend to use: azure (AzureOpenAIServerModel) or openai (ChatGPT)')
    # General
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--log-level', type=str, default='INFO')
    p.add_argument('--config', type=Path)
    return p


# --------- Analysis (observation) --------- #

class ObservationReducer(JinjaLLMTemplate):
    def __init__(self, model_name: str, prompt_template: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=prompt_template, llm_backend=llm_backend)
    def build_prompt(self, **kw) -> str:
        return self.render(**kw)
    def reduce(self, prompt: str) -> str:
        return self.generate(prompt)


def read_eval_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def infer_paths_appworld(run_id: str, split: str) -> Dict[str, Path]:
    experiments_root = Path(__file__).parent.parent
    appworld_root = experiments_root / 'appworld'
    model_output_dir = appworld_root / 'outputs' / run_id / split
    eval_dir = appworld_root / 'experiments' / 'outputs' / run_id / 'evaluations'
    ej = eval_dir / f'{split}.json'; et = eval_dir / f'{split}.txt'
    if ej.exists(): ev = ej
    elif et.exists(): ev = et
    else: raise FileNotFoundError(f'No eval file for run {run_id}')
    if not model_output_dir.exists():
        raise FileNotFoundError(f'Model output dir not found: {model_output_dir}')
    return {'model_output_dir': model_output_dir.resolve(), 'eval_output_file': ev.resolve()}


def infer_paths_smolagents(run_id: str, split: str) -> Dict[str, Path]:
    experiments_root = Path(__file__).parent.parent
    smol_root = experiments_root / 'smolagents' / 'outputs' / run_id / split
    eval_file = smol_root / 'evaluations.json'
    if not eval_file.exists(): raise FileNotFoundError(f'Missing eval: {eval_file}')
    if not smol_root.exists(): raise FileNotFoundError(f'Model output dir missing: {smol_root}')
    return {'model_output_dir': smol_root.resolve(), 'eval_output_file': eval_file.resolve()}


def approx_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def flatten_history(session_any: Any) -> str:
    if not session_any: return ''
    lines: List[str] = []
    for msg in session_any:
        if isinstance(msg, dict): role = msg.get('role','UNKNOWN'); content = msg.get('content','')
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2: role, content = msg[0], msg[1]
        else: continue
        if not isinstance(role, str): role = str(role)
        text = ''
        if isinstance(content, str): text = content
        elif isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, str): parts.append(c)
                elif isinstance(c, dict):
                    for k in ('text','value','content'):
                        v = c.get(k)
                        if isinstance(v,str): parts.append(v); break
                else: parts.append(str(c))
            text = '\n'.join(p for p in parts if p)
        elif isinstance(content, dict):
            for k in ('text','value','content','message'):
                v = content.get(k)
                if isinstance(v,str): text = v; break
        lines.append(f"{role.upper()}:\n{text}\n")
    return '\n'.join(lines).strip()


def process_obs_task(task_name: str, optimized_task_dir: Path, reducer: ObservationReducer, output_task_dir: Path,
                     skip_existing: bool) -> Optional[List[Dict[str, Any]]]:
    obs_hist_path = optimized_task_dir / 'obs_optimizer_history.json'
    hist_path = optimized_task_dir / 'llm_history.json'
    if not obs_hist_path.exists() or not hist_path.exists(): return None
    try:
        optimizer_history = json.loads(obs_hist_path.read_text())
        optimized_history_raw = json.loads(hist_path.read_text())
    except Exception as e:
        logging.warning(f'History parse fail {task_name}: {e}')
        return None
    session0 = optimized_history_raw[0] if isinstance(optimized_history_raw, list) and optimized_history_raw else optimized_history_raw
    next_context_text = flatten_history(session0)
    num_passes = len(optimizer_history) if isinstance(optimizer_history, list) else 0
    if num_passes <= 0: return None
    metas: List[Dict[str, Any]] = []
    for idx in range(num_passes):
        pass_num = idx + 1
        pass_dir = output_task_dir / f'pass_{pass_num}'
        res_file = pass_dir / 'analysis.json'
        if skip_existing and res_file.exists(): continue
        item = optimizer_history[idx]
        refined_obs = None; original_obs = None
        if isinstance(item, list):
            try: refined_obs = item[2]
            except Exception: refined_obs = None
            try: original_obs = item[3].get('observation') if isinstance(item[3], dict) else None
            except Exception: original_obs = None
        elif isinstance(item, dict):
            refined_obs = item.get('refined') or item.get('response'); original_obs = item.get('original') or item.get('observation')
        refined_str = str(refined_obs or ''); original_str = str(original_obs or '')
        obs_pair_text = f"--- OBS OPT PASS {pass_num} ---\nORIGINAL_OBSERVATION:\n{original_str}\n\nREFINED_OBSERVATION:\n{refined_str}\n"
        orig_chars = len(refined_str)
        prompt = reducer.build_prompt(task_name=task_name, obs_pair_text=obs_pair_text, next_context=next_context_text, orig_chars=orig_chars)
        raw = reducer.reduce(prompt).strip()
        pass_dir.mkdir(parents=True, exist_ok=True)
        (pass_dir / 'prompt.txt').write_text(prompt)
        m = re.search(r"<<<ANALYSIS_JSON>>>\n([\s\S]*?)\n<<<END>>>", raw)
        meta = {'task_name': task_name, 'pass': pass_num, 'orig_chars': orig_chars, 'orig_tokens_approx': approx_token_count(refined_str)}
        if m:
            block = m.group(1).strip()
            try: js = json.loads(block)
            except Exception:
                m2 = re.search(r"\{[\s\S]*\}", raw)
                if m2:
                    try: js = json.loads(m2.group(0))
                    except Exception: js = {'raw': raw}
                else: js = {'raw': raw}
            (pass_dir / 'analysis.json').write_text(json.dumps(js, indent=2))
            (pass_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
            metas.append(meta)
        else:
            (pass_dir / 'analysis.raw.txt').write_text(raw)
            (pass_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return metas or None


def run_observation_analysis(cfg: Dict[str, Any]) -> Optional[Path]:
    if not cfg.get('optimized_model_output_dir') or not cfg.get('optimized_eval_output_file'):
        if not cfg.get('optimized_run'):
            raise SystemExit('Need --optimized-run or explicit optimized paths')
        paths = infer_paths_appworld(cfg['optimized_run'], cfg['task_split']) if cfg['benchmark'] == 'appworld' else infer_paths_smolagents(cfg['optimized_run'], cfg['task_split'])
        optimized_model_output_dir = paths['model_output_dir']; optimized_eval_output_file = paths['eval_output_file']
    else:
        optimized_model_output_dir = Path(cfg['optimized_model_output_dir']).resolve(); optimized_eval_output_file = Path(cfg['optimized_eval_output_file']).resolve()
    
    eval_data = read_eval_file(optimized_eval_output_file); individual = eval_data.get('individual', {})
    if cfg['benchmark'] in ('smolagents'):
        tasks_dir = optimized_model_output_dir / 'samples'
        all_task_dirs = {d.name: d for d in tasks_dir.iterdir() if d.is_dir()} if tasks_dir.exists() else {}
        thr = float(cfg.get('smolagents_min_f1', 0.6))
        task_names: List[str] = []
        for t, rec in individual.items():
            if t not in all_task_dirs:
                continue
            try:
                if float(rec.get('f1', 0.0)) >= thr:
                    task_names.append(t)
            except Exception:
                continue
    else:
        tasks_dir = optimized_model_output_dir; all_task_dirs = {d.name.lstrip('task_'): d for d in tasks_dir.iterdir() if d.is_dir()} if tasks_dir.exists() else {}
        task_names = [t for t, rec in individual.items() if rec.get('success') is True and t in all_task_dirs]
    task_names.sort()
    if cfg.get('max_tasks'): task_names = task_names[: int(cfg['max_tasks'])]
    if not task_names:
        logging.info('No successful observation tasks.')
        return None
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True); (out / 'reduction_results').mkdir(exist_ok=True)
    reducer = ObservationReducer(
        model_name=cfg['analysis_model'], 
        prompt_template=cfg['analysis_prompt_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    aggregated: List[Dict[str, Any]] = []
    for t in tqdm(task_names, desc='Observation ctx analysis', unit='task'):
        meta = process_obs_task(t, all_task_dirs[t], reducer, out / 'reduction_results' / t, cfg.get('skip_existing_analysis', False))
        if meta: aggregated.extend(meta)
    if not aggregated:
        logging.warning('No observation context analyses produced.')
        return None
    (out / 'aggregated_analyses.json').write_text(json.dumps(aggregated, indent=2))
    logging.info(f'Wrote {out / "aggregated_analyses.json"}')
    return out


# --------- Update (prompt) --------- #

class ObservationContextPromptOptimizer(JinjaLLMTemplate):
    def __init__(self, model_name: str, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path='', llm_backend=llm_backend)

    def forward(self, prompt: str) -> str: 
        return self.llm.generate(prompt)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"<<<ANALYSIS_JSON>>>\n([\s\S]*?)\n<<<END>>>", text)
    if m:
        block = m.group(1).strip()
        try: return json.loads(block)
        except Exception: pass
    return None


def find_analysis_files(root: Path) -> List[Path]:
    res: List[Path] = []
    for p in root.rglob('reduction_results/*/pass_*/analysis.json'):
        if p.is_file(): res.append(p)
    for p in root.rglob('reduction_results/*/pass_*/analysis.raw.txt'):
        if p.is_file(): res.append(p)
    return sorted(res)


def load_analysis(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.suffix == '.json': return json.loads(path.read_text())
        raw = path.read_text(); return _extract_json(raw)
    except Exception: return None


def run_observation_update(cfg: Dict[str, Any], analysis_dir: Path) -> None:
    base_template = Path(cfg['base_prompt_template']).resolve()
    if not base_template.exists(): raise SystemExit(f'Base prompt template missing: {base_template}')
    analysis_files = find_analysis_files(analysis_dir)
    if not analysis_files:
        logging.warning('No analysis files for observation update.')
        return
    random.seed(int(cfg.get('seed', 42)))
    original_prompt = base_template.read_text()
    out_dir = Path(cfg['output_dir'])
    opt_dir = out_dir / 'optimized_prompts'; opt_dir.mkdir(exist_ok=True)
    optimizer = ObservationContextPromptOptimizer(
        model_name=cfg['update_model'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    num_prompts = int(cfg['num_prompts']); spp = int(cfg['samples_per_prompt'])
    tmpl = jinja2.Template(Path(cfg['optimizer_template']).read_text())
    manifest: List[Dict[str, Any]] = []
    for i in range(num_prompts):
        k = min(spp, len(analysis_files))
        selected = random.sample(analysis_files, k) if k < len(analysis_files) else list(analysis_files)
        payload: List[Dict[str, Any]] = []
        total_chars = 0; n_chars = 0
        for f in selected:
            data = load_analysis(f)
            if not data: continue
            overview = data.get('overview') or {}
            if 'orig_chars' in overview:
                try: total_chars += int(overview.get('orig_chars')); n_chars += 1
                except Exception: pass
            removals = [r for r in data.get('remove_or_shorten', []) if isinstance(r, dict)]
            keeps = [r for r in data.get('keep', []) if isinstance(r, dict)]
            rules = [r for r in data.get('summarize_rules', []) if isinstance(r, str)]
            parts = f.parts; task_label = 'unknown'; pass_label = None
            if 'reduction_results' in parts:
                idx = parts.index('reduction_results')
                if idx + 1 < len(parts): task_label = parts[idx+1]
                if idx + 2 < len(parts) and parts[idx+2].startswith('pass_'): pass_label = parts[idx+2]
            payload.append({'task_label': task_label, 'pass': pass_label, 'overview': overview, 'removals': removals, 'keeps': keeps, 'rules': rules})
        avg_chars = int(total_chars / n_chars) if n_chars else None
        prompt_text = tmpl.render(original_prompt=original_prompt, avg_orig_chars=avg_chars, samples=payload)
        (opt_dir / f'optimizer_prompt_input_{i}.txt').write_text(prompt_text)
        improved = optimizer.forward(prompt_text)
        out_file = opt_dir / f'length_optimized_observation_prompt_{i}.jinja'
        out_file.write_text(improved)
        manifest.append({'index': i, 'output_file': str(out_file.relative_to(out_dir)), 'n_samples_used': len(payload), 'avg_orig_chars': avg_chars})
        logging.info(f'Generated observation length-optimized prompt {i}: {out_file}')
    (opt_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    logging.info('Observation context prompt update complete')


def main():  # pragma: no cover
    args = setup_args().parse_args()
    cfg = load_config_file_and_merge(args)
    
    logging.basicConfig(level=getattr(logging, cfg.get('log_level', 'INFO').upper()), format='%(asctime)s %(levelname)s %(message)s')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    analysis_dir: Optional[Path] = None
    if cfg['phase'] in ('both','analysis'):
        analysis_dir = run_observation_analysis(cfg)
        if cfg['phase'] == 'both' and analysis_dir is None:
            logging.info('Analysis produced no results; skipping update phase.')
            return
    if cfg['phase'] == 'update' and not cfg.get('analysis_dir') and not analysis_dir:
        raise SystemExit('Provide --analysis-dir for update-only phase')
    if cfg['phase'] in ('both','update'):
        if not analysis_dir:
            provided = cfg.get('analysis_dir')
            if not provided:
                raise SystemExit('No analysis results available and no --analysis-dir supplied')
            analysis_dir = Path(provided).resolve()
        if not analysis_dir.exists(): raise SystemExit(f'Analysis directory missing: {analysis_dir}')
        if not cfg.get('base_prompt_template'): raise SystemExit('Need --base-prompt-template for update phase')
        run_observation_update(cfg, analysis_dir)


if __name__ == '__main__':
    main()
