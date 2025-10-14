#!/usr/bin/env python3
"""Unified Observation Prompt Update Pipeline

Phase 1: Run observation regression analysis (baseline vs observation optimized)
Phase 2: Generate improved observation compression prompt variants using per-sample
         sampling over aggregated_observation_regressions.json

Run both phases (default) or individually with --phase.
For update-only runs, pass --aggregated-observation-regressions pointing at an
existing aggregated file.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2  # noqa: F401
from tqdm import tqdm  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from productive_agents.llm import AzureOpenAIServerModel, ChatGPT  # noqa: E402
from common.paths import infer_paths_appworld, infer_paths_smolagents, infer_paths_officebench  # type: ignore
from common.config import load_config_file_and_merge  # type: ignore
from common.system_prompts import copy_system_prompts_from_source  # type: ignore
from common.aggregation import dedupe_list  # type: ignore
from common.llm import JinjaLLMTemplate  # type: ignore


# ------------- Args ------------- #


def setup_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Unified Observation Prompt Update Pipeline')
    p.add_argument('--phase', choices=['both', 'analysis', 'update'], default='both')

    # Analysis
    p.add_argument('--baseline-run', type=str)
    p.add_argument('--optimized-run', type=str)
    p.add_argument('--baseline-model-output-dir', type=Path)
    p.add_argument('--baseline-eval-output-file', type=Path)
    p.add_argument('--optimized-model-output-dir', type=Path)
    p.add_argument('--optimized-eval-output-file', type=Path)
    p.add_argument('--task-split', type=str, default='train')
    p.add_argument('--benchmark', type=str, choices=['appworld', 'smolagents', 'officebench'], default='appworld')
    p.add_argument('--analysis-model', type=str, default='o3')
    p.add_argument('--analysis-prompt-template', type=Path, default=Path('prompts/observation_regression_prompt.jinja'))
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--skip-existing-analysis', action='store_true')

    # Update
    p.add_argument('--aggregated-observation-regressions', type=Path)
    p.add_argument('--base-prompt-template', type=Path)
    p.add_argument('--optimizer-template', type=Path, default=Path('prompts/observation_prompt_optimizer_prompt_by_samples.jinja'))
    p.add_argument('--update-model', type=str, default='o3')
    p.add_argument('--num-prompts', type=int, default=5)
    p.add_argument('--samples-per-prompt', type=int, default=30)
    p.add_argument('--max-problematic', type=int)
    p.add_argument('--max-missing', type=int)
    p.add_argument('--max-suggestions', type=int)
    p.add_argument('--dedupe', action='store_true')
    p.add_argument('--seed', type=int, default=42)

    # LLM Backend options
    p.add_argument('--llm-backend', type=str, choices=['azure', 'openai'], default='azure',
                   help='LLM backend to use: azure (AzureOpenAIServerModel) or openai (ChatGPT)')

    # General
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--log-level', type=str, default='INFO')
    p.add_argument('--config', type=Path)
    return p


# ------------- Analysis helpers ------------- #


class ObservationRegressionAnalyzer(JinjaLLMTemplate):
    def __init__(self, model_name: str, prompt_template: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=prompt_template, llm_backend=llm_backend)

    def build_prompt(self, **kwargs) -> str:  # type: ignore[override]
        return self.render(**kwargs)

    def analyze(self, prompt: str) -> str:
        return self.generate(prompt)

def flatten(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, str): parts.append(c)
            elif isinstance(c, dict):
                for k in ('text', 'value', 'content'):
                    v = c.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                        break
            else: parts.append(str(c))
        return '\n'.join(p for p in parts if p)
    if isinstance(content, dict):
        for k in ('text', 'value', 'content', 'message'):
            v = content.get(k)
            if isinstance(v, str): return v
            if isinstance(v, (list, dict)): return flatten(v)
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def render_history(raw: Any) -> str:
    if not raw: return ''
    lines: List[str] = []
    for msg in raw:
        if isinstance(msg, dict):
            role = msg.get('role', 'UNKNOWN')
            content = msg.get('content')
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role, content = msg[0], msg[1]
        else:
            continue
        if not isinstance(role, str): role = str(role)
        lines.append(f"{role.upper()}:\n{flatten(content).strip()}\n")
    return '\n'.join(lines).strip()


def load_optimizer_outputs(task_dir: Path) -> Optional[List[Any]]:
    p = task_dir / 'obs_optimizer_history.json'
    if not p.exists(): return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list): return data
    except Exception:
        return None
    return None


def process_obs_task(task: str, b_dir: Path, o_dir: Path, failure_report: Any, analyzer: ObservationRegressionAnalyzer,
                     out_dir: Path, skip_existing: bool) -> Optional[Dict[str, Any]]:
    res_file = out_dir / 'observation_regression_analysis.json'
    if skip_existing and res_file.exists(): return None
    b_hist = b_dir / 'llm_history.json'
    o_hist = o_dir / 'llm_history.json'
    if not b_hist.exists() or not o_hist.exists(): return None
    try:
        b_raw = json.loads(b_hist.read_text())
        o_raw = json.loads(o_hist.read_text())
    except Exception as e:
        logging.warning(f'History parse error {task}: {e}')
        return None
    b_txt = render_history(b_raw[0]) if isinstance(b_raw, list) and b_raw else render_history(b_raw)
    o_txt = render_history(o_raw[0]) if isinstance(o_raw, list) and o_raw else render_history(o_raw)
    opt_outputs = load_optimizer_outputs(o_dir)
    if not opt_outputs: return None
    opt_hist_parts: List[str] = []
    for i, op in enumerate(opt_outputs):
        try:
            refined = op[2]
            original = op[3]['observation']
            opt_hist_parts.append(f"=== Observation optimization #{i} ===\n\nOriginal observation:\n{original}\n\nRefined observation:\n{refined}")
        except Exception:
            continue
    opt_hist_txt = '\n\n'.join(opt_hist_parts)
    prompt = analyzer.build_prompt(task_name=task, baseline_history=b_txt, optimized_history=o_txt,
                                   optimization_history_txt=opt_hist_txt, baseline_success=True,
                                   optimized_success=False, failure_report=failure_report)
    raw = analyzer.analyze(prompt).strip()
    block = raw
    if not (block.startswith('{') and block.endswith('}')):
        m = re.search(r'\{[\s\S]*\}$', raw)
        if m: block = m.group(0)
    try:
        parsed = json.loads(block)
    except Exception:
        parsed = {'raw_output': raw}
    parsed['task_name'] = task
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'prompt.txt').write_text(prompt)
    res_file.write_text(json.dumps(parsed, indent=2))
    return parsed


def run_obs_analysis(cfg: Dict[str, Any]) -> Optional[Path]:
    # Resolve model/eval paths per benchmark
    if not cfg.get('baseline_model_output_dir') or not cfg.get('baseline_eval_output_file'):
        if not cfg.get('baseline_run'):
            raise SystemExit('Need --baseline-run or explicit baseline paths')
        if cfg['benchmark'] == 'appworld':
            paths = infer_paths_appworld(cfg['baseline_run'], cfg['task_split'])
        elif cfg['benchmark'] == 'smolagents':
            paths = infer_paths_smolagents(cfg['baseline_run'], cfg['task_split'])
        elif cfg['benchmark'] == 'officebench':
            paths = infer_paths_officebench(cfg['baseline_run'], cfg['task_split'])
        else:
            raise SystemExit('Unsupported benchmark')
        b_model_dir = paths['model_output_dir']; b_eval = paths['eval_output_file']
    else:
        b_model_dir = Path(cfg['baseline_model_output_dir']).resolve(); b_eval = Path(cfg['baseline_eval_output_file']).resolve()
    
    if not cfg.get('optimized_model_output_dir') or not cfg.get('optimized_eval_output_file'):
        if not cfg.get('optimized_run'):
            raise SystemExit('Need --optimized-run or explicit optimized paths')
        if cfg['benchmark'] == 'appworld':
            paths = infer_paths_appworld(cfg['optimized_run'], cfg['task_split'])
        elif cfg['benchmark'] == 'smolagents':
            paths = infer_paths_smolagents(cfg['optimized_run'], cfg['task_split'])
        elif cfg['benchmark'] == 'officebench':
            paths = infer_paths_officebench(cfg['optimized_run'], cfg['task_split'])
        else:
            raise SystemExit('Unsupported benchmark')
        o_model_dir = paths['model_output_dir']; o_eval = paths['eval_output_file']
    else:
        o_model_dir = Path(cfg['optimized_model_output_dir']).resolve(); o_eval = Path(cfg['optimized_eval_output_file']).resolve()
    
    b_eval_data = json.loads(b_eval.read_text()); o_eval_data = json.loads(o_eval.read_text())
    if cfg['benchmark'] == 'officebench':
        b_succ = set(b_eval_data.get('successful_tasks') or [])
        b_fail = set(b_eval_data.get('failed_tasks') or [])
        o_succ = set(o_eval_data.get('successful_tasks') or [])
        o_fail = set(o_eval_data.get('failed_tasks') or [])
        b_ind = {t: {'success': t in b_succ} for t in (b_succ | b_fail)}
        o_ind = {t: {'success': t in o_succ} for t in (o_succ | o_fail)}
    else:
        b_ind = b_eval_data.get('individual', {})
        o_ind = o_eval_data.get('individual', {})
        
    # Regression detection
    regressions: Dict[str, Any] = {}
    if cfg['benchmark'] == 'appworld':
        for t, base_rec in b_ind.items():
            opt_rec = o_ind.get(t)
            if base_rec.get('success') and opt_rec and not opt_rec.get('success'):
                regressions[t] = opt_rec.get('failures')
    elif cfg['benchmark'] == 'smolagents':
        for t, base_rec in b_ind.items():
            opt_rec = o_ind.get(t)
            if not opt_rec:
                continue
            try:
                bf = float(base_rec.get('f1', 0.0)); of = float(opt_rec.get('f1', 0.0))
            except Exception:
                continue
            if bf > of and bf - of > 0.2:
                regressions[t] = opt_rec.get('failures')
    elif cfg['benchmark'] == 'officebench':
        for t, base_rec in b_ind.items():
            opt_rec = o_ind.get(t)
            if base_rec.get('success') and opt_rec and not opt_rec.get('success'):
                regressions[t] = opt_rec.get('failures')
    if not regressions:
        logging.info('No observation regression tasks.')
        return None
    if cfg['benchmark'] == 'smolagents':
        b_tasks_dir = b_model_dir / 'samples'
        o_tasks_dir = o_model_dir / 'samples'
    elif cfg['benchmark'] == 'officebench':
        b_tasks_dir = b_model_dir
        o_tasks_dir = o_model_dir
    else:
        b_tasks_dir = b_model_dir
        o_tasks_dir = o_model_dir
    if not b_tasks_dir.exists() or not o_tasks_dir.exists(): return None
    if cfg['benchmark'] == 'smolagents':
        b_dirs = {d.name: d for d in b_tasks_dir.iterdir() if d.is_dir()}
        o_dirs = {d.name: d for d in o_tasks_dir.iterdir() if d.is_dir()}
    else:
        b_dirs = {d.name.lstrip('task_'): d for d in b_tasks_dir.iterdir() if d.is_dir()}
        o_dirs = {d.name.lstrip('task_'): d for d in o_tasks_dir.iterdir() if d.is_dir()}
    if cfg['benchmark'] == 'officebench':
        # Expand each task dir into subtask dirs named task_subtask
        def expand(dmap: Dict[str, Path]) -> Dict[str, Path]:
            out: Dict[str, Path] = {}
            for tid, tdir in dmap.items():
                if not tdir.is_dir():
                    continue
                subs = [c for c in tdir.iterdir() if c.is_dir() and c.name.isdigit()]
                if not subs:
                    out[f'{tid}_0'] = tdir
                else:
                    for sd in subs:
                        out[f'{tid}_{sd.name}'] = sd
            return out
        b_dirs = expand(b_dirs)
        o_dirs = expand(o_dirs)
    selected = [t for t in regressions if t in b_dirs and t in o_dirs]
    selected.sort()
    for s in selected: print(s)
    if cfg.get('max_tasks'):
        selected = selected[: int(cfg['max_tasks'])]
    if not selected: return None
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'regression_results').mkdir(exist_ok=True)
    # Ensure all Path instances in cfg are stringified for JSON serialization
    serializable_cfg = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    (out / 'analysis_run_config.json').write_text(
        json.dumps(
            dict(
                serializable_cfg,
                baseline_model_output_dir=str(b_tasks_dir),
                optimized_model_output_dir=str(o_tasks_dir),
            ),
            indent=2,
        )
    )
    analyzer = ObservationRegressionAnalyzer(
        model_name=cfg['analysis_model'], 
        prompt_template=cfg['analysis_prompt_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    aggregated: List[Dict[str, Any]] = []
    for t in tqdm(selected, desc='Observation analysis', unit='task'):
        res = process_obs_task(t, b_dirs[t], o_dirs[t], regressions[t], analyzer, out / 'regression_results' / t, cfg.get('skip_existing_analysis', False))
        if res: aggregated.append(res)
    if not aggregated:
        logging.warning('No analyses produced.')
        return None
    agg_path = out / 'aggregated_observation_regressions.json'
    agg_path.write_text(json.dumps(aggregated, indent=2))
    logging.info(f'Wrote {agg_path}')
    return agg_path


# ------------- Update helpers ------------- #


class ObservationPromptOptimizer(JinjaLLMTemplate):
    def __init__(self, model_name: str, template_path: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=template_path, llm_backend=llm_backend)
    def build_prompt(self, **kw) -> str:  # type: ignore[override]
        return self.render(**kw)
    def forward(self, prompt: str) -> str:
        return self.generate(prompt)


def run_observation_update(cfg: Dict[str, Any], agg_path: Path) -> None:
    if not cfg.get('base_prompt_template'): raise SystemExit('Need --base-prompt-template for update phase')
    base_template = Path(cfg['base_prompt_template']).resolve()
    entries = json.loads(agg_path.read_text())
    if not isinstance(entries, list) or not entries:
        logging.warning('Aggregated observation regressions empty')
        return
    random.seed(int(cfg.get('seed', 42)))
    original_prompt = base_template.read_text()
    out_dir = Path(cfg['output_dir'])
    opt_dir = out_dir / 'optimized_prompts'
    opt_dir.mkdir(exist_ok=True)
    copy_system_prompts_from_source(base_template, opt_dir)
    optimizer = ObservationPromptOptimizer(
        model_name=cfg['update_model'], 
        template_path=cfg['optimizer_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    num_prompts = int(cfg['num_prompts'])
    spp = int(cfg['samples_per_prompt'])
    max_prob = cfg.get('max_problematic')
    max_missing = cfg.get('max_missing')
    max_suggestions = cfg.get('max_suggestions')
    manifest: List[Dict[str, Any]] = []
    for i in range(num_prompts):
        k = min(spp, len(entries))
        selected = random.sample(entries, k) if k < len(entries) else list(entries)
        payload: List[Dict[str, Any]] = []
        for r in selected:
            imps: List[str] = []
            rr = r.get('improvement_suggestions')
            if isinstance(rr, list): imps = [str(x).strip() for x in rr if isinstance(x, str)]
            elif isinstance(rr, str): imps = [rr.strip()]
            if cfg.get('dedupe'): imps = dedupe_list(imps)
            if isinstance(max_suggestions, int) and max_suggestions > 0: imps = imps[:max_suggestions]
            prob: List[str] = []
            spans = r.get('problematic_observation_spans')
            if isinstance(spans, list): prob = [str(x).strip() for x in spans if isinstance(x, str)]
            if isinstance(max_prob, int) and max_prob > 0: prob = prob[:max_prob]
            missing: List[str] = []
            spans_m = r.get('missing_observation_spans')
            if isinstance(spans_m, list): missing = [str(x).strip() for x in spans_m if isinstance(x, str)]
            if isinstance(max_missing, int) and max_missing > 0: missing = missing[:max_missing]
            cats: List[str] = []
            c = r.get('issue_categories')
            if isinstance(c, list): cats = [str(x) for x in c if isinstance(x, str)]
            payload.append({'suggestions': imps, 'problematic_spans': prob, 'missing_spans': missing, 'issue_categories': cats})
        prompt_text = optimizer.build_prompt(original_prompt=original_prompt, samples=payload)
        (opt_dir / f'optimizer_prompt_input_{i}.txt').write_text(prompt_text)
        (opt_dir / f'samples_payload_{i}.json').write_text(json.dumps(payload, indent=2))
        improved = optimizer.forward(prompt_text)
        out_file = opt_dir / f'improved_observation_prompt_samples_{i}.jinja'
        out_file.write_text(improved)
        cat_counts: Dict[str, int] = {}
        for s in payload:
            for c in s.get('issue_categories', []) or []:
                if isinstance(c, str): cat_counts[c] = cat_counts.get(c, 0) + 1
        manifest.append({'index': i, 'output_file': str(out_file.relative_to(out_dir)), 'n_samples_used': len(payload), 'issue_categories_top3': sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]})
        logging.info(f'Generated improved observation prompt {i}: {out_file}')
    (opt_dir / 'manifest_by_samples.json').write_text(json.dumps(manifest, indent=2))
    logging.info('Observation prompt update complete')


# ------------- Main ------------- #


def main():  # pragma: no cover
    args = setup_args().parse_args()
    cfg = load_config_file_and_merge(args)
    
    logging.basicConfig(level=getattr(logging, cfg.get('log_level', 'INFO').upper()), format='%(asctime)s %(levelname)s %(message)s')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    agg_path: Optional[Path] = None
    if cfg['phase'] in ('both', 'analysis'):
        agg_path = run_obs_analysis(cfg)
    if cfg['phase'] == 'update' and not cfg.get('aggregated_observation_regressions') and not agg_path:
        raise SystemExit('Provide --aggregated-observation-regressions for update-only phase')
    if cfg['phase'] in ('both', 'update'):
        if not agg_path:
            supplied = cfg.get('aggregated_observation_regressions')
            if not supplied:
                logging.info('No aggregated observation regressions available; skipping update phase.')
                return
            agg_path = Path(supplied).resolve()
        if not agg_path.exists(): raise SystemExit(f'Aggregated observation regressions file missing: {agg_path}')
        run_observation_update(cfg, agg_path)


if __name__ == '__main__':
    main()
