#!/usr/bin/env python3
"""Unified History Prompt Update Pipeline

Runs history regression analysis then immediately generates improved history
prompt variants from the produced aggregated_history_regressions.json.

Phases:
  1) analysis:  baseline vs optimized run => aggregated_history_regressions.json
  2) update:    per-sample sampling => improved_history_prompt_samples_*.jinja

You can run either phase individually with --phase analysis or --phase update.
For --phase update you must supply --aggregated-history-regressions if you did
not just run analysis in the same invocation.
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

import jinja2  # noqa: F401 (kept for template parity)
from tqdm import tqdm  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from productive_agents.llm import AzureOpenAIServerModel, ChatGPT  # noqa: E402
from common.paths import read_eval_file, infer_paths_appworld, infer_paths_smolagents, infer_paths_officebench  # type: ignore
from common.llm import JinjaLLMTemplate  # type: ignore
from common.config import load_config_file_and_merge  # type: ignore
from common.system_prompts import copy_system_prompts_from_source  # type: ignore
from common.aggregation import dedupe_list  # type: ignore


# ---------------- Args ---------------- #


def setup_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified History Prompt Update Pipeline")
    # Phase control
    p.add_argument('--phase', choices=['both', 'analysis', 'update'], default='both')

    # Analysis inputs (matching main_history_regression_analysis.py)
    p.add_argument('--baseline-run', type=str)
    p.add_argument('--optimized-run', type=str)
    p.add_argument('--baseline-model-output-dir', type=Path)
    p.add_argument('--baseline-eval-output-file', type=Path)
    p.add_argument('--optimized-model-output-dir', type=Path)
    p.add_argument('--optimized-eval-output-file', type=Path)
    p.add_argument('--task-split', type=str, default='train')
    p.add_argument('--benchmark', type=str, choices=['appworld', 'smolagents', 'officebench'], default='appworld')
    p.add_argument('--analysis-model', type=str, default='o3')
    p.add_argument('--analysis-prompt-template', type=Path, default=Path('prompts/history_regression_prompt.jinja'))
    p.add_argument('--include-performance-regressions', action='store_true')
    p.add_argument('--step-ratio-threshold', type=float, default=1.5)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--skip-existing-analysis', action='store_true')

    # Update inputs (matching main_update_history_prompt_by_samples.py)
    p.add_argument('--aggregated-history-regressions', type=Path, help='If phase includes update and analysis skipped, supply path')
    p.add_argument('--base-prompt-template', type=Path, help='History base prompt template (jinja)')
    p.add_argument('--optimizer-template', type=Path, default=Path('prompts/history_prompt_optimizer_prompt_by_samples.jinja'))
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


# ---------------- Analysis helpers ---------------- #


class HistoryRegressionAnalyzer(JinjaLLMTemplate):
    def __init__(self, model_name: str, prompt_template: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=prompt_template, llm_backend=llm_backend)

    def build_prompt(self, **kwargs) -> str:  # type: ignore[override]
        return self.render(**kwargs)

    def analyze(self, prompt: str) -> str:
        return self.generate(prompt)


def flatten_message_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                for k in ('text', 'value', 'content'):
                    v = c.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                        break
            else:
                parts.append(str(c))
        return '\n'.join(p for p in parts if p)
    if isinstance(content, dict):
        for k in ('text', 'value', 'content', 'message'):
            v = content.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, (list, dict)):
                return flatten_message_content(v)
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def render_flat_history(llm_history: Any) -> str:
    if not llm_history:
        return ''
    lines: List[str] = []
    for msg in llm_history:
        role = 'UNKNOWN'
        content: Any = ''
        if isinstance(msg, dict):
            role = msg.get('role', 'UNKNOWN')
            content = msg.get('content', '')
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            role, content = msg[0], msg[1]
        else:
            continue
        if not isinstance(role, str):
            role = str(role)
        text = flatten_message_content(content).strip()
        lines.append(f"{role.upper()}:\n{text}\n")
    return '\n'.join(lines).strip()


def render_multi_session_history(sessions: List[Any]) -> str:
    rendered: List[str] = []
    for i, session in enumerate(sessions):
        rendered.append(f"===== SESSION {i+1} START =====\n{render_flat_history(session)}\n===== SESSION {i+1} END =====")
    return '\n\n'.join(rendered)


def extract_histories(baseline_raw: Any, optimized_raw: Any) -> Tuple[str, str]:
    if isinstance(baseline_raw, list) and baseline_raw and all(isinstance(x, list) for x in baseline_raw):
        baseline_txt = render_flat_history(baseline_raw[0])
    else:
        baseline_txt = render_flat_history(baseline_raw)
    if isinstance(optimized_raw, list) and optimized_raw and all(isinstance(x, list) for x in optimized_raw):
        optimized_txt = render_multi_session_history(optimized_raw)
    else:
        optimized_txt = render_flat_history(optimized_raw)
    return baseline_txt, optimized_txt


def load_env_steps(task_dir: Path) -> Optional[int]:
    p = task_dir / 'env_history.json'
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return len(data)
    except Exception:
        return None
    return None


def process_history_task(task_name: str, baseline_task_dir: Path, optimized_task_dir: Path, failure_report: Any,
                         analyzer: HistoryRegressionAnalyzer, out_dir: Path, skip_existing: bool,
                         baseline_success: bool, optimized_success: bool,
                         include_perf: bool, step_ratio_threshold: float) -> Optional[Dict[str, Any]]:
    res_file = out_dir / 'history_regression_analysis.json'
    if skip_existing and res_file.exists():
        logging.info(f'Skip existing analysis {task_name}')
        return None
    b_hist = baseline_task_dir / 'llm_history.json'
    o_hist = optimized_task_dir / 'llm_history.json'
    if not b_hist.exists() or not o_hist.exists():
        return None
    try:
        b_raw = json.loads(b_hist.read_text())
        o_raw = json.loads(o_hist.read_text())
    except Exception as e:
        logging.warning(f'History parse error {task_name}: {e}')
        return None
    b_txt, o_txt = extract_histories(b_raw, o_raw)
    b_steps = load_env_steps(baseline_task_dir)
    o_steps = load_env_steps(optimized_task_dir)
    step_ratio = None
    perf_reg = False
    if b_steps and o_steps and b_steps > 0:
        step_ratio = o_steps / b_steps
        if include_perf and step_ratio >= step_ratio_threshold and optimized_success:
            perf_reg = True
    if not perf_reg and (baseline_success and optimized_success):
        return None
    prompt = analyzer.build_prompt(task_name=task_name, baseline_history=b_txt, optimized_history=o_txt,
                                   baseline_success=baseline_success, optimized_success=optimized_success,
                                   failure_report=failure_report, baseline_env_steps=b_steps,
                                   optimized_env_steps=o_steps, step_ratio=step_ratio,
                                   performance_regression=perf_reg)
    raw = analyzer.analyze(prompt).strip()
    block = raw
    if not (block.startswith('{') and block.endswith('}')):
        m = re.search(r'\{[\s\S]*\}$', raw)
        if m:
            block = m.group(0)
    try:
        parsed = json.loads(block)
    except Exception:
        parsed = {'raw_output': raw}
    parsed.update({'task_name': task_name, 'baseline_env_steps': b_steps, 'optimized_env_steps': o_steps,
                   'step_ratio': step_ratio, 'performance_regression': perf_reg})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'prompt.txt').write_text(prompt)
    res_file.write_text(json.dumps(parsed, indent=2))
    return parsed


def run_history_analysis(cfg: Dict[str, Any]) -> Optional[Path]:
    if cfg['benchmark'] in ['swebench', 'webvoyager']:
        raise SystemExit(f'Benchmark {cfg["benchmark"]} is no longer supported')
        
    baseline_absent = False
    # ---------------- Resolve inputs ---------------- #
    if not (cfg.get('baseline_model_output_dir') or cfg.get('baseline_eval_output_file') or cfg.get('baseline_run')):
        baseline_absent = True
        baseline_model_output_dir = None  # type: ignore
        baseline_eval_output_file = None  # type: ignore
        base_ind = {}
    else:
        if not (cfg.get('baseline_model_output_dir') and cfg.get('baseline_eval_output_file')):
            if not cfg.get('baseline_run'):
                raise SystemExit('Need --baseline-run or explicit baseline paths (or omit all baseline args)')
            if cfg['benchmark'] == 'appworld':
                paths = infer_paths_appworld(cfg['baseline_run'], cfg['task_split'])
            elif cfg['benchmark'] == 'smolagents':
                paths = infer_paths_smolagents(cfg['baseline_run'], cfg['task_split'])
            elif cfg['benchmark'] == 'officebench':
                paths = infer_paths_officebench(cfg['baseline_run'], cfg['task_split'])
            else:
                raise SystemExit('Unsupported benchmark')
            baseline_model_output_dir = paths['model_output_dir']  # type: ignore
            baseline_eval_output_file = paths['eval_output_file']  # type: ignore
        else:
            baseline_model_output_dir = Path(cfg['baseline_model_output_dir']).resolve()  # type: ignore
            baseline_eval_output_file = Path(cfg['baseline_eval_output_file']).resolve()  # type: ignore

        baseline_eval = read_eval_file(baseline_eval_output_file)  # type: ignore[arg-type]
        if cfg['benchmark'] == 'officebench':
            succ = set(baseline_eval.get('successful_tasks') or [])
            failed = set(baseline_eval.get('failed_tasks') or [])
            base_ind = {t: {'success': t in succ, 'failures': None if t in succ else 'failed'} for t in (succ | failed)}
        else:
            base_ind = baseline_eval.get('individual', {})
    if not (cfg.get('optimized_model_output_dir') and cfg.get('optimized_eval_output_file')):
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
        optimized_model_output_dir = paths['model_output_dir']  # type: ignore
        optimized_eval_output_file = paths['eval_output_file']  # type: ignore
    else:
        optimized_model_output_dir = Path(cfg['optimized_model_output_dir']).resolve()  # type: ignore
        optimized_eval_output_file = Path(cfg['optimized_eval_output_file']).resolve()  # type: ignore
    
    optimized_eval = read_eval_file(optimized_eval_output_file)  # type: ignore[arg-type]
    if cfg['benchmark'] == 'officebench':
        succ = set(optimized_eval.get('successful_tasks') or [])
        failed = set(optimized_eval.get('failed_tasks') or [])
        opt_ind = {t: {'success': t in succ, 'failures': None if t in succ else 'failed'} for t in (succ | failed)}
    else:
        opt_ind = optimized_eval.get('individual', {})

    # ---------------- Detect regressions ---------------- #
    regression: Dict[str, Any] = {}
    candidate_perf: Dict[str, Any] = {}
    if baseline_absent:
        for t, opt_rec in opt_ind.items():
            if opt_rec.get('success') is False:
                regression[t] = opt_rec.get('failures')
    else:
        if cfg['benchmark'] == 'appworld':
            for t, b in base_ind.items():
                o = opt_ind.get(t)
                if not o:
                    continue
                if b.get('success') and o.get('success') is False:
                    regression[t] = o.get('failures')
                elif cfg.get('include_performance_regressions') and b.get('success') and o.get('success'):
                    candidate_perf[t] = None
        elif cfg['benchmark'] == 'smolagents':
            for t, b in base_ind.items():
                o = opt_ind.get(t)
                if not o:
                    continue
                try:
                    bf = float(b.get('f1', 0.0)); of = float(o.get('f1', 0.0))
                except Exception:
                    continue
                if bf > of and bf - of > 0.05:
                    regression[t] = o.get('failures')
                if cfg.get('include_performance_regressions'):
                    candidate_perf[t] = None
        elif cfg['benchmark'] == 'officebench':
            for t, b in base_ind.items():
                o = opt_ind.get(t)
                if not o:
                    continue
                if b.get('success') and o.get('success') is False:
                    regression[t] = o.get('failures')
                elif cfg.get('include_performance_regressions') and b.get('success') and o.get('success'):
                    candidate_perf[t] = None
        else:
            for t, b in base_ind.items():
                o = opt_ind.get(t)
                if not o:
                    continue
                if b.get('success') and not o.get('success'):
                    regression[t] = o.get('failures')
                elif cfg.get('include_performance_regressions') and b.get('success') and o.get('success'):
                    candidate_perf[t] = None

    if not regression and not candidate_perf:
        fallback = {t: o.get('failures') for t, o in opt_ind.items() if o.get('success') is False}
        if fallback:
            logging.info(f'No direct regressions; using all optimized failures ({len(fallback)}).')
            regression = fallback
        if not regression and not candidate_perf:
            logging.info('No history regression candidates.')
            return None

    # ---------------- Map task directories ---------------- #
    if cfg['benchmark'] in ('smolagents'):
        base_tasks_dir = None if baseline_absent else baseline_model_output_dir / 'samples'  # type: ignore
        opt_tasks_dir = optimized_model_output_dir / 'samples'  # type: ignore
    elif cfg['benchmark'] == 'officebench':
        base_tasks_dir = None if baseline_absent else baseline_model_output_dir  # type: ignore
        opt_tasks_dir = optimized_model_output_dir  # type: ignore
    else:  # appworld
        base_tasks_dir = None if baseline_absent else baseline_model_output_dir  # type: ignore
        opt_tasks_dir = optimized_model_output_dir  # type: ignore
    if not opt_tasks_dir.exists():
        logging.error('Optimized task directory missing.')
        return None
    if base_tasks_dir is not None and not base_tasks_dir.exists():  # type: ignore
        logging.warning('Baseline task directory missing; proceeding optimized-only.')
        base_tasks_dir = None  # type: ignore
    if cfg['benchmark'] in ('smolagents'):
        opt_dirs = {d.name: d for d in opt_tasks_dir.iterdir() if d.is_dir()}
        base_dirs = {d.name: d for d in (base_tasks_dir.iterdir() if base_tasks_dir else []) if d.is_dir()} if base_tasks_dir else {n: p for n, p in opt_dirs.items()}
    else:
        opt_dirs = {d.name.lstrip('task_'): d for d in opt_tasks_dir.iterdir() if d.is_dir()}
        if base_tasks_dir:
            base_dirs = {d.name.lstrip('task_'): d for d in base_tasks_dir.iterdir() if d.is_dir()}
        else:
            base_dirs = {n: p for n, p in opt_dirs.items()}
    if cfg['benchmark'] == 'officebench':
        # Remove leftover breakpoint and expand each task directory to subtask-level keys "task_subtask"
        # Original opt_dirs/base_dirs have mapping task_id -> task_dir
        expanded_opt: Dict[str, Path] = {}
        for tid, tdir in list(opt_dirs.items()):
            # subtask folders are immediate numeric children (e.g., 0,1,...)
            if not tdir.is_dir():
                continue
            subs = [d for d in tdir.iterdir() if d.is_dir() and d.name.isdigit()]
            if not subs:
                # fallback: if no numeric child, keep original
                expanded_opt[tid + '_0'] = tdir
                continue
            for sd in subs:
                expanded_opt[f'{tid}_{sd.name}'] = sd
        opt_dirs = expanded_opt
        if base_tasks_dir:
            expanded_base: Dict[str, Path] = {}
            for tid, tdir in list(base_dirs.items()):  # type: ignore
                if not tdir.is_dir():
                    continue
                subs = [d for d in tdir.iterdir() if d.is_dir() and d.name.isdigit()]
                if not subs:
                    expanded_base[tid + '_0'] = tdir
                    continue
                for sd in subs:
                    expanded_base[f'{tid}_{sd.name}'] = sd
            base_dirs = expanded_base  # type: ignore
        else:
            base_dirs = {k: v for k, v in opt_dirs.items()}

    selected = sorted([t for t in set(list(regression.keys()) + list(candidate_perf.keys())) if t in opt_dirs and t in base_dirs])
    if cfg.get('max_tasks'):
        selected = selected[: int(cfg['max_tasks'])]
    if not selected:
        logging.warning('No matching task dirs.')
        return None
    for n in selected:
        print(n)
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'regression_results').mkdir(exist_ok=True)
    run_cfg = dict(cfg)
    run_cfg.update({
        'baseline_model_output_dir_resolved': None if baseline_absent else (str(base_tasks_dir) if base_tasks_dir else None),
        'baseline_eval_output_file_resolved': None if baseline_absent else (str(baseline_eval_output_file) if not baseline_absent else None),
        'optimized_model_output_dir_resolved': str(opt_tasks_dir),
        'optimized_eval_output_file_resolved': str(optimized_eval_output_file),
        'baseline_absent': baseline_absent,
    })
    serializable_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in run_cfg.items()}
    (out / 'analysis_run_config.json').write_text(json.dumps(serializable_cfg, indent=2))

    analyzer = HistoryRegressionAnalyzer(
        model_name=cfg['analysis_model'], 
        prompt_template=cfg['analysis_prompt_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    aggregated: List[Dict[str, Any]] = []
    for t in tqdm(selected, desc='History analysis', unit='task'):
        base_rec = base_ind.get(t, {})
        opt_rec = opt_ind.get(t, {})
        if baseline_absent:
            o_hist = opt_dirs[t] / 'llm_history.json'
            try:
                o_raw = json.loads(o_hist.read_text()) if o_hist.exists() else None
            except Exception:
                o_raw = None
            o_txt = extract_histories([], o_raw)[1] if o_raw else ''
            prompt = analyzer.build_prompt(task_name=t, baseline_history='', optimized_history=o_txt,
                                           baseline_success=False, optimized_success=opt_rec.get('success', False),
                                           failure_report=regression.get(t), baseline_env_steps=None,
                                           optimized_env_steps=load_env_steps(opt_dirs[t]), step_ratio=None,
                                           performance_regression=False)
            raw = analyzer.analyze(prompt).strip()
            block = raw if (raw.startswith('{') and raw.endswith('}')) else raw
            try:
                parsed = json.loads(block)
            except Exception:
                parsed = {'raw_output': raw}
            parsed.update({'task_name': t, 'baseline_env_steps': None, 'optimized_env_steps': load_env_steps(opt_dirs[t]), 'step_ratio': None, 'performance_regression': False, 'baseline_absent': True})
            task_out = out / 'regression_results' / t
            task_out.mkdir(parents=True, exist_ok=True)
            (task_out / 'prompt.txt').write_text(prompt)
            (task_out / 'history_regression_analysis.json').write_text(json.dumps(parsed, indent=2))
            aggregated.append(parsed)
        else:
            r = process_history_task(
                task_name=t,
                baseline_task_dir=base_dirs[t],
                optimized_task_dir=opt_dirs[t],
                failure_report=regression.get(t),
                analyzer=analyzer,
                out_dir=out / 'regression_results' / t,
                skip_existing=cfg.get('skip_existing_analysis', False),
                baseline_success=base_rec.get('success', False),
                optimized_success=opt_rec.get('success', False),
                include_perf=cfg.get('include_performance_regressions', False),
                step_ratio_threshold=float(cfg.get('step_ratio_threshold', 1.5)),
            )
            if r:
                aggregated.append(r)
    if not aggregated:
        logging.warning('No analyses produced.')
        return None
    agg_path = out / 'aggregated_history_regressions.json'
    agg_path.write_text(json.dumps(aggregated, indent=2))
    logging.info(f'Wrote {agg_path}')
    return agg_path


# ---------------- Update helpers ---------------- #


class HistoryPromptOptimizer(JinjaLLMTemplate):
    def __init__(self, model_name: str, template_path: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=template_path, llm_backend=llm_backend)

    def build_prompt(self, **kwargs) -> str:  # type: ignore[override]
        return self.render(**kwargs)

    def forward(self, prompt: str) -> str:
        return self.generate(prompt)


def run_history_update(cfg: Dict[str, Any], agg_path: Path) -> None:
    if not cfg.get('base_prompt_template'):
        raise SystemExit('Need --base-prompt-template for update phase')
    base_template = Path(cfg['base_prompt_template']).resolve()
    if not base_template.exists():
        raise SystemExit(f'Base prompt template missing: {base_template}')
    entries = json.loads(agg_path.read_text())
    if not isinstance(entries, list) or not entries:
        logging.error('Aggregated history regressions file empty or invalid')
        return
    random.seed(int(cfg.get('seed', 42)))
    original_prompt = base_template.read_text()
    out_dir = Path(cfg['output_dir'])
    opt_dir = out_dir / 'optimized_prompts'
    opt_dir.mkdir(exist_ok=True)
    copy_system_prompts_from_source(base_template, opt_dir)
    optimizer = HistoryPromptOptimizer(
        model_name=cfg['update_model'], 
        template_path=cfg['optimizer_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    num_prompts = int(cfg['num_prompts'])
    spp = int(cfg['samples_per_prompt'])
    max_prob = cfg.get('max_problematic') or cfg.get('max_problEMatic')  # tolerate typos
    if max_prob is None:
        max_prob = cfg.get('max_problEmatic') or cfg.get('max_problematic')
    max_missing = cfg.get('max_missing')
    max_suggestions = cfg.get('max_suggestions')
    manifest: List[Dict[str, Any]] = []
    for i in range(num_prompts):
        k = min(spp, len(entries))
        selected = random.sample(entries, k) if k < len(entries) else list(entries)
        payload: List[Dict[str, Any]] = []
        for r in selected:
            rem: List[str] = []
            rr = r.get('remediation_recommendations')
            if isinstance(rr, list):
                rem = [str(x).strip() for x in rr if isinstance(x, str)]
            ineff: List[str] = []
            if isinstance(r.get('inefficiency_patterns'), list):
                for it in r['inefficiency_patterns']:
                    if isinstance(it, dict):
                        patt = it.get('pattern')
                        if isinstance(patt, str) and patt:
                            ineff.append(patt.strip())
                    elif isinstance(it, str):
                        ineff.append(it.strip())
            elif isinstance(r.get('inefficiency_evidence'), list):
                for it in r['inefficiency_evidence']:
                    if isinstance(it, dict):
                        for key in ('pattern', 'cause'):
                            val = it.get(key)
                            if isinstance(val, str) and val:
                                ineff.append(val.strip())
                                break
            if isinstance(max_prob, int) and max_prob > 0:
                ineff = ineff[:max_prob]
            missing: List[str] = []
            distorted: List[str] = []
            md = r.get('missing_or_distorted_facts')
            if isinstance(md, list):
                for it in md:
                    if isinstance(it, dict):
                        b = it.get('baseline')
                        o = it.get('optimized_context_absent_or_changed')
                        if isinstance(b, str) and b:
                            missing.append(b.strip())
                        if isinstance(o, str) and o:
                            distorted.append(o.strip())
            if isinstance(max_missing, int) and max_missing > 0:
                missing = missing[:max_missing]
            summary_inacc: List[str] = []
            si_old = r.get('summary_inaccuracies')
            if isinstance(si_old, list):
                for it in si_old:
                    if isinstance(it, dict):
                        ex = it.get('summary_excerpt')
                        if isinstance(ex, str) and ex:
                            summary_inacc.append(ex.strip())
            elif isinstance(r.get('summary_or_session_contributors'), list):
                for it in r['summary_or_session_contributors']:
                    if isinstance(it, dict):
                        ex = it.get('summary_excerpt') or it.get('issue')
                        if isinstance(ex, str) and ex:
                            summary_inacc.append(ex.strip())
            lost_vars: List[str] = []
            lv_old = r.get('lost_state_variables')
            if isinstance(lv_old, list):
                for it in lv_old:
                    if isinstance(it, dict):
                        nm = it.get('name_or_pattern')
                        if isinstance(nm, str) and nm:
                            lost_vars.append(nm.strip())
            elif isinstance(r.get('lost_or_recreated_state'), list):
                for it in r['lost_or_recreated_state']:
                    if isinstance(it, dict):
                        st = it.get('state_item')
                        if isinstance(st, str) and st:
                            lost_vars.append(st.strip())
            cats: List[str] = []
            rc = r.get('root_cause_categories') or r.get('primary_inflation_causes')
            if isinstance(rc, list):
                for c in rc:
                    if isinstance(c, str):
                        cats.append(c)
            if r.get('performance_regression'):
                cats.append('Performance Regression')
            if 'step_ratio' in r and r.get('step_ratio') is not None:
                cats.append('Efficiency Regression')
            if isinstance(max_suggestions, int) and max_suggestions > 0:
                rem = rem[:max_suggestions]
            payload.append({
                'remediation': dedupe_list(rem) if cfg.get('dedupe') else rem,
                'inefficiency_patterns': ineff,
                'missing_facts': missing,
                'distorted_facts': distorted,
                'summary_inaccuracies': summary_inacc,
                'lost_vars': lost_vars,
                'issue_categories': cats,
            })
        prompt_text = optimizer.build_prompt(original_prompt=original_prompt, samples=payload)
        (opt_dir / f'optimizer_prompt_input_{i}.txt').write_text(prompt_text)
        (opt_dir / f'samples_payload_{i}.json').write_text(json.dumps(payload, indent=2))
        improved = optimizer.forward(prompt_text)
        out_file = opt_dir / f'improved_history_prompt_samples_{i}.jinja'
        out_file.write_text(improved)
        cat_counts: Dict[str, int] = {}
        for s in payload:
            for c in s.get('issue_categories', []) or []:
                if isinstance(c, str):
                    cat_counts[c] = cat_counts.get(c, 0) + 1
        manifest.append({
            'index': i,
            'output_file': str(out_file.relative_to(out_dir)),
            'n_samples_used': len(payload),
            'issue_categories_top3': sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3],
        })
        logging.info(f'Generated improved history prompt {i}: {out_file}')
    (opt_dir / 'manifest_by_samples.json').write_text(json.dumps(manifest, indent=2))
    logging.info('History prompt update complete')


# ---------------- Main ---------------- #


def main():  # pragma: no cover
    args = setup_args().parse_args()
    cfg = load_config_file_and_merge(args)
    
    logging.basicConfig(level=getattr(logging, cfg.get('log_level', 'INFO').upper()),
                        format='%(asctime)s %(levelname)s %(message)s')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    agg_path: Optional[Path] = None
    if cfg['phase'] in ('both', 'analysis'):
        agg_path = run_history_analysis(cfg)
    if cfg['phase'] == 'update' and not cfg.get('aggregated_history_regressions') and not agg_path:
        raise SystemExit('Provide --aggregated-history-regressions for update-only phase')
    if cfg['phase'] in ('both', 'update'):
        if not agg_path:
            agg_path = Path(cfg['aggregated_history_regressions']).resolve()
        if not agg_path.exists():
            raise SystemExit(f'Aggregated history regressions file missing: {agg_path}')
        run_history_update(cfg, agg_path)


if __name__ == '__main__':
    main()
