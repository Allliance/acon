#!/usr/bin/env python3
"""Unified History Context Reduction -> Prompt Update Pipeline

Phases:
  analysis : Run history context reduction on a history-optimized run generating per-session analyses.
  update   : Sample per-session analyses to create length-optimized history prompt templates.
  both     : Do analysis then update (default).

Update phase expects the directory containing reduction_results (analysis-dir). When running both, it reuses --output-dir.
For update-only you must supply --analysis-dir explicitly.
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
try:  # optional
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **k): return x

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from productive_agents.llm import AzureOpenAIServerModel, ChatGPT  # noqa: E402
from common.paths import read_eval_file, infer_paths_appworld, infer_paths_smolagents  # type: ignore
from common.llm import JinjaLLMTemplate  # type: ignore
from common.config import load_config_file_and_merge  # type: ignore


# ---------------- Args ---------------- #

def setup_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Unified History Context Reduction + Prompt Update')
    p.add_argument('--phase', choices=['both', 'analysis', 'update'], default='both')
    # Analysis args (from main_history_context_reduction.py)
    p.add_argument('--optimized-run', type=str)
    p.add_argument('--optimized-model-output-dir', type=Path)
    p.add_argument('--optimized-eval-output-file', type=Path)
    p.add_argument('--task-split', type=str, default='train')
    p.add_argument('--benchmark', type=str, choices=['appworld', 'smolagents'], default='appworld')
    p.add_argument('--analysis-model', type=str, default='o3')
    default_analysis_prompt = Path(__file__).parent / 'prompts' / 'history_context_reducer_analysis_prompt.jinja'
    p.add_argument('--analysis-prompt-template', type=Path, default=default_analysis_prompt)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--skip-existing-analysis', action='store_true')
    p.add_argument('--min-keep-ratio', type=float, default=0.35)
    p.add_argument('--smolagents-min-f1', type=float, default=0.6)
    # Update args (from main_history_context_prompt_update.py)
    p.add_argument('--analysis-dir', type=Path, help='Directory containing reduction_results for update-only phase')
    p.add_argument('--base-prompt-template', type=Path, help='Original history prompt template (jinja)')
    p.add_argument('--optimizer-template', type=Path, default=Path('prompts/history_context_prompt_optimizer_prompt_by_samples.jinja'))
    p.add_argument('--update-model', type=str, default='o3')
    p.add_argument('--num-prompts', type=int, default=5)
    p.add_argument('--samples-per-prompt', type=int, default=30)
    p.add_argument('--suggestions-per-prompt', type=int, default=40)
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


# ---------------- Analysis (history context reduction) ---------------- #

class ContextReducer(JinjaLLMTemplate):
    def __init__(self, model_name: str, prompt_template: Path, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=prompt_template, llm_backend=llm_backend)
    def build_prompt(self, **kw) -> str:  # type: ignore[override]
        return self.render(**kw)
    def reduce(self, prompt: str) -> Optional[str]:
        try:
            return self.generate(prompt)
        except Exception:  # pragma: no cover
            return None


def approx_token_count(text: str) -> int:
    return max(1, int(len(text) / 4))


def process_history_task(task_name: str, optimized_task_dir: Path, reducer: ContextReducer, output_task_dir: Path,
                         skip_existing: bool) -> Optional[List[Dict[str, Any]]]:
    hist_path = optimized_task_dir / 'llm_history.json'
    opt_hist_path = optimized_task_dir / 'history_optimizer_history.json'
    if not hist_path.exists() or not opt_hist_path.exists():
        return None
    try:
        optimized_history_raw = json.loads(hist_path.read_text())
        optimizer_history = json.loads(opt_hist_path.read_text())
    except Exception as e:
        logging.warning(f'History parse fail {task_name}: {e}')
        return None
    if not (isinstance(optimized_history_raw, list) and optimized_history_raw and all(isinstance(s, list) for s in optimized_history_raw) and len(optimized_history_raw) > 1):
        return None
    num_sessions = len(optimized_history_raw)
    max_align = min(len(optimizer_history) if isinstance(optimizer_history, list) else 0, num_sessions - 1)
    if max_align <= 0:
        return None
    metas: List[Dict[str, Any]] = []
    for idx in range(max_align):
        session_num = idx + 2
        session_dir = output_task_dir / f'session_{session_num}'
        res_file = session_dir / 'analysis.json'
        if skip_existing and res_file.exists():
            continue
        item = optimizer_history[idx]
        try:
            response_text = item[2]
            prompt_text = "### TASK: \n" + item[3]["task"] + "\n\n### PREVIOUS SUMMARY:\n" + item[3]["prev_summary"] + "\n\n### HISTORY:\n" + item[3]["history"]
        except Exception:
            continue
        def render_flat(session_any: Any) -> str:
            if not session_any: return ''
            lines: List[str] = []
            skipped_first_user = False
            for msg in session_any:
                if not isinstance(msg, (dict, list, tuple)): continue
                if isinstance(msg, dict):
                    role = msg.get('role', 'UNKNOWN'); content = msg.get('content', '')
                else:
                    role, content = msg[0], msg[1]
                if not isinstance(role, str): role = str(role)
                rl = role.lower()
                if rl == 'system': continue
                if rl == 'user' and not skipped_first_user:
                    skipped_first_user = True
                    continue
                text = ''
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, str): parts.append(c)
                        elif isinstance(c, dict):
                            for k in ('text', 'value', 'content'):
                                v = c.get(k) if isinstance(c, dict) else None
                                if isinstance(v, str): parts.append(v); break
                        else: parts.append(str(c))
                    text = '\n'.join(p for p in parts if p)
                elif isinstance(content, dict):
                    for k in ('text', 'value', 'content', 'message'):
                        v = content.get(k) if isinstance(content, dict) else None
                        if isinstance(v, str): text = v; break
                lines.append(f"{role.upper()}:\n{text}\n")
            return '\n'.join(lines).strip()
        next_session_text = ''
        if (idx + 1) < len(optimized_history_raw):
            next_session_text = render_flat(optimized_history_raw[idx + 1])
        aligned_text = f"--- OPT PASS {idx + 1} ---\nINPUT_PROMPT:\n{prompt_text}\n\nOUTPUT_SUMMARY:\n{response_text}\n"
        orig_chars = len(response_text)
        prompt = reducer.build_prompt(task_name=task_name, optimized_history=aligned_text, next_session=next_session_text, orig_chars=orig_chars)
        raw = reducer.reduce(prompt)
        if raw is None: continue
        raw = raw.strip()
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / 'prompt.txt').write_text(prompt)
        m = re.search(r"<<<ANALYSIS_JSON>>>\n([\s\S]*?)\n<<<END>>>", raw)
        meta = {'task_name': task_name, 'session': session_num, 'orig_chars': orig_chars, 'orig_tokens_approx': approx_token_count(response_text)}
        if m:
            block = m.group(1).strip()
            try:
                js = json.loads(block)
            except Exception:
                m2 = re.search(r"\{[\s\S]*\}", raw)
                if m2:
                    try: js = json.loads(m2.group(0))
                    except Exception: js = {'raw': raw}
                else:
                    js = {'raw': raw}
            (session_dir / 'analysis.json').write_text(json.dumps(js, indent=2))
            (session_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
            metas.append(meta)
        else:
            (session_dir / 'analysis.raw.txt').write_text(raw)
            (session_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    return metas or None


def run_history_analysis(cfg: Dict[str, Any]) -> Optional[Path]:
    if not cfg.get('optimized_model_output_dir') or not cfg.get('optimized_eval_output_file'):
        if not cfg.get('optimized_run'):
            raise SystemExit('Need --optimized-run or explicit optimized paths')
        paths = infer_paths_appworld(cfg['optimized_run'], cfg['task_split']) if cfg['benchmark'] == 'appworld' else infer_paths_smolagents(cfg['optimized_run'], cfg['task_split'])
        optimized_model_output_dir = paths['model_output_dir']
        optimized_eval_output_file = paths['eval_output_file']
    else:
        optimized_model_output_dir = Path(cfg['optimized_model_output_dir']).resolve()
        optimized_eval_output_file = Path(cfg['optimized_eval_output_file']).resolve()
    eval_data = read_eval_file(optimized_eval_output_file)
    individual = eval_data.get('individual', {})
    if cfg['benchmark'] in ('smolagents'):
        thr = float(cfg.get('smolagents_min_f1', 0.6))
        task_names: List[str] = []
        for t, rec in individual.items():
            if t not in all_task_dirs: continue
            try:
                if float(rec.get('f1', 0.0)) >= thr: task_names.append(t)
            except Exception: continue
    else:
        tasks_dir = optimized_model_output_dir
        all_task_dirs = {d.name.lstrip('task_'): d for d in tasks_dir.iterdir() if d.is_dir()} if tasks_dir.exists() else {}
        task_names = [t for t, rec in individual.items() if rec.get('success') is True and t in all_task_dirs]
    task_names.sort()
    if cfg.get('max_tasks'): task_names = task_names[: int(cfg['max_tasks'])]
    if not task_names:
        logging.info('No successful tasks for history analysis.')
        return None
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'reduction_results').mkdir(exist_ok=True)
    reducer = ContextReducer(
        model_name=cfg['analysis_model'], 
        prompt_template=cfg['analysis_prompt_template'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    aggregated: List[Dict[str, Any]] = []
    for t in tqdm(task_names, desc='History ctx analysis', unit='task'):
        meta = process_history_task(t, all_task_dirs[t], reducer, out / 'reduction_results' / t, cfg.get('skip_existing_analysis', False))
        if meta:
            aggregated.extend(meta)
    if not aggregated:
        logging.warning('No history context analyses produced (will still proceed to update phase).')
        return out
    agg_path = out / 'aggregated_analyses.json'
    agg_path.write_text(json.dumps(aggregated, indent=2))
    logging.info(f'Wrote {agg_path}')
    return out  # analysis directory


# ---------------- Update (prompt generation) ---------------- #

def dedupe_list(values: List[str]) -> List[str]:
    seen = set(); out: List[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out


class HistoryContextPromptOptimizer(JinjaLLMTemplate):
    def __init__(self, model_name: str, llm_backend: str = 'azure'):
        super().__init__(model_name=model_name, template_path=None, llm_backend=llm_backend)

    def forward(self, prompt: str) -> str:
        return self.generate(prompt)


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"<<<ANALYSIS_JSON>>>\n([\s\S]*?)\n<<<END>>>", text)
    if m:
        block = m.group(1).strip()
        try: return json.loads(block)
        except Exception: pass
    # fallback scan
    candidates: List[str] = []
    stack = []; start = None
    for i,ch in enumerate(text):
        if ch == '{':
            stack.append('{'); start = i if start is None else start
        elif ch == '}':
            if stack: stack.pop();
            if not stack and start is not None:
                candidates.append(text[start:i+1]); start=None
    for c in sorted(candidates, key=len, reverse=True):
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and any(k in obj for k in ('remove_or_shorten','summarize_rules','critical_refs','keep','overview')):
                return obj
        except Exception: continue
    return None


def find_analysis_files(root: Path) -> List[Path]:
    results: List[Path] = []
    for p in root.rglob('reduction_results/*/session_*/analysis.json'):
        if p.is_file(): results.append(p)
    for p in root.rglob('reduction_results/*/session_*/analysis.raw.txt'):
        if p.is_file(): results.append(p)
    return sorted(results)


def load_analysis(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.suffix == '.json':
            return json.loads(path.read_text())
        raw = path.read_text(); return _extract_json_from_text(raw)
    except Exception:
        return None


def run_history_update(cfg: Dict[str, Any], analysis_dir: Path) -> None:
    base_template = Path(cfg['base_prompt_template']).resolve()
    if not base_template.exists():
        raise SystemExit(f'Base prompt template missing: {base_template}')
    analysis_files = find_analysis_files(analysis_dir)
    if not analysis_files:
        logging.warning('No analysis files for update.')
        return
    random.seed(int(cfg.get('seed', 42)))
    original_prompt = base_template.read_text()
    out_dir = Path(cfg['output_dir'])
    opt_dir = out_dir / 'optimized_prompts'
    opt_dir.mkdir(exist_ok=True)
    optimizer = HistoryContextPromptOptimizer(
        model_name=cfg['update_model'],
        llm_backend=cfg.get('llm_backend', 'azure')
    )
    num_prompts = int(cfg['num_prompts'])
    spp = int(cfg['samples_per_prompt'])
    suggestions_cap = int(cfg.get('suggestions_per_prompt') or 0)
    manifest: List[Dict[str, Any]] = []
    template_path = cfg['optimizer_template']
    for i in range(num_prompts):
        k = min(spp, len(analysis_files))
        selected = random.sample(analysis_files, k) if k < len(analysis_files) else list(analysis_files)
        samples_payload: List[Dict[str, Any]] = []
        cat_counts: Dict[str, int] = {}
        total_chars = 0; n_chars = 0
        for f in selected:
            data = load_analysis(f)
            if not data: continue
            overview = data.get('overview') or {}
            if isinstance(overview, dict) and 'orig_chars' in overview:
                try: total_chars += int(overview.get('orig_chars')); n_chars += 1
                except Exception: pass
            removes = [r for r in data.get('remove_or_shorten', []) if isinstance(r, dict)]
            keeps = [r for r in data.get('keep', []) if isinstance(r, dict)]
            rules = [r for r in data.get('summarize_rules', []) if isinstance(r, str)]
            for r in removes:
                c = r.get('category')
                if isinstance(c, str): cat_counts[c] = cat_counts.get(c, 0) + 1
            # derive labels
            parts = f.parts
            task_label = 'unknown'; session_label = None
            if 'reduction_results' in parts:
                idx = parts.index('reduction_results')
                if idx + 1 < len(parts): task_label = parts[idx+1]
                if idx + 2 < len(parts) and parts[idx+2].startswith('session_'): session_label = parts[idx+2]
            if cfg.get('dedupe'):
                rules = dedupe_list(rules)
            if suggestions_cap > 0:
                rules = rules[:suggestions_cap]
            samples_payload.append({'task_label': task_label, 'session': session_label, 'overview': overview, 'removals': removes, 'keeps': keeps, 'rules': rules})
        avg_chars = int(total_chars / n_chars) if n_chars else None
        tmpl = jinja2.Template(Path(template_path).read_text())
        prompt_text = tmpl.render(original_prompt=original_prompt, avg_orig_chars=avg_chars, samples=samples_payload)
        (opt_dir / f'optimizer_prompt_input_{i}.txt').write_text(prompt_text)
        improved = optimizer.forward(prompt_text)
        out_file = opt_dir / f'length_optimized_history_prompt_{i}.jinja'
        out_file.write_text(improved)
        manifest.append({
            'index': i,
            'output_file': str(out_file.relative_to(out_dir)),
            'n_samples_used': len(samples_payload),
            'avg_orig_chars': avg_chars,
            'category_counts_top3': sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3],
        })
        logging.info(f'Generated history length-optimized prompt {i}: {out_file}')
    (opt_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    logging.info('History context prompt update complete')


# ---------------- Main ---------------- #

def main():  # pragma: no cover
    args = setup_args().parse_args()
    cfg = load_config_file_and_merge(args)
    
    logging.basicConfig(level=getattr(logging, cfg.get('log_level', 'INFO').upper()), format='%(asctime)s %(levelname)s %(message)s')
    Path(cfg['output_dir']).mkdir(parents=True, exist_ok=True)
    analysis_dir: Optional[Path] = None
    if cfg['phase'] in ('both', 'analysis'):
        analysis_dir = run_history_analysis(cfg)
    if cfg['phase'] == 'update' and not cfg.get('analysis_dir') and not analysis_dir:
        raise SystemExit('Provide --analysis-dir for update-only phase')
    if cfg['phase'] in ('both', 'update'):
        if not analysis_dir:
            provided = cfg.get('analysis_dir')
            if not provided:
                raise SystemExit('No analysis results available and no --analysis-dir supplied')
            analysis_dir = Path(provided).resolve()
        if not analysis_dir.exists():
            raise SystemExit(f'Analysis directory missing: {analysis_dir}')
        if not cfg.get('base_prompt_template'):
            raise SystemExit('Need --base-prompt-template for update phase')
        run_history_update(cfg, analysis_dir)


if __name__ == '__main__':
    main()
