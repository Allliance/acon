#!/usr/bin/env python3
"""
OfficeBench context-optimization pipeline.

Generates context optimization YAML configs from *.jinja templates in a prompts
directory (typically experiments/officebench/prompts/context_opt) and runs
OfficeBench experiments (run_all.py) once per generated config, collecting and
comparing success_rate from each experiment_summary.json.

Example:
  python experiments/officebench/run_ctxopt_pipeline.py \
    --prompts-dir experiments/officebench/prompts/context_opt \
    --model-name gpt-4.1 \
    --split train \
    --ctxopt-type history \
    --tag 250901_ctxopt_hist
"""
import argparse
import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
from typing import List, Dict, Optional

try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None
    HAS_YAML = False

try:
    from rich.console import Console  # type: ignore
    from rich.table import Table  # type: ignore
    RICH_OK = True
except Exception:  # pragma: no cover
    Console = None
    Table = None
    RICH_OK = False


def generate_history_cfg(template_name: str, model: str, threshold: int, preserve_last_k: Optional[int], prompts_dir: Path, opt_version: int) -> dict:
    cfg = {
        "type": "history",
        "model": model,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_history_user": template_name,
        },
        "history_summarization_threshold": threshold,
        "history_prompt_dir": str(prompts_dir),
        "history_version": opt_version,
    }
    if preserve_last_k is not None and preserve_last_k >= 0:
        cfg["preserve_last_k_turns"] = int(preserve_last_k)
    return cfg


def generate_obs_cfg(template_name: str, model: str, threshold: int, prompts_dir: Path, opt_version: int) -> dict:
    return {
        "type": "obs",
        "model": model,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_user": template_name,
        },
        "obs_summarization_threshold": threshold,
        "obs_prompt_dir": str(prompts_dir),
        "obs_version": opt_version,
    }


def discover_prompt_files(prompts_dir: Path, ctxopt_type: str) -> List[Path]:
    all_jinja = sorted([p for p in prompts_dir.glob('*.jinja') if p.name not in {"system_prompt.jinja", "manifest.json"}])
    if ctxopt_type == 'history':
        candidates = [p for p in all_jinja if 'prompt_history' in p.stem]
    else:
        candidates = [p for p in all_jinja if p.stem == 'prompt_user']
    return candidates or all_jinja


def write_yaml(path: Path, cfg: dict):
    if HAS_YAML:
        with open(path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    else:  # minimal manual serialization
        def emit(obj, indent=0):
            lines = []
            sp = '  ' * indent
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{sp}{k}:")
                        lines.extend(emit(v, indent + 1))
                    else:
                        lines.append(f"{sp}{k}: {json.dumps(v)}")
            elif isinstance(obj, list):
                for it in obj:
                    if isinstance(it, (dict, list)):
                        lines.append(f"{sp}-")
                        lines.extend(emit(it, indent + 1))
                    else:
                        lines.append(f"{sp}- {json.dumps(it)}")
            return lines
        path.write_text('\n'.join(emit(cfg)) + '\n')


def main():  # noqa: C901 (simple enough despite length)
    parser = argparse.ArgumentParser(description="Generate ctxopt YAMLs for OfficeBench and run run_all.py per config.")
    parser.add_argument('--ctxopt-type', choices=['history', 'obs'], default='history')
    parser.add_argument('--prompts-dir', required=True)
    parser.add_argument('--model-name', default='gpt-4.1')
    parser.add_argument('--split', default='train', help='train/test split file consumed by run_all.py')
    parser.add_argument('--tag', default=None, help='Base experiment tag; variant suffix appended')
    parser.add_argument('--config-out-dir', default=None, help='Directory to write generated YAMLs (default auto)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Pass --debug to run_all.py')
    parser.add_argument('--task', default=None, help='Optional single task id to pass to run_all.py')
    parser.add_argument('--history-threshold', type=int, default=4096)
    parser.add_argument('--obs-threshold', type=int, default=1024)
    parser.add_argument('--preserve-last-k-turns', type=int, default=1)
    parser.add_argument('--limit-configs', type=int, default=None, help='Optional cap on number of configs to execute')
    parser.add_argument('--opt-version', type=int, default=1, help='Optional optimization version tag appended to folder')
    parser.add_argument('--run-eval', action='store_true', help='Run evaluation.main after each config to compute success metrics')
    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir).resolve()
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompt_files = discover_prompt_files(prompts_dir, args.ctxopt_type)
    if not prompt_files:
        raise RuntimeError(f"No prompt templates found in {prompts_dir}")

    ts = datetime.now().strftime('%y%m%d')
    base_folder = f"{ts}_{args.model_name}_{args.ctxopt_type}_ctxopt"
    version_pattern = r'(?:^|_)(v\d+)(?:$|_)'
    version_tag = None
    for name in (prompts_dir.name, prompts_dir.parent.name):
        m = re.search(version_pattern, name, flags=re.IGNORECASE)
        if m:
            version_tag = m.group(1)
            break
    if version_tag:
        base_folder += f"_{version_tag}"
    if args.opt_version != 1:
        base_folder += f"_optv{args.opt_version}"

    if args.config_out_dir:
        config_out_dir = Path(args.config_out_dir).resolve()
    else:
        config_out_dir = Path(__file__).resolve().parent / 'configs' / 'context_opt' / base_folder
    config_out_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for pf in prompt_files:
        stem = pf.stem
        if args.ctxopt_type == 'history':
            cfg = generate_history_cfg(stem, args.model_name, args.history_threshold, args.preserve_last_k_turns, prompts_dir, args.opt_version)
        else:
            cfg = generate_obs_cfg(stem, args.model_name, args.obs_threshold, prompts_dir, args.opt_version)
        out_path = config_out_dir / f'{stem}.yaml'
        write_yaml(out_path, cfg)
        generated.append(out_path)

    print(f"Generated {len(generated)} ctxopt configs under: {config_out_dir}")
    for p in generated:
        print(f" - {p}")

    if args.dry_run:
        print('Dry-run: skipping execution.')
        return

    run_all_script = Path(__file__).resolve().parent / 'run_all.py'
    if not run_all_script.exists():
        raise FileNotFoundError(f"run_all.py not found at {run_all_script}")

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / 'src')
    env['PYTHONPATH'] = f"{src_path}:{env.get('PYTHONPATH','')}" if env.get('PYTHONPATH') else src_path

    base_tag = args.tag or (prompts_dir.parent.name)
    if version_tag and (version_tag not in base_tag):
        base_tag = f"{base_tag}_{version_tag}"

    summaries: List[Dict[str, Optional[float]]] = []
    exec_configs = generated[: args.limit_configs] if args.limit_configs else generated

    for cfg_path in exec_configs:
        variant = cfg_path.stem
        per_tag = f"{base_tag}_{variant}" if variant not in base_tag else base_tag
        print('\n========================================')
        print(f"Running OfficeBench for config: {cfg_path.name} | tag: {per_tag}")
        print('========================================')
        cmd = [
            sys.executable,
            str(run_all_script),
            '--split', args.split,
            '--model_name', args.model_name,
            '--tag', per_tag,
            '--co_config_path', str(cfg_path),
        ]
        if args.task:
            cmd.extend(['--task', args.task])
        if args.debug:
            cmd.append('--debug')
        print('Command:', ' '.join(cmd))
        proc = subprocess.run(cmd, env=env, cwd=str(run_all_script.parent))
        if proc.returncode != 0:
            print(f"Run failed (exit {proc.returncode}) for {cfg_path}")
            continue
        # Load quick run_all summary success_rate
        success_rate = 0.0
        summary = {}
        try:
            model_part = args.model_name.replace('/', '_')
            summary_path = run_all_script.parent / 'outputs' / f"{model_part}_{per_tag}" / args.split / 'experiment_summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            success_rate = float(summary.get('success_rate', 0.0))
        except Exception as e:  # pragma: no cover
            print(f"Warning: failed to load experiment_summary for {cfg_path}: {e}")

        eval_success = None
        if args.run_eval:
            print('Running evaluation.main for this config...')
            eval_cmd = [
                sys.executable,
                '-m', 'evaluation.main',
                '--model_name', args.model_name,
                '--tag_name', per_tag,
                '--split', args.split,
            ]
            eval_proc = subprocess.run(eval_cmd, env=env, cwd=str(run_all_script.parent))
            if eval_proc.returncode != 0:
                print(f"Evaluation failed (exit {eval_proc.returncode}) for {cfg_path}")
            else:
                # Attempt to read overall report written by evaluation ( *_overall.json )
                try:
                    eval_dir = run_all_script.parent / 'outputs' / f"{model_part}_{per_tag}" / args.split
                    overall_json = None
                    for f in eval_dir.glob('*_overall.json'):
                        overall_json = f
                        break
                    if overall_json and overall_json.exists():
                        eval_stats = json.loads(overall_json.read_text())
                        # overall.success_avg is already between 0 and N tasks; we treat it as average passes per trial.
                        if 'overall' in eval_stats and 'success_avg' in eval_stats['overall']:
                            eval_success = float(eval_stats['overall']['success_avg'])
                except Exception as e:  # pragma: no cover
                    print(f"Warning: failed to parse evaluation stats for {cfg_path}: {e}")
        summaries.append({
            'cfg': cfg_path.name,
            'tag': per_tag,
            'success_rate': success_rate,
            'eval_success_avg': eval_success,
        })

    if not summaries:
        print('No successful runs recorded.')
        return

    use_eval_metric = any(r.get('eval_success_avg') is not None for r in summaries)
    if use_eval_metric:
        summaries.sort(key=lambda r: (-(r.get('eval_success_avg') or 0.0), -r['success_rate']))
    else:
        summaries.sort(key=lambda r: -r['success_rate'])

    if RICH_OK:
        console = Console()
        table = Table(title='OfficeBench CtxOpt Config Comparison')
        table.add_column('Rank', justify='right')
        table.add_column('Config')
        table.add_column('Tag')
        table.add_column('Success Rate', justify='right')
        if any(rs.get('eval_success_avg') is not None for rs in summaries):
            table.add_column('Eval success_avg', justify='right')
        for i, rs in enumerate(summaries, start=1):
            row = [str(i), rs['cfg'], rs['tag'], f"{rs['success_rate']*100:.1f}%"]
            if 'eval_success_avg' in rs and rs['eval_success_avg'] is not None:
                row.append(f"{rs['eval_success_avg']:.3f}")
            table.add_row(*row)
        console.print('\n')
        console.print(table)
        best = summaries[0]
        if use_eval_metric and best.get('eval_success_avg') is not None:
            console.print(f"\nBest: [bold]{best['tag']}[/] -> Eval success_avg {best['eval_success_avg']:.3f} (success {best['success_rate']*100:.1f}%)")
        else:
            console.print(f"\nBest: [bold]{best['tag']}[/] -> Success {best['success_rate']*100:.1f}%")
    else:
        print('\nOfficeBench CtxOpt Config Comparison')
        print('Rank | Config | Tag | Success Rate | Eval success_avg')
        for i, rs in enumerate(summaries, start=1):
            eval_field = f"{rs['eval_success_avg']:.3f}" if rs.get('eval_success_avg') is not None else '-'
            print(f"{i} | {rs['cfg']} | {rs['tag']} | {rs['success_rate']*100:.1f}% | {eval_field}")
        best = summaries[0]
        if use_eval_metric and best.get('eval_success_avg') is not None:
            print(f"Best: {best['tag']} -> Eval success_avg {best['eval_success_avg']:.3f} (success {best['success_rate']*100:.1f}%)")
        else:
            print(f"Best: {best['tag']} -> Success {best['success_rate']*100:.1f}%")

    try:
        best_cfg_path = (config_out_dir / summaries[0]['cfg']).resolve()
        best_stem = best_cfg_path.stem
        best_out_path = config_out_dir / f"best_{best_stem}.yaml"
        if best_cfg_path.exists():
            shutil.copyfile(best_cfg_path, best_out_path)
            print(f"Saved best config copy to: {best_out_path}")
    except Exception as e:  # pragma: no cover
        print(f"Warning: failed to save best config copy: {e}")

    try:
        ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = prompts_dir / f'officebench_ctxopt_comparison_{ts_stamp}.md'
        header_extra = ' | Eval success_avg' if any(rs.get('eval_success_avg') is not None for rs in summaries) else ''
        lines = [f"# OfficeBench Context-Opt Comparison ({args.ctxopt_type})\n", f"Model: {args.model_name} • Split: {args.split}\n", f"| Rank | Config | Tag | Success Rate{header_extra} |", f"| ---: | :----- | :-- | -----------: {'| ---------------: ' if header_extra else '|'}"]
        for i, rs in enumerate(summaries, start=1):
            if rs.get('eval_success_avg') is not None:
                lines.append(f"| {i} | {rs['cfg']} | {rs['tag']} | {rs['success_rate']*100:.1f}% | {rs['eval_success_avg']:.3f} |")
            else:
                lines.append(f"| {i} | {rs['cfg']} | {rs['tag']} | {rs['success_rate']*100:.1f}% |")
        best = summaries[0]
        lines.append('\n')
        if use_eval_metric and best.get('eval_success_avg') is not None:
            lines.append(f"Best: **{best['tag']}** → Eval success_avg {best['eval_success_avg']:.3f} (success {best['success_rate']*100:.1f}%)\n")
        else:
            lines.append(f"Best: **{best['tag']}** → Success {best['success_rate']*100:.1f}%\n")
        md_path.write_text('\n'.join(lines))
        print(f'Saved comparison markdown: {md_path}')
    except Exception as e:  # pragma: no cover
        print(f'Warning: failed to save markdown summary: {e}')


if __name__ == '__main__':
    main()
