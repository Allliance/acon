#!/usr/bin/env python3
"""
Smolagents context-optimization pipeline

Generates context optimization YAML configs from optimized prompt templates and
runs the smolagents MuSiQue evaluation (`run.py`) once per generated config.

Parallels `experiments/appworld/run_ctxopt_pipeline.py` but targets the
Smolagents MuSiQue runner. After each run it collects EM / F1 from the
produced `summary.json` and prints a comparison table (Rich if available).

Example:
  python experiments/smolagents/run_ctxopt_pipeline.py \
    --prompts-dir experiments/smolagents/prompts/context_opt \
    --model-name gpt-4o-mini \
    --split dev \
    --ctxopt-type history \
    --tag 250821_ctxopt_hist

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
except Exception:
    yaml = None
    HAS_YAML = False

try:
    from rich.console import Console
    from rich.table import Table
    RICH_OK = True
except Exception:
    Console = None
    Table = None
    RICH_OK = False

# ----------------------------- Config Generators ----------------------------- #

def generate_yaml_config_history(template_name: str, prompts_dir: Path, model_name: str, opt_version: int) -> dict:
    return {
        "type": "history",
        "model": model_name,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_history_user": template_name,
        },
        "history_summarization_threshold": 2048,
        "preserve_last_k_turns": 1,
        "history_summary_rule": "reset",
        "history_prompt_dir": str(prompts_dir),
        "history_version": opt_version,
    }

def generate_yaml_config_obs(template_name: str, prompts_dir: Path, model_name: str, opt_version: int) -> dict:
    return {
        "type": "obs",
        "model": model_name,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_user": template_name,
        },
        "obs_summarization_threshold": 400,
        "obs_prompt_dir": str(prompts_dir),
        "obs_version": opt_version,
    }

# ----------------------------- Utility Helpers ------------------------------ #

def discover_prompt_files(prompts_dir: Path, ctxopt_type: str) -> List[Path]:
    all_jinja = sorted([p for p in prompts_dir.glob('*.jinja') if p.name not in {"system_prompt.jinja", "system_success_workflow.jinja", "manifest.json"}])
    if ctxopt_type == 'history':
        preferred = [p for p in all_jinja if p.stem.startswith('improved_history_prompt_')]
    else:
        preferred = [p for p in all_jinja if p.stem.startswith('improved_observation_prompt_')]
    if preferred:
        return sorted(preferred)
    return all_jinja

# ----------------------------- Main Pipeline -------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate ctxopt YAMLs for Smolagents and run MuSiQue eval for each.")
    parser.add_argument('--ctxopt-type', choices=['history', 'obs'], default='history')
    parser.add_argument('--prompts-dir', required=True, help='Directory with optimized *.jinja templates')
    parser.add_argument('--model-name', default='gpt-4o-mini')
    parser.add_argument('--split', default='dev', help='Dataset split passed to run.py (dev/test/train)')
    parser.add_argument('--tag', default=None, help='Base experiment tag; per-config suffix appended automatically')
    parser.add_argument('--config-out-dir', default=None, help='Directory to write generated YAMLs (default: configs/context_opt/<derived>)')
    parser.add_argument('--dry-run', action='store_true', help='Only generate YAMLs; skip execution')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for run.py')
    parser.add_argument('--debug', action='store_true', help='Pass --debug to run.py')
    parser.add_argument('--id-list-file', '--id_list_file', dest='id_list_file', default=None, help='Optional list of example IDs to restrict run (hyphen/underscore both accepted)')
    parser.add_argument('--data-folder', default=None, help='Optional data folder override for run.py')
    parser.add_argument("--opt-version", default=1, type=int)

    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir).resolve()
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    prompt_files = discover_prompt_files(prompts_dir, args.ctxopt_type)
    if not prompt_files:
        raise RuntimeError(f"No prompt templates found in {prompts_dir}")

    base_name = prompts_dir.parent.name
    ts = datetime.now().strftime('%y%m%d')
    suffix = 'history_optimized_prompt' if args.ctxopt_type == 'history' else 'obs_optimized_prompt'

    # version tag detection
    version_tag = None
    version_pattern = r'(?:^|_)(v\d+|gpt\d+)(?:$|_)'
    for name in (prompts_dir.name, base_name):
        m = re.search(version_pattern, name, flags=re.IGNORECASE)
        if m:
            version_tag = m.group(1)
            break

    derived_folder = f"{ts}_{args.model_name}_{suffix}"
    if version_tag:
        derived_folder = f"{derived_folder}_{version_tag}"
    if args.opt_version != 1:
        derived_folder = f"{derived_folder}_optv{args.opt_version}"

    if args.config_out_dir:
        config_out_dir = Path(args.config_out_dir).resolve()
    else:
        config_out_dir = Path(__file__).resolve().parent / 'configs' / 'context_opt' / derived_folder
    config_out_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for pf in prompt_files:
        tpl_stem = pf.stem
        yaml_stem = tpl_stem
        template_ref = tpl_stem
        # Allow cross-mapping (history<->obs) like in AppWorld script
        if args.ctxopt_type == 'obs' and tpl_stem.startswith('improved_history_prompt_'):
            sfx = tpl_stem[len('improved_history_prompt_'):]
            mapped = f'improved_observation_prompt_{sfx}'
            yaml_stem = mapped
            if (prompts_dir / f'{mapped}.jinja').exists():
                template_ref = mapped
        elif args.ctxopt_type == 'history' and tpl_stem.startswith('improved_observation_prompt_'):
            sfx = tpl_stem[len('improved_observation_prompt_'):]
            mapped = f'improved_history_prompt_{sfx}'
            yaml_stem = mapped
            if (prompts_dir / f'{mapped}.jinja').exists():
                template_ref = mapped

        if args.ctxopt_type == 'history':
            cfg = generate_yaml_config_history(template_ref, prompts_dir, args.model_name, args.opt_version)
        else:
            cfg = generate_yaml_config_obs(template_ref, prompts_dir, args.model_name, args.opt_version)

        out_path = config_out_dir / f'{yaml_stem}.yaml'
        if HAS_YAML:
            with open(out_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            # minimal manual YAML
            def q(v: str) -> str:
                return '"' + str(v).replace('"', '\"') + '"'
            lines = []
            lines.append(f"type: {q(cfg['type'])}")
            lines.append(f"model: {q(cfg['model'])}")
            lines.append(f"compressor_type: {q(cfg['compressor_type'])}")
            lines.append('prompts:')
            lines.append(f"  prompt_system: {q(cfg['prompts']['prompt_system'])}")
            if args.ctxopt_type == 'history':
                lines.append(f"  prompt_history_user: {q(cfg['prompts']['prompt_history_user'])}")
                lines.append(f"history_summarization_threshold: {int(cfg['history_summarization_threshold'])}")
                lines.append(f"preserve_last_k_turns: {int(cfg['preserve_last_k_turns'])}")
                lines.append(f"history_summary_rule: {q(cfg['history_summary_rule'])}")
                lines.append(f"history_prompt_dir: {q(cfg['history_prompt_dir'])}")
            else:
                lines.append(f"  prompt_user: {q(cfg['prompts']['prompt_user'])}")
                lines.append(f"obs_summarization_threshold: {int(cfg['obs_summarization_threshold'])}")
                lines.append(f"obs_prompt_dir: {q(cfg['obs_prompt_dir'])}")
            out_path.write_text('\n'.join(lines))
        generated.append(out_path)

    print(f"Generated {len(generated)} ctxopt configs under: {config_out_dir}")
    for p in generated:
        print(f" - {p}")

    # If no explicit id list provided, attempt automatic tiny split file resolution for known patterns
    if not args.id_list_file:
        # For history tiny split
        if args.split == 'train_history_tiny':
            auto_path = Path(__file__).resolve().parent / 'data' / 'MuSiQue_4hop' / 'folds' / 'train_history_tiny.txt'
            if auto_path.exists():
                args.id_list_file = str(auto_path)
        # (Extend here for other splits, e.g., observation tiny)

    if args.dry_run:
        print('Dry-run: skipping execution.')
        return

    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / 'run.py'
    if not run_script.exists():
        raise FileNotFoundError(f"run.py not found at {run_script}")

    # Prepare PYTHONPATH to include repo src
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / 'src')
    env['PYTHONPATH'] = f"{src_path}:{env.get('PYTHONPATH','')}" if env.get('PYTHONPATH') else src_path

    base_tag = args.tag or base_name
    if version_tag and not re.search(version_pattern, base_tag, flags=re.IGNORECASE):
        base_tag = f"{base_tag}_{version_tag}"

    run_summaries: List[Dict[str, Optional[float]]] = []

    for cfg_path in generated:
        yaml_stem = cfg_path.stem
        suffix_candidate = yaml_stem
        for prefix in ('improved_history_prompt_', 'improved_observation_prompt_'):
            if yaml_stem.startswith(prefix):
                sfx = yaml_stem[len(prefix):]
                suffix_candidate = sfx or yaml_stem
                break
        per_tag = f"{base_tag}_{suffix_candidate}" if suffix_candidate not in base_tag else base_tag

        print('\n========================================')
        print(f"Running smolagents eval for config: {cfg_path.name} | tag: {per_tag}")
        print('========================================')

        cmd = [
            sys.executable,
            str(run_script),
            '--split', args.split,
            '--model_name', args.model_name,
            '--tag', per_tag,
            '--co_config_path', str(cfg_path),
        ]
        if args.limit is not None:
            cmd.extend(['--limit', str(args.limit)])
        if args.debug:
            cmd.append('--debug')
        if args.id_list_file:
            # Pass canonical hyphen form downstream (run.py expects --id_list_file?)
            cmd.extend(['--id_list_file', args.id_list_file])
        if args.data_folder:
            cmd.extend(['--data_folder', args.data_folder])

        print('Command:', ' '.join(cmd))
        proc = subprocess.run(cmd, env=env, cwd=str(script_dir))
        if proc.returncode != 0:
            print(f"Run failed (exit {proc.returncode}) for {cfg_path}")
            continue

        # Collect summary
        try:
            # run.py writes outputs/<model>_<tag>/<split>/summary.json under experiments/smolagents
            model_part = args.model_name.replace('/', '_')
            tag_part = per_tag if per_tag else 'notag'
            split_part = (args.split or 'test').lower()
            summary_path = script_dir / 'outputs' / f"{model_part}_{tag_part}" / split_part / 'summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            avg_em = float(summary.get('avg_em', 0.0))
            avg_f1 = float(summary.get('avg_f1', 0.0))
        except Exception as e:
            print(f"Warning: failed to load summary for {cfg_path}: {e}")
            avg_em = 0.0
            avg_f1 = 0.0

        run_summaries.append({
            'cfg': cfg_path.name,
            'tag': per_tag,
            'avg_em': avg_em,
            'avg_f1': avg_f1,
        })

    if not run_summaries:
        print('No successful runs recorded.')
        return

    # Sort by avg_em desc then avg_f1 desc
    run_summaries.sort(key=lambda r: (-r['avg_em'], -r['avg_f1']))

    if RICH_OK:
        console = Console()
        table = Table(title='Smolagents CtxOpt Config Comparison')
        table.add_column('Rank', justify='right')
        table.add_column('Config')
        table.add_column('Tag')
        table.add_column('Avg EM', justify='right')
        table.add_column('Avg F1', justify='right')
        for i, rs in enumerate(run_summaries, start=1):
            table.add_row(str(i), rs['cfg'], rs['tag'], f"{rs['avg_em']:.3f}", f"{rs['avg_f1']:.3f}")
        console.print('\n')
        console.print(table)
        best = run_summaries[0]
        console.print(f"\nBest (highest avg EM): [bold]{best['tag']}[/] -> EM {best['avg_em']:.3f}, F1 {best['avg_f1']:.3f}")
    else:
        print('\nSmolagents CtxOpt Config Comparison')
        print('Rank | Config | Tag | Avg EM | Avg F1')
        for i, rs in enumerate(run_summaries, start=1):
            print(f"{i} | {rs['cfg']} | {rs['tag']} | {rs['avg_em']:.3f} | {rs['avg_f1']:.3f}")
        best = run_summaries[0]
        print(f"Best (highest avg EM): {best['tag']} -> EM {best['avg_em']:.3f}, F1 {best['avg_f1']:.3f}")

    # Persist best-performing config as best.yaml in the same config directory
    try:
        best_cfg_path = (config_out_dir / best['cfg']).resolve()
        best_stem = best_cfg_path.stem
        if best_stem.startswith('improved_'):
            best_stem = best_stem[len('improved_'):]
        best_out_path = config_out_dir / f"best_{best_stem}.yaml"
        if best_cfg_path.exists():
            shutil.copyfile(best_cfg_path, best_out_path)
            print(f"Saved best config copy to: {best_out_path}")
        else:
            print(f"Warning: best config file not found for copying: {best_cfg_path}")
    except Exception as e:
        print(f"Warning: failed to save best config copy: {e}")

    # Save markdown summary next to prompts
    try:
        ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md_path = prompts_dir / f'smolagents_ctxopt_comparison_{ts_stamp}.md'
        lines = []
        lines.append(f"# Smolagents Context-Opt Comparison ({args.ctxopt_type})\n")
        lines.append(f"Model: {args.model_name} • Split: {args.split}\n")
        lines.append('| Rank | Config | Tag | Avg EM | Avg F1 |')
        lines.append('| ---: | :----- | :-- | -----: | -----: |')
        for i, rs in enumerate(run_summaries, start=1):
            lines.append(f"| {i} | {rs['cfg']} | {rs['tag']} | {rs['avg_em']:.3f} | {rs['avg_f1']:.3f} |")
        best = run_summaries[0]
        lines.append('\n')
        lines.append(f"Best: **{best['tag']}** → EM {best['avg_em']:.3f}, F1 {best['avg_f1']:.3f}\n")
        md_path.write_text('\n'.join(lines))
        print(f'Saved comparison markdown: {md_path}')
    except Exception as e:
        print(f'Warning: failed to save markdown summary: {e}')

if __name__ == '__main__':
    main()
