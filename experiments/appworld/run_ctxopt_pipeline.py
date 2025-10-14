#!/usr/bin/env python3
"""
Pipeline to:
1) Read optimized prompt templates (.jinja) from a directory.
2) Generate context-optimization YAML configs for AppWorld history optimizer.
3) Launch experiments/appworld/run_all.py for each generated config.

Usage example:
  python experiments/appworld/run_ctxopt_pipeline.py \
    --prompts-dir /home/t-minkikang/productive-agents/experiments/prompt_optimizer/outputs_appworld/250818_history_regression/optimized_prompts \
    --model-name gpt-4.1 \
    --tag 250818_hist_opt \
    --split train_history_tiny

This will create YAMLs under experiments/appworld/configs/context_opt/<derived_folder>/
and sequentially run run_all.py with --co_config_path pointing to each YAML.
"""

import argparse
import os
import sys
import subprocess
import json
import glob
from pathlib import Path
from datetime import datetime
import re
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


def find_repo_root() -> Path:
    # This file is experiments/appworld/run_ctxopt_pipeline.py
    # repo root should be three parents up
    return Path(__file__).resolve().parents[2]


def generate_yaml_config_history(template_name: str, prompts_dir: Path, model_name: str, opt_version: int) -> dict:
    """Create a history optimizer YAML dict for a given prompt template name (without .jinja)."""
    return {
        "type": "history",
        "model": model_name,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_history_user": template_name,
        },
        "history_summarization_threshold": 4096,
        "preserve_last_k_turns": 1,
        "history_summary_rule": "reset",
        # HistoryOptimizer reads history_prompt_dir (not prompt_dir)
        "history_prompt_dir": str(prompts_dir),
        "history_version": opt_version
    }


def generate_yaml_config_obs(template_name: str, prompts_dir: Path, model_name: str, opt_version: int) -> dict:
    """Create an observation optimizer YAML dict for a given prompt template name (without .jinja)."""
    return {
        "type": "obs",
        "model": model_name,
        "compressor_type": "full",
        "prompts": {
            "prompt_system": "system_prompt",
            "prompt_user": template_name,
        },
        "obs_summarization_threshold": 1024,
        # ObservationOptimizer reads obs_prompt_dir (not prompt_dir)
        "obs_prompt_dir": str(prompts_dir),
        "obs_version": opt_version
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ctxopt YAMLs from optimized prompts and run AppWorld experiments.")
    parser.add_argument("--ctxopt-type", choices=["history", "obs"], default="history", help="Context optimization type to generate and run")
    parser.add_argument("--prompts-dir", required=True, type=str, help="Directory containing optimized .jinja prompts (system_prompt.jinja + improved_history_prompt_*.jinja)")
    parser.add_argument("--model-name", default="gpt-4.1-mini", type=str, help="Model name to record in ctxopt YAMLs and to use for run_all")
    parser.add_argument("--main-model-name", default="gpt-4.1-mini", type=str, help="Model name to use for run_all")
    parser.add_argument("--tag", default=None, type=str, help="Experiment tag for run_all (defaults to derived from prompts dir name)")
    parser.add_argument("--split", default="train_history_tiny", type=str, help="Dataset split to run")
    parser.add_argument("--config-out-dir", default=None, type=str, help="Output directory for generated YAMLs. If not set, will create under experiments/appworld/configs/context_opt/<derived>")
    parser.add_argument("--dry-run", action="store_true", help="Only generate configs; do not launch runs")
    parser.add_argument("--task-ids", default=None, type=str, help="Optional: comma-separated task IDs or path to a file with one task ID per line; if provided, bypasses split loader")
    parser.add_argument("--debug", action="store_true", help="Pass --debug to run_all to reduce workload")
    parser.add_argument("--opt-version", default=1, type=int)

    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir).resolve()
    if not prompts_dir.exists() or not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    # Collect prompt templates; prefer improved_<type>_prompt_*.jinja, else any *.jinja except system files
    prompt_files = sorted([p for p in prompts_dir.glob("*.jinja") if p.name not in {"system_prompt.jinja", "system_success_workflow.jinja", "manifest.json"}])
    if args.ctxopt_type == "history":
        preferred = [p for p in prompt_files if p.stem.startswith("improved_history_prompt_")]
    else:
        preferred = [p for p in prompt_files if p.stem.startswith("improved_observation_prompt_")]
    if preferred:
        prompt_files = sorted(preferred)
    if not prompt_files:
        raise RuntimeError(f"No prompt templates found in {prompts_dir}")

    # Derive a folder name for configs
    base_name = prompts_dir.parent.name  # e.g., 250818_observation_regression_v2
    ts = datetime.now().strftime("%y%m%d")
    suffix = "history_optimized_prompt" if args.ctxopt_type == "history" else "obs_optimized_prompt"

    # Detect version tag (e.g., v2 or gpt5) from the prompt directory name or its parent
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
    # Default to experiments/appworld/configs/context_opt/<derived>
    config_out_dir = (
        Path(args.config_out_dir)
        if args.config_out_dir
        else Path(__file__).resolve().parent / "configs" / "context_opt" / derived_folder
    )
    config_out_dir.mkdir(parents=True, exist_ok=True)

    # Generate YAMLs
    generated = []
    for pf in prompt_files:
        tpl_stem = pf.stem  # without .jinja

        # Normalize output YAML name and internal template reference when crossing types
        yaml_stem = tpl_stem
        template_name_for_yaml = tpl_stem
        if args.ctxopt_type == "obs" and tpl_stem.startswith("improved_history_prompt_"):
            suffix = tpl_stem.split("improved_history_prompt_")[-1]
            mapped = f"improved_observation_prompt_{suffix}"
            yaml_stem = mapped
            # Only change the template reference if the mapped template exists
            if (prompts_dir / f"{mapped}.jinja").exists():
                template_name_for_yaml = mapped
        elif args.ctxopt_type == "history" and tpl_stem.startswith("improved_observation_prompt_"):
            suffix = tpl_stem.split("improved_observation_prompt_")[-1]
            mapped = f"improved_history_prompt_{suffix}"
            yaml_stem = mapped
            if (prompts_dir / f"{mapped}.jinja").exists():
                template_name_for_yaml = mapped

        if args.ctxopt_type == "history":
            cfg = generate_yaml_config_history(template_name_for_yaml, prompts_dir, args.model_name, opt_version=args.opt_version)
        else:
            cfg = generate_yaml_config_obs(template_name_for_yaml, prompts_dir, args.model_name, opt_version=args.opt_version)

        out_path = config_out_dir / f"{yaml_stem}.yaml"
        if HAS_YAML:
            with open(out_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            # Minimal YAML writer for this specific schema
            def q(v: str) -> str:
                return '"' + str(v).replace('"', '\\"') + '"'
            lines = []
            lines.append(f"type: {q(cfg['type'])}")
            lines.append(f"model: {q(cfg['model'])}")
            lines.append(f"compressor_type: {q(cfg['compressor_type'])}")
            lines.append("prompts:")
            lines.append(f"  prompt_system: {q(cfg['prompts']['prompt_system'])}")
            if args.ctxopt_type == "history":
                lines.append(f"  prompt_history_user: {q(cfg['prompts']['prompt_history_user'])}")
                lines.append(f"history_summarization_threshold: {int(cfg['history_summarization_threshold'])}")
                lines.append(f"preserve_last_k_turns: {int(cfg['preserve_last_k_turns'])}")
                lines.append(f"history_summary_rule: {q(cfg['history_summary_rule'])}")
                lines.append(f"history_prompt_dir: {q(cfg['history_prompt_dir'])}")
            else:
                lines.append(f"  prompt_user: {q(cfg['prompts']['prompt_user'])}")
                lines.append(f"obs_summarization_threshold: {int(cfg['obs_summarization_threshold'])}")
                lines.append(f"obs_prompt_dir: {q(cfg['obs_prompt_dir'])}")
            out_path.write_text("\n".join(lines))
        generated.append(out_path)

    print(f"Generated {len(generated)} ctxopt configs under: {config_out_dir}")
    for p in generated:
        print(f" - {p}")

    if args.dry_run:
        print("Dry-run enabled; skipping execution.")
        return

    # Prepare to launch run_all.py for each config (path relative to this script)
    script_dir = Path(__file__).resolve().parent
    run_all = script_dir / "run_all.py"
    if not run_all.exists():
        raise FileNotFoundError(f"run_all.py not found at {run_all}")

    base_tag = args.tag or base_name
    # Ensure version tag is reflected in the experiment tag (used in output dir naming)
    if version_tag and not re.search(version_pattern, base_tag, flags=re.IGNORECASE):
        base_tag = f"{base_tag}_{version_tag}"

    # Ensure PYTHONPATH includes src for local imports
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH','')}" if env.get("PYTHONPATH") else src_path

    # Sequentially run
    # Optional task ids
    task_ids_list = None
    if args.task_ids:
        cand = Path(args.task_ids)
        if cand.exists():
            task_ids_list = [line.strip() for line in cand.read_text().splitlines() if line.strip()]
        else:
            task_ids_list = [t.strip() for t in args.task_ids.split(',') if t.strip()]

    run_summaries = []  # collect run_id, cfg, success, token stats later
    for cfg_path in generated:
        # Compute a per-config tag suffix from the YAML stem (prefer numeric suffix if present)
        yaml_stem = cfg_path.stem
        suffix = yaml_stem
        for prefix in ("improved_history_prompt_", "improved_observation_prompt_"):
            if yaml_stem.startswith(prefix):
                sfx = yaml_stem[len(prefix):]
                suffix = sfx if sfx else yaml_stem
                break
        per_run_tag = f"{base_tag}_{suffix}"
        print("\n========================================")
        print(f"Launching run_all for config: {cfg_path.name} | tag: {per_run_tag}")
        print("========================================")
        cmd = [
            sys.executable,
            str(run_all),
            "--split", args.split,
            "--tag", per_run_tag,
            "--model_name", args.main_model_name,
            "--co_config_path", str(cfg_path),
        ]
        if args.debug:
            cmd.append("--debug")
        if task_ids_list:
            cmd.append("--task_ids")
            cmd.extend(task_ids_list)
        print("Command:", " ".join(cmd))
        # Execute and stream output
        proc = subprocess.run(cmd, env=env, cwd=str(script_dir))
        if proc.returncode != 0:
            print(f"run_all failed for {cfg_path} with exit code {proc.returncode}")
            continue

        # Post-run evaluation: launch external CLI
        try:
            run_id = f"{args.main_model_name.replace('/', '_')}_{per_run_tag}"
            eval_cmd = ["appworld", "evaluate", run_id, args.split]
            print("Running evaluation:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, env=env, cwd=str(script_dir))
        except Exception as e:
            print(f"Warning: evaluation command failed for {run_id}: {e}")

        # Gather quick summary from experiment_summary.json
        try:
            summary_path = script_dir / 'outputs' / run_id / args.split / 'experiment_summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            success_rate = float(summary.get('success_rate', 0.0))
            total_tasks = int(summary.get('total_tasks', 0))
        except Exception:
            success_rate, total_tasks = 0.0, 0
        # Read evaluation metric: aggregate.task_goal_completion
        task_goal_completion = None
        try:
            eval_json_path = script_dir / 'experiments' / 'outputs' / run_id / 'evaluations' / f"{args.split}.json"
            if eval_json_path.exists():
                eval_data = json.loads(eval_json_path.read_text())
                task_goal_completion = eval_data.get('aggregate', {}).get('task_goal_completion')
        except Exception:
            task_goal_completion = None
        run_summaries.append({
            'cfg': cfg_path.name,
            'tag': per_run_tag,
            'run_id': run_id,
            'success_rate': success_rate,
            'total_tasks': total_tasks,
            'task_goal_completion': task_goal_completion,
        })

    # After all runs, compute accumulated input tokens and show comparison table
    if run_summaries:
        try:
            # Ensure we can import the analyzer
            repo_root = Path(__file__).resolve().parents[2]
            sys.path.insert(0, str(repo_root))
            from experiments.analysis_tools.utils import analyze_experiment_tokens_v2  # type: ignore

            comp_rows = []
            for rs in run_summaries:
                exp_name = rs['run_id']
                res = analyze_experiment_tokens_v2(exp_name, agent_model=args.main_model_name, optimizer_model=args.model_name)
                agent_in = opt_in = 0
                total_steps = 0
                step_tasks = 0
                if res and isinstance(res.get('tasks', {}), dict):
                    for _tid, t in res['tasks'].items():
                        a = t.get('agent', {}) or {}
                        o = t.get('optimizer', {}) or {}
                        agent_in += int(a.get('total_input_tokens', 0) or 0)
                        opt_in += int(o.get('total_input_tokens', 0) or 0)
                        if 'num_steps' in a:
                            total_steps += int(a.get('num_steps') or 0)
                            step_tasks += 1
                total_in = agent_in + opt_in
                avg_steps = (total_steps / step_tasks) if step_tasks else 0.0
                comp_rows.append({
                    'cfg': rs['cfg'],
                    'tag': rs['tag'],
                    'run_id': exp_name,
                    'tasks': rs['total_tasks'],
                    'success_rate': rs['success_rate'],
                    'task_goal_completion': rs.get('task_goal_completion'),
                    'agent_in': agent_in,
                    'opt_in': opt_in,
                    'total_in': total_in,
                    'avg_steps': avg_steps,
                })

            # Sort by total input tokens ascending
            comp_rows.sort(key=lambda r: r['total_in'])

            if RICH_OK:
                console = Console()
                table = Table(title="Context-Opt Prompt Comparison (Accumulated Input Tokens)")
                table.add_column("Rank", justify="right")
                table.add_column("Config", overflow="fold")
                table.add_column("Tag")
                table.add_column("Tasks", justify="right")
                table.add_column("Avg Steps", justify="right")
                table.add_column("Task GC", justify="right")
                table.add_column("Agent In", justify="right")
                table.add_column("Opt In", justify="right")
                table.add_column("Total In", justify="right")
                for i, r in enumerate(comp_rows, start=1):
                    table.add_row(
                        f"{i}", r['cfg'], r['tag'],
                        str(r['tasks']), f"{r.get('avg_steps', 0):.1f}", f"{(r.get('task_goal_completion') or 0):.1f}%",
                        f"{r['agent_in']:,}", f"{r['opt_in']:,}", f"{r['total_in']:,}"
                    )
                console.print("\n")
                console.print(table)
                best = comp_rows[0]
                console.print(f"\nBest (lowest total input tokens): [bold]{best['tag']}[/] -> {best['total_in']:,} tokens across {best['tasks']} tasks")
            else:
                # Fallback plain text
                print("\nContext-Opt Prompt Comparison (Accumulated Input Tokens)")
                print("Rank | Config | Tag | Tasks | Avg Steps | Task GC | Agent In | Opt In | Total In")
                for i, r in enumerate(comp_rows, start=1):
                    tgc = r.get('task_goal_completion') or 0
                    print(f"{i} | {r['cfg']} | {r['tag']} | {r['tasks']} | {r.get('avg_steps',0):.1f} | {tgc*100:.1f}% | {r['agent_in']} | {r['opt_in']} | {r['total_in']}")
                best = comp_rows[0]
                print(f"Best (lowest total input tokens): {best['tag']} -> {best['total_in']} tokens across {best['tasks']} tasks")

            # Save markdown table into the prompts folder
            try:
                ts_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_md = prompts_dir / f"ctxopt_comparison_{ts_stamp}.md"
                lines = []
                lines.append(f"# Context-Opt Prompt Comparison ({args.ctxopt_type})\n")
                lines.append(f"Model: {args.model_name} • Split: {args.split}\n")
                lines.append("| Rank | Config | Tag | Tasks | Avg Steps | Task GC | Agent In | Opt In | Total In |")
                lines.append("| ---: | :----- | :-- | ---: | ---: | ---: | ---: | ---: | ---: |")
                for i, r in enumerate(comp_rows, start=1):
                    tgc = (r.get('task_goal_completion') or 0) * 100.0
                    lines.append(
                        f"| {i} | {r['cfg']} | {r['tag']} | {r['tasks']} | {r.get('avg_steps',0):.1f} | {tgc:.1f}% | {r['agent_in']:,} | {r['opt_in']:,} | {r['total_in']:,} |"
                    )
                best = comp_rows[0]
                lines.append("\n")
                lines.append(f"Best (lowest total input tokens): **{best['tag']}** → {best['total_in']:,} tokens across {best['tasks']} tasks\n")
                out_md.write_text("\n".join(lines))
                print(f"Saved comparison table: {out_md}")
            except Exception as e:
                print(f"Warning: failed to save comparison table to prompts dir: {e}")
        except Exception as e:
            print(f"Warning: token comparison failed: {e}")

        # Save the best-performing config for both modes (history/obs)
        try:
            if generated:
                # Prefer rows with token info if available (comp_rows), else fall back to run_summaries
                def _tie_tokens(row):
                    ti = row.get('total_in') if isinstance(row, dict) else None
                    return float('inf') if ti is None else float(ti)

                best_cfg_name = None
                try:
                    # Pick by highest task_goal_completion, tie-breaker: lowest total_in
                    best_row = max(
                        comp_rows,  # type: ignore[name-defined]
                        key=lambda r: (
                            float(r.get('task_goal_completion') or 0),
                            -_tie_tokens(r),
                        ),
                    )
                    best_cfg_name = best_row['cfg']
                except Exception:
                    # Fallback: use run_summaries (no token tie-breaker available)
                    best_rs = max(
                        run_summaries,
                        key=lambda r: float(r.get('task_goal_completion') or 0),
                    )
                    best_cfg_name = best_rs['cfg']

                if best_cfg_name:
                    best_cfg_path = next((p for p in generated if p.name == best_cfg_name), None)
                    if best_cfg_path:
                        out_name = (
                            "best_improved_history_prompt_samples.yaml"
                            if args.ctxopt_type == "history"
                            else "best_improved_observation_prompt_samples.yaml"
                        )
                        dest_path = config_out_dir / out_name
                        dest_path.write_text(best_cfg_path.read_text())
                        print(
                            f"Saved best-performing {args.ctxopt_type} config -> {dest_path.name} (from {best_cfg_path.name})"
                        )
                    else:
                        print("Warning: could not locate the best config to copy.")
            else:
                print("Info: no generated configs to select from.")
        except Exception as e:
            print(f"Warning: failed to save best-performing config: {e}")


if __name__ == "__main__":
    main()
