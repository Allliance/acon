#!/usr/bin/env python3
"""
Entry point for running all AppWorld experiments.

This script runs tasks from different AppWorld dataset splits with configurable parameters.
"""

import argparse
import shutil
import os
import json
import time
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any

from run import main
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress noisy loggers
    for logger_name in ["httpx", "azure.identity", "azure.core"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_base_config(tag: str, model_name: str, use_workflow_memory: bool = False, co_config: dict = None) -> dict:
    """Load and create minimal configuration."""
    base_config_path = Path("configs/base_config.yaml")
    
    if base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal fallback config
        config = {}
    
    # Override with runtime values
    config.update({
        'exp_id': f"{tag}_{model_name}",
        'model_name': model_name,
        'tag': tag,
        'max_iter': 50,
        'use_workflow_memory': use_workflow_memory,
        'use_thinking_tokens': True,
        'prompt_file': "./prompts/prompts_v1.json",
        'co_config': co_config,
        'experiment_name': f"experiment_{tag}"
    })
    
    return config


def load_task_ids_from_split(split: str) -> List[str]:
    """
    Load task IDs for a given dataset split.
    
    Args:
        split: Dataset split ('train', 'dev', 'test_normal', 'test_challenge')
        
    Returns:
        List of task IDs for the split
    """
    # Try to import AppWorld and load task IDs dynamically
    from appworld import load_task_ids
    return load_task_ids(split)


def main_runner():
    """Main entry point for running AppWorld experiments."""
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Run AppWorld experiments")
    # Allow arbitrary split names (e.g., custom subsets like 'train_history_tiny')
    parser.add_argument("--split", type=str,
                       default="train", help="Dataset split to run (e.g., train, dev, test_normal, test_challenge, train_history_tiny)")
    parser.add_argument("--task_id", type=str, help="Specific task ID to run", default=None)
    parser.add_argument("--task_ids", type=str, nargs='+', help="List of specific task IDs to run", default=None)
    parser.add_argument("--tag", type=str, help="Tag for the experiment", default="debug")
    parser.add_argument("--model_name", type=str, help="Model name to use", default="gpt-4o")
    parser.add_argument("--use_workflow_memory", action='store_true', help="Use workflow memory")
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    parser.add_argument("--co_config_path", type=str, help="Context optimization config file", default=None)
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    parser.add_argument("--max_iter", type=int, help="Maximum iterations per task", default=50)
    parser.add_argument("--continue_existing", action='store_true', help="Continue from existing results")
    parser.add_argument("--rerun_failed", action='store_true', help="Rerun only failed tasks")
    parser.add_argument("--lora_name", type=str, help="LoRA model name for agent", default=None)
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for LLM generation")
    
    # Parse the arguments
    args = parser.parse_args()
    
    setup_logging()
    
    # Load co_config if provided
    co_config = None
    if args.co_config_path:
        with open(args.co_config_path, "r") as file:
            co_config = yaml.safe_load(file)
    elif args.co_config_path:
        if not os.path.exists(args.co_config_path):
            logging.warning(f"Context optimization config file {args.co_config_path} not found. Using default config.")

    if co_config and co_config.get("model_type", None):
        if co_config["model_type"] == "local":
            from productive_agents.llm import vLLMLocal
            model_ctxopt = vLLMLocal(co_config["model"], lora_path=co_config.get("lora_name"))
        else:
            model_ctxopt = None
    else:
        model_ctxopt = None

    # Create minimal configuration
    exp_config = load_base_config(
        tag=args.tag,
        model_name=args.model_name,
        use_workflow_memory=args.use_workflow_memory,
        co_config=co_config
    )
    experiment_name = f'{args.model_name.replace("/","_")}_{args.tag}'

    # Update config with command line arguments, avoiding conflicts
    exp_config.update({
        'max_iter': args.max_iter,
        'experiment_name': experiment_name,
        'debug_mode': args.debug,  # This will override the one from load_base_config
        "co_config_path": args.co_config_path,
        'seed': args.seed
    })
    
    # Clear the error log
    if os.path.exists("error.log"):
        os.remove("error.log")
    
    # Get task list based on arguments
    if args.task_id:
        task_list = [args.task_id]
        print(f"Running single task: {args.task_id}")
    elif args.task_ids:
        task_list = args.task_ids
        print(f"Running specified tasks: {task_list}")
    else:
        task_list = load_task_ids_from_split(args.split)
        print(f"Running all tasks from {args.split} split: {len(task_list)} tasks")

    # Set up output directory (route gpt-5-chat to *_gpt5 folders)
    is_gpt5 = 'gpt-5-chat' in args.model_name
    outputs_base = './outputs_gpt5' if is_gpt5 else './outputs'
    # Keep experiments directory consistent (do not divert for gpt-5-chat)
    output_root_dir = f'{outputs_base}/{args.model_name.replace("/","_")}_{args.tag}/{args.split}/'
    experiments_root_dir = f'./experiments/outputs/{args.model_name.replace("/","_")}_{args.tag}/'

    if args.rerun_failed:
        summary_path = Path(output_root_dir, "experiment_summary.json")
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        task_list = summary.get('failed_tasks', [])
        for _task in task_list:
            # remove experiment folder
            exp_file = Path(experiments_root_dir, 'tasks', f'{_task}')
            if exp_file.exists():
                print(f"Removing existing experiment folder for task {_task}: {exp_file}")
                shutil.rmtree(exp_file)
        print(f"Rerunning {len(task_list)} failed tasks: {task_list}")
    
    start_time = time.time()
    successful_tasks = []
    failed_tasks = []
    
    # Cost tracking variables
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_requests = 0
    task_costs = {}  # Store individual task costs

    # Progress bar with elapsed and ETA across tasks (disabled in debug for clarity)
    total_tasks = len(task_list)
    if args.debug:
        for i, task_id in enumerate(task_list):
            print(f'\n{"="*60}')
            print(f'Running task {task_id} ({i+1}/{total_tasks}) from {args.split} split')
            print(f'{"="*60}')

            task_output_dir = os.path.join(output_root_dir, f'task_{task_id}')
            # In debug we always rerun the first task; skip existing check
            result = main(
                task_id=task_id,
                split=args.split,
                output_dir=task_output_dir,
                exp_config=exp_config,
                model_name=args.model_name,
                debug_mode=True,
                experiment_name=experiment_name,
                max_iter=args.max_iter,
                model_ctxopt=model_ctxopt,
                lora_name=args.lora_name
            )
            
            # Collect cost information
            if 'token_usage' in result and result['token_usage']:
                token_info = result['token_usage']
                task_cost = token_info.get('total_cost_usd', 0.0)
                task_costs[task_id] = task_cost
                total_cost += task_cost
                total_input_tokens += token_info.get('total_input_tokens', 0)
                total_output_tokens += token_info.get('total_output_tokens', 0)
                total_requests += token_info.get('total_requests', 0)
            
            if result.get('success', False):
                successful_tasks.append(task_id)
                print(f"✅ Task {task_id} completed successfully!")
            else:
                failed_tasks.append(task_id)
                print(f"❌ Task {task_id} failed: {result.get('termination_reason', 'unknown')}")
            # Only run one task in debug mode
            break
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        ) as progress:
            pb = progress.add_task("Running tasks", total=total_tasks)

            for i, task_id in enumerate(task_list):
                progress.update(pb, description=f"Task {i+1}/{total_tasks}: {task_id}")
                print(f'\n{"="*60}')
                print(f'Running task {task_id} ({i+1}/{total_tasks}) from {args.split} split')
                print(f'{"="*60}')

                task_output_dir = os.path.join(output_root_dir, f'task_{task_id}')
                if not args.continue_existing and os.path.exists(task_output_dir) and not args.rerun_failed:
                    print(f"Output directory for task {task_id} already exists, skipping...")
                    progress.advance(pb, 1)
                    continue

                result = main(
                    task_id=task_id,
                    split=args.split,
                    output_dir=task_output_dir,
                    exp_config=exp_config,
                    model_name=args.model_name,
                    debug_mode=False,
                    experiment_name=experiment_name,
                    max_iter=args.max_iter,
                    model_ctxopt=model_ctxopt,
                    lora_name=args.lora_name
                )
                
                # Collect cost information
                if 'token_usage' in result and result['token_usage']:
                    token_info = result['token_usage']
                    task_cost = token_info.get('total_cost_usd', 0.0)
                    task_costs[task_id] = task_cost
                    total_cost += task_cost
                    total_input_tokens += token_info.get('total_input_tokens', 0)
                    total_output_tokens += token_info.get('total_output_tokens', 0)
                    total_requests += token_info.get('total_requests', 0)
                
                if result.get('success', False):
                    successful_tasks.append(task_id)
                    print(f"✅ Task {task_id} completed successfully!")
                else:
                    failed_tasks.append(task_id)
                    print(f"❌ Task {task_id} failed: {result.get('termination_reason', 'unknown')}")

                progress.advance(pb, 1)
    
    # Calculate and print experiment summary
    end_time = time.time()
    total_time = end_time - start_time
    
    # Convert to hours, minutes, and seconds
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"APPWORLD EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_name}")
    print(f"Tag: {args.tag}")
    print(f"Total tasks: {len(task_list)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    print(f"Success rate: {len(successful_tasks)/len(task_list)*100:.1f}%")
    print(f"Total experiment time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"{'='*60}")
    
    # Print cost summary
    if total_cost > 0:
        print(f"TOKEN USAGE AND COST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Requests: {total_requests:,}")
        print(f"Total Input Tokens: {total_input_tokens:,}")
        print(f"Total Output Tokens: {total_output_tokens:,}")
        print(f"Total Tokens: {total_input_tokens + total_output_tokens:,}")
        print(f"TOTAL COST: ${total_cost:.6f}")
        if len(task_list) > 1:
            print(f"Average cost per task: ${total_cost/len(task_list):.6f}")
        print(f"{'='*60}")
    
    if failed_tasks:
        print(f"Failed tasks: {failed_tasks}")
    
    # Save experiment summary
    summary = {
        'experiment_config': exp_config,
        'split': args.split,
        'total_tasks': len(task_list),
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': len(successful_tasks)/len(task_list),
        'total_time_seconds': total_time,
        'run_timestamp': time.time(),
        # Add cost information
        'cost_summary': {
            'total_cost_usd': total_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'total_requests': total_requests,
            'average_cost_per_task': total_cost / len(task_list) if len(task_list) > 0 else 0,
            'task_costs': task_costs
        }
    }
    
    summary_path = os.path.join(output_root_dir, 'experiment_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to: {summary_path}")
    
    # Copy outputs to specified directory if requested
    if args.output_dir is not None and args.output_dir != "None":
        outputs_base_name = outputs_base.replace('./','')
        target_dir = f'{args.output_dir}/{outputs_base_name}/{args.model_name.replace("/","_")}_{args.tag}/{args.split}/'
        shutil.copytree(output_root_dir, target_dir, dirs_exist_ok=True)
        print(f'Outputs copied to {target_dir}')

        # experiments are always under experiments/outputs
        target_experiment_dir = f'{args.output_dir}/experiments/outputs/{args.model_name.replace("/","_")}_{args.tag}/'
        shutil.copytree(experiments_root_dir, target_experiment_dir, dirs_exist_ok=True)
        print(f'Outputs copied to {target_experiment_dir}')


if __name__ == "__main__":
    main_runner()
