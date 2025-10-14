"""
Evaluation module for Blueprint tasks.

This module provides functionality to evaluate task results and generate reports.
"""

import fire
import json
import os
import logging

# Import our local modules directly, not as part of a package
from .report_generator import ReportGenerator
from . import task_discovery
from .task_evaluator import TaskEvaluator, TaskEvaluation


# Constants and configuration
DEFAULT_MODEL = 'gpt-4o-2024-05-13'
DEFAULT_TAG = 'test'
DEFAULT_RESULT_DIR = './results'
DEFAULT_OUTPUT_SUBDIR = 'outputs'


def configure_logging(debug):
    """
    Configure logging settings based on debug flag.
    
    Args:
        debug: Whether to enable debug logging
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level)
    logging.getLogger().setLevel(log_level)
    logging.getLogger("utils.evaluate").setLevel(log_level)
    logging.debug("Test")


def load_non_image_tasks():
    """
    Load the list of non-image tasks from the non_image_tasks.txt file.
    
    Returns:
        set: Set of task IDs that are non-image tasks
    """
    non_image_tasks_file = "non_image_tasks.txt"
    non_image_tasks = set()
    
    try:
        with open(non_image_tasks_file, 'r') as f:
            for line in f:
                task_id = line.strip()
                if task_id:  # Skip empty lines
                    non_image_tasks.add(task_id)
        logging.info(f"Loaded {len(non_image_tasks)} non-image tasks from {non_image_tasks_file}")
    except FileNotFoundError:
        logging.warning(f"Non-image tasks file {non_image_tasks_file} not found. Proceeding with all tasks.")
    
    return non_image_tasks


def main(model_name=DEFAULT_MODEL, 
         tag_name=DEFAULT_TAG, 
         result_dir=DEFAULT_RESULT_DIR, 
         output_subdir=DEFAULT_OUTPUT_SUBDIR, 
         debug:bool=False, 
         list_missing:bool=False, 
         html:bool=False,
         web:bool=False,
         task=None,
         only_text:bool=False,
         task_dir="tasks",
         split=None):
    """
    Main function to evaluate tasks and generate reports.
    
    Args:
        model_name: Name of the model to evaluate
        tag_name: Tag for the evaluation
        result_dir: Directory to store results
        output_subdir: Subdirectory for outputs
        debug: Enable debug logging
        list_missing: List missing result cases
        html: Generate HTML report
        web: Generate web report
        task: Specific task to evaluate (optional)
        only_text: Only evaluate non-image tasks (text-only tasks)
        task_dir: Directory containing task definitions
        split: Split name to evaluate (e.g., 'train', 'test') for fold-based evaluation
    """
    # Configure logging
    configure_logging(debug)
    logging.debug("Debug mode is enabled")
    
    # Create result directory if it doesn't exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Set up file paths
    if split:
        result_path = f'{result_dir}/{model_name.replace("/", "_")}_{tag_name}_{split}_result.jsonl'
    else:
        result_path = f'{result_dir}/{model_name.replace("/", "_")}_{tag_name}_result.jsonl'
    
    # Discover tasks
    task_discovery_obj = task_discovery.TaskDiscovery()
    all_tasks_info = task_discovery_obj.discover_tasks(task_dir)
    
    # Filter tasks if only_text flag is set
    if only_text:
        non_image_tasks = load_non_image_tasks()
        all_tasks_info = [(task_id, subtask_id) for task_id, subtask_id in all_tasks_info 
                         if task_id in non_image_tasks]
        logging.info(f"Filtered to {len(all_tasks_info)} non-image tasks")
    
    # Initialize the evaluator
    evaluator = TaskEvaluator(model_name, tag_name, output_subdir, debug, split)
    
    # Open result file for writing task-specific results
    with open(result_path, 'w') as f_result:
        # Evaluate each task
        for task_id, subtask_id in all_tasks_info:
            if task is not None and task_id != task:
                continue
            
            # Evaluate the task
            evaluation = evaluator.evaluate_task(task_id, subtask_id)
            
            # Write task result to file
            f_result.write(json.dumps(evaluation.to_dict()) + '\n')
    
    # Generate statistics
    stats = evaluator.generate_stats()

    # Augment stats with explicit accuracy percentages so they are persisted to JSON
    try:
        for num_app_tag, tag_stats in stats.get('app_tags', {}).items():
            denom = tag_stats.get('result_available_avg', 0.0)
            tag_stats['accuracy_pct'] = (tag_stats['success_avg'] / denom * 100.0) if denom else 0.0
        overall_stats = stats.get('overall', {})
        denom_overall = overall_stats.get('result_available_avg', 0.0)
        overall_stats['accuracy_pct'] = (overall_stats['success_avg'] / denom_overall * 100.0) if denom_overall else 0.0
    except Exception as e:
        logging.error(f"Failed computing accuracy_pct fields: {e}")
    
    # Print report to console
    print(ReportGenerator.format_overall_report(stats))
    
    # Save JSON report
    ReportGenerator.save_json_report(stats, result_path)

    # Also save the overall summary inside the original experiment output directory
    try:
        output_root = f'{output_subdir}/{model_name.replace("/","_")}_{tag_name}/'
        if split:
            output_root = os.path.join(output_root, split)
        os.makedirs(output_root, exist_ok=True)
        summary_filename = os.path.basename(result_path).replace('.jsonl', '_overall.json')
        alt_summary_path = os.path.join(output_root, summary_filename)
        with open(alt_summary_path, 'w') as f_alt:
            json.dump(stats, f_alt, indent=4)
        logging.debug(f"Also saved overall summary to {alt_summary_path}")
    except Exception as e:
        logging.error(f"Failed saving summary to output folder: {e}")
    
    # Print missing results if requested
    if list_missing:
        logging.debug(f"Unfound result cases:")
        for task_id, subtask_id in sorted(stats['unfound_result_cases'], 
                                         key=lambda x: tuple(map(int, x[0].split('-'))) + (int(x[1]),)):
            logging.debug(f"{task_id} {subtask_id}")
        logging.debug('Total unfound result cases: %d', len(stats['unfound_result_cases']))
    
    # Generate HTML report if requested
    if html:
        if web:
            return ReportGenerator.make_html_report(stats, evaluator.html_results, None)
        else:
            ReportGenerator.make_html_report(stats, evaluator.html_results, result_path)


# Update the main function to accept a debug flag
if __name__ == '__main__':
    fire.Fire(main)
