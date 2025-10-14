"""
Task Evaluator module for Blueprint task evaluation.

This module provides functionality to evaluate tasks and collect results.
"""

import json
import os
import logging
import re
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np


# append the parent directory of this file to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluate import (
    evaluate_contain, 
    evaluate_not_contain,
    evaluate_file_exist, 
    evaluate_file_not_exist, 
    evaluate_diff_contain_text, 
    evaluate_excel_cell_value, 
    evaluate_excel_cell_comparator,
    evaluate_exact_match,
    evaluate_calendar_no_overlap
)


# Import our local modules directly
from . import report_generator

@dataclass
class TaskEvaluation:
    """Data class for task evaluation results.  Each list maps to N trials."""
    
    task_id: str = ""
    subtask_id: str = ""
    is_pass: List[bool] = None
    result_available: List[bool] = None
    apps_used: List[Set[str]] = None
    errors: List[List[str]] = None
    num_steps: List[int] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.is_pass is None:
            self.is_pass = []
        if self.result_available is None:
            self.result_available = []
        if self.apps_used is None:
            self.apps_used = []
        if self.errors is None:
            self.errors = []
        if self.num_steps is None:
            self.num_steps = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the TaskEvaluation object to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the evaluation results
        """
        return {
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "is_pass": self.is_pass,
            "result_available": self.result_available,
            "apps_used": [list(app_set) for app_set in self.apps_used],
            "errors": self.errors,
            "num_steps": self.num_steps
        }

class TaskEvaluator:
    """Evaluates tasks and collects results."""
    
    def __init__(self, model_name: str, tag_name: str, output_subdir: str, debug: bool, split: str = None):
        """
        Initialize the task evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            tag_name: Tag for the evaluation
            output_subdir: Subdirectory for outputs
            debug: Enable debug logging
            split: Split name (e.g., 'train', 'test') for fold-based evaluation
        """
        self.model_name = model_name
        self.tag_name = tag_name
        self.output_subdir = output_subdir
        self.debug = debug
        self.split = split
        # Per app-tag aggregated raw trial records (strings '1','2','3')
        self.results_dict = {"1": [], "2": [], "3": []}
        # HTML report fragments
        self.html_results = {"1": [], "2": [], "3": []}
        # Counters for tool usage and error types
        self.tool_stats = Counter()
        self.error_stats = Counter()
        # Tasks for which no result folder/file was found
        self.unfound_result_cases = []
        # Cache of (task_id, subtask_id) -> bool pass outcome across trials (any pass=True)
        self._subtask_outcomes = {}
    
    def evaluate_task(self, task_id: str, subtask_id: str) -> TaskEvaluation:
        """
        Evaluate a single task and return results.
        
        Args:
            task_id: Task identifier
            subtask_id: Subtask identifier
            
        Returns:
            TaskEvaluation object containing evaluation results for multiple trials
        """
        num_app_tag = task_id[0]
        is_pass = []
        result_available = []
        apps_used = []
        errors = []
        num_steps = []

        # Check if results exist
        root_folder = f'{self.output_subdir}/{self.model_name.replace("/","_")}_{self.tag_name}/'
        if self.split:
            root_folder = os.path.join(f'{root_folder}', f'{self.split}/')
        
        if not os.path.exists(root_folder):
            logging.debug(f"Not Found {root_folder}")
            self.unfound_result_cases.append((task_id, subtask_id))
            empty_eval = TaskEvaluation(task_id=task_id, subtask_id=subtask_id)
            self._subtask_outcomes[(task_id, subtask_id)] = False
            return empty_eval

        # For fold-based structure, look for task folders
        if self.split:
            task_folder = f'{root_folder}{task_id}'
            if not os.path.exists(task_folder):
                logging.debug(f"Not Found task folder {task_folder}")
                self.unfound_result_cases.append((task_id, subtask_id))
                empty_eval = TaskEvaluation(task_id=task_id, subtask_id=subtask_id)
                self._subtask_outcomes[(task_id, subtask_id)] = False
                return empty_eval
            
            result_testbed_dir = f'{task_folder}/{subtask_id}/testbed'
            output_dir = f'{task_folder}/{subtask_id}'
            
            # Evaluate the task if results are available
            if os.path.exists(result_testbed_dir):
                result_available.append(True)
                logging.debug(f"Found {result_testbed_dir}")
                try:
                    is_pass.append(self.evaluate_output(task_id, subtask_id, result_testbed_dir))
                    
                    # Extract app usage and error data
                    trial_apps, trial_errors, trial_steps = self._extract_trial_data(output_dir)
                    apps_used.append(trial_apps)
                    errors.append(trial_errors)
                    num_steps.append(trial_steps)
                    
                except Exception as e:
                    is_pass.append(False)
                    apps_used.append(set())
                    errors.append([f"Evaluation error: {str(e)}"])
                    num_steps.append(0)
                    logging.error(f"!!! Error: {task_id}.{subtask_id}: {e}", exc_info=self.debug)
            else:
                logging.debug(f"Not Found {result_testbed_dir}")
                result_available.append(False)
                is_pass.append(False)
                apps_used.append(set())
                errors.append(["Result directory not found"])
                num_steps.append(0)
                self.unfound_result_cases.append((task_id, subtask_id))
            # Record outcome (any pass) for split structure (single trial scenario)
            self._subtask_outcomes[(task_id, subtask_id)] = any(is_pass)
        else:
            # Legacy trial-based structure
            trials = sorted(os.listdir(root_folder))
            trials = [trial for trial in trials if os.path.isdir(os.path.join(root_folder, trial))]
            if len(trials) == 0:
                logging.debug(f"Not Found trials in {root_folder}")
                self.unfound_result_cases.append((task_id, subtask_id))
                empty_eval = TaskEvaluation(task_id=task_id, subtask_id=subtask_id)
                self._subtask_outcomes[(task_id, subtask_id)] = False
                return empty_eval
            
            for trial in trials:
                result_testbed_dir = f'{root_folder}/{trial}/{task_id}/{subtask_id}/testbed'
                output_dir = f'{root_folder}/{trial}/{task_id}/{subtask_id}'
                
                # Evaluate the task if results are available
                if os.path.exists(result_testbed_dir):
                    result_available.append(True)
                    logging.debug(f"Found {result_testbed_dir}")
                    try:
                        is_pass.append(self.evaluate_output(task_id, subtask_id, result_testbed_dir))
                        
                        # Extract app usage and error data
                        trial_apps, trial_errors, trial_steps = self._extract_trial_data(output_dir)
                        apps_used.append(trial_apps)
                        errors.append(trial_errors)
                        num_steps.append(trial_steps)
                        
                    except Exception as e:
                        is_pass.append(False)
                        apps_used.append(set())
                        errors.append([f"Evaluation error: {str(e)}"])
                        num_steps.append(0)
                        logging.error(f"!!! Error: {task_id}.{subtask_id}: {e}", exc_info=self.debug)
                else:
                    logging.debug(f"Not Found {result_testbed_dir}")
                    result_available.append(False)
                    is_pass.append(False)
                    apps_used.append(set())
                    errors.append(["Result directory not found"])
                    num_steps.append(0)
                    self.unfound_result_cases.append((task_id, subtask_id))
                
        # Create evaluation object
        evaluation = TaskEvaluation(
            task_id=task_id,
            subtask_id=subtask_id,
            is_pass=is_pass,
            result_available=result_available,
            apps_used=apps_used,
            errors=errors,
            num_steps=num_steps
        )
                
        # Store the results
        logging.debug(f"task_id: {task_id}, subtask_id: {subtask_id}, evaluation: {evaluation}")
        # Record aggregated success for legacy multi-trial structure (or split)
        self._subtask_outcomes[(task_id, subtask_id)] = any(is_pass)

        # Add to results dictionary for statistics
        self.results_dict[num_app_tag].append({'is_pass': is_pass,'result_available': result_available, 'num_steps': num_steps})
        # Generate HTML results
        self.html_results[num_app_tag].append(report_generator.ReportGenerator.make_html_div(
            task_id,
            subtask_id,
            is_pass,
            result_available,
            apps=list(set([app for app_set in apps_used for app in app_set])),
            model_name=self.model_name,
            tag_name=self.tag_name
        ))
        # Analyze tool usage for statistics
        self._analyze_tool_usage(task_id, subtask_id, is_pass)
        return evaluation
    
    def evaluate_output(self, task_id: str, subtask_id: str, output_dir: str) -> bool:
        """
        Evaluate the output of a task according to its evaluation config.
        
        Args:
            task_id: Task identifier
            subtask_id: Subtask identifier
            output_dir: Directory containing the output to evaluate
            
        Returns:
            Whether the evaluation passed
        """
        logging.debug(f"Evaluating {task_id} {subtask_id} {output_dir}...")
        config_path = f"./tasks/{task_id}/subtasks/{subtask_id}.json"
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            eval_config = config['evaluation']
            for eval_item in eval_config:
                function = eval_item['function']
                args = eval_item['args']
                if not eval(f"{function}(output_dir, args)"):
                    logging.debug(f"Failed: {function} {args}")
                    return False
            return True
        except Exception as e:
            logging.error(f"Error evaluating {task_id}.{subtask_id}: {e}", exc_info=self.debug)
            return False
    
    def _analyze_tool_usage(self, task_id: str, subtask_id: str, is_pass: List[bool]) -> None:
        """
        Analyze tool usage from the task's history file and update global statistics counters.
        This method is used to collect aggregate statistics across all tasks.
        
        Args:
            task_id: Task identifier
            subtask_id: Subtask identifier
            is_pass: List of booleans indicating whether trials passed evaluation
            
        Note:
            This method updates the self.tool_stats and self.error_stats counters.
            For per-trial data extraction, see _extract_trial_data method.
        """
        root_folder = f'{self.output_subdir}/{self.model_name.replace("/","_")}_{self.tag_name}/'
        if self.split:
            root_folder = f'{root_folder}{self.split}/'
        
        if self.split:
            # Split-based structure now uses task_id/subtask_id directly, no per-trial directories.
            task_folder = f'{root_folder}{task_id}'
            if not os.path.exists(task_folder):
                return
            subtask_dir = f'{task_folder}/{subtask_id}'
            result_file = f"{subtask_dir}/llm_history.json"
            if os.path.exists(result_file):
                self._process_history_file(result_file, is_pass)
        else:
            # Legacy trial-based structure
            trials = os.listdir(root_folder)
            trials = [trial for trial in trials if os.path.isdir(os.path.join(root_folder, trial))]
            for trial in trials:
                output_dir = f'{root_folder}/{trial}/{task_id}/{subtask_id}'
                result_file = f"{output_dir}/llm_history.json"
                
                if not os.path.exists(result_file):
                    continue
                    
                self._process_history_file(result_file, is_pass)
    
    def _process_history_file(self, result_file: str, is_pass: List[bool]) -> None:
        """
        Process a single history file for tool usage analysis.
        
        Args:
            result_file: Path to the llm_history.json file
            is_pass: List of booleans indicating whether trials passed evaluation
        """
        try:
            with open(result_file, "r") as f:
                latest_report = json.load(f)
                assert isinstance(latest_report, list), "Expected a list of results"

            # latest_report = '\n'.join(latest_report[-1])  # Get the latest report
            latest_report = '\n'.join([i['content'] for i in latest_report[-1]])
            scratchpad = len(re.findall("wrote to scratchpad", latest_report))
            app_switches = re.findall(r"\[Successfully switched to app: (.*?). Available.*?\]", latest_report)

            # Record app usage statistics
            for app in app_switches:
                app = app.split(' ')[0]
                if app not in self.tool_stats:
                    self.tool_stats[app] = 0
                self.tool_stats[app] += 1

            if 'scratchpad' not in self.tool_stats:
                self.tool_stats['scratchpad'] = 0
            self.tool_stats['scratchpad'] += scratchpad

            app_errors = re.findall(r"\[\s?Error: (.*)\]", latest_report) + \
                re.findall(r"\[\s?Malformed action (.*?)\]", latest_report) + \
                re.findall(r"\[(.*?Failed .*?)\]", latest_report)
            for error in app_errors:
                if error not in self.error_stats:
                    self.error_stats[error] = 0
                self.error_stats[error] += 1
                
        except Exception as e:
            logging.error(f"Error analyzing tool usage for {result_file}: {e}")
    
    def generate_stats(self) -> Dict[str, Any]:
        """
        Generate statistics from the evaluation results.
        
        Returns:
            Dictionary containing evaluation statistics
        """
        stats = {
            'model_name': self.model_name,
            'tag_name': self.tag_name,
            'app_tags': {},
            'tool_stats': self.tool_stats,
            'error_stats': self.error_stats,
            'unfound_result_cases': self.unfound_result_cases,
            # Store task outcomes as single hashable identifiers: f"{task_id}_{subtask_id}" (e.g. "1-1_1")
            'successful_tasks': [],
            'failed_tasks': []
        }
        
        # Safe aggregation helpers: the original lambdas treated any generator as truthy
        # causing np.mean([]) -> NaN when there were no tasks for an app tag (e.g. no '2' or '3' app tasks in a split)
        def _avg(iterable):
            data = list(iterable)
            return float(np.mean(data)) if data else 0.0
        def _std(iterable):
            data = list(iterable)
            return float(np.std(data)) if data else 0.0

        # Calculate success metrics for each app tag
        # this was tricker than expected and I'm open to suggestions for useful metrics
        all_trials = {}
        all_available = {}
        all_steps = {}
        # Populate successful/failed lists from cached outcomes
        for (tid, sid), ok in sorted(
            self._subtask_outcomes.items(),
            key=lambda x: (tuple(int(p) for p in x[0][0].split('-')), int(x[0][1])) if '-' in x[0][0] and x[0][1].isdigit() else (x[0][0], x[0][1])
        ):
            ident = f"{tid}_{sid}"
            if ok:
                stats['successful_tasks'].append(ident)
            else:
                stats['failed_tasks'].append(ident)

        for num_app_tag in self.results_dict:
            results = self.results_dict[num_app_tag]
            #macro_total = sum([len(r['is_available']) for r in results])
            #micro_total = len(results)
            #macro_success_avg = avg(p for r in results for p in r['is_pass']) if macro_total > 0 else 0
            success_trials = {}
            is_available_trials = {}
            num_steps_trials = {}
            for r in results:
                for i, p in enumerate(r['is_pass']):
                    if i not in success_trials:
                        success_trials[i] = []
                    success_trials[i].append(p)
                for i, p in enumerate(r['result_available']):
                    if i not in is_available_trials:
                        is_available_trials[i] = []
                    is_available_trials[i].append(p)
                for i, p in enumerate(r['num_steps']):
                    if i not in num_steps_trials:
                        num_steps_trials[i] = []
                    num_steps_trials[i].append(p)
                    
            success_avg = _avg((sum(success_trials[i]) for i in success_trials))
            success_std = _std((sum(success_trials[i]) for i in success_trials))
            success_total = _avg((len(success_trials[i]) for i in success_trials))

            result_available_avg = _avg((sum(is_available_trials[i]) for i in is_available_trials))
            result_available_std = _std((sum(is_available_trials[i]) for i in is_available_trials))
            result_available_total = _avg((len(is_available_trials[i]) for i in is_available_trials))

            steps_avg = _avg((_avg(num_steps_trials[i]) for i in num_steps_trials))
            steps_std = _std((_avg(num_steps_trials[i]) for i in num_steps_trials))
            #steps_total = avg(len(num_steps_trials[i]) for i in num_steps_trials)


            for i in success_trials:
                if i not in all_trials:
                    all_trials[i] = []
                all_trials[i].extend(success_trials[i])
            for i in is_available_trials:
                if i not in all_available:
                    all_available[i] = []
                all_available[i].extend(is_available_trials[i])
            for i in num_steps_trials:
                if i not in all_steps:
                    all_steps[i] = []
                all_steps[i].extend(num_steps_trials[i])
            
            stats['app_tags'][num_app_tag] = {
                'success_avg': success_avg,
                'success_std': success_std,
                'success_total': success_total,
                'result_available_avg': result_available_avg,
                'result_available_std': result_available_std,
                'result_available_total': result_available_total,
                'steps_avg': steps_avg,
                'steps_std': steps_std,
                #'steps_total': steps_total,
            }

        # Calculate overall success metrics
        #overall_results = self.results_dict['1'] + self.results_dict['2'] + self.results_dict['3']
        overall_success_avg = _avg((sum(all_trials[i]) for i in all_trials))
        overall_success_std = _std((sum(all_trials[i]) for i in all_trials))
        overall_success_total = _avg((len(all_trials[i]) for i in all_trials))

        overall_result_available_avg = _avg((sum(all_available[i]) for i in all_available))
        overall_result_available_std = _std((sum(all_available[i]) for i in all_available))
        overall_result_available_total = _avg((len(all_available[i]) for i in all_available))

        # Calculate overall step metrics
        overall_steps_avg = _avg((_avg(all_steps[i]) for i in all_steps))
        overall_steps_std = _std((_avg(all_steps[i]) for i in all_steps))
        #overall_steps_total = avg(len(all_steps[i]) for i in all_steps)

        stats['overall'] = {
                'success_avg': overall_success_avg,
                'success_std': overall_success_std,
                'success_total': overall_success_total,
                'result_available_avg': overall_result_available_avg,
                'result_available_std': overall_result_available_std,
                'result_available_total': overall_result_available_total,
                'steps_avg': overall_steps_avg,
                'steps_std': overall_steps_std,            
                #'steps_total': overall_steps_total,
            }

        return stats

    def _extract_trial_data(self, output_dir: str) -> Tuple[Set[str], List[str], int]:
        """
        Extract app usage, errors, and step count from trial history.
        
        Args:
            output_dir: Directory containing the trial output
            
        Returns:
            Tuple of (apps_used, errors, num_steps)
        """
        result_file = f"{output_dir}/llm_history.json"
        apps_used = set()
        errors = []
        num_steps = 0
        
        if not os.path.exists(result_file):
            return apps_used, errors, num_steps
            
        try:
            with open(result_file, "r") as f:
                latest_report = json.load(f)
                assert isinstance(latest_report, list), "Expected a list of results"
            
            # Count steps based on LLM interactions
            if type(latest_report[0]) is list:
                num_steps = len(latest_report[-1])
            else:
                num_steps = (len(latest_report[-1]) - 1) // 2
            
            # Now focus on the last step
            latest_report = '\n'.join([i['content'] for i in latest_report[-1]])
            
            # Extract app switches
            app_switches = re.findall(r"\[Successfully switched to app: (.*?). Available.*?\]", latest_report)
            apps_used = set([app.split(' ')[0] for app in app_switches])
            
            # Extract error patterns
            errors = re.findall(r"\[\s?Error: (.*)\]", latest_report) + \
                re.findall(r"\[\s?Malformed action (.*?)\]", latest_report) + \
                re.findall(r"\[(.*?Failed .*?)\]", latest_report)


        except Exception as e:
            logging.error(f"Error extracting trial data from {output_dir}: {e}")
            
        return apps_used, errors, num_steps