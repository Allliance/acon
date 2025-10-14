import os
import json
import glob
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import jinja2
from jinja2 import Environment, FileSystemLoader
import numpy as np
from statistics import mean

def get_history_file_paths(base_dir: str, file_type: str = "context_optimizer_history") -> List[str]:
    """
    Get all history file paths for a given directory and file type.
    Based on the get_history_file_paths function from token_counting.py
    """
    # Pattern to match the directory structure: base_dir/0/*/*/{file_type}.json
    pattern1 = f"{base_dir}/*/*/{file_type}.json"
    pattern2 = f"{base_dir}/*/{file_type}.json"

    # Use glob to find all matching files
    paths = glob.glob(pattern1) + glob.glob(pattern2) # not +.... 
    paths = list(set(paths))
    
    print(f"Searching with pattern: {pattern1} and {pattern2}")
    print(f"Found {len(paths)} files")
    
    return sorted(paths)

def extract_task_info(task: str, file_path: str) -> Dict[str, str]:
    """Extract task ID and run ID from the file path."""
    parts = file_path.split('/')
    # Structure: .../outputs/experiment_name/0/task_id/run_id/file.json
    if task == "officebench":
        task_id = parts[-3]  # e.g., "1-1"
        run_id = parts[-2]   # e.g., "0"
    elif task == "appworld":
        task_id = parts[-2]
        run_id = ''
    return {"task_id": task_id, "run_id": run_id}

def is_single_app_task(task_id: str) -> bool:
    """Check if the task is a single app task (starts with '1-')."""
    return task_id.startswith("1-")

def is_pass_task(task: str, task_info: Dict[str, str], task_id_to_result: Dict[str, bool]) -> bool:
    """Check if the task passed based on the results dictionary."""
    if len(task_id_to_result) == 0:
        return True

    if task == "officebench":
        task_id = task_info["task_id"]
        run_id = task_info["run_id"]
        _id = f"{task_id}_{run_id}"
    elif task == "appworld":
        task_id = task_info["task_id"]
        _id = task_id
    return task_id_to_result.get(_id, False)

def load_task_results(result_path: str, task: str) -> Dict[str, bool]:
    """Load task results from a jsonl file."""
    task_id_to_result = dict()

    if task == "officebench":
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            successful_tasks = results["successful_tasks"]
            failed_tasks = results["failed_tasks"]
            task_id_to_result.update({t: True for t in successful_tasks})
            task_id_to_result.update({t: False for t in failed_tasks})
    elif task == "appworld":
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        results = results["individual"]
        for k, v in results.items():
            task_id = "task_" + k
            success = v["success"]
            task_id_to_result[task_id] = success
    else:
        raise NotImplementedError(f"Task {task} is not implemented for loading results")
    return task_id_to_result

def parse_output(response: str, keyword: str = "# Refined Observation") -> str:
    """Parse the output from a model response."""
    if keyword in response:
        # Extract the refined observation
        start_idx = response.index(keyword) + len(keyword)
        refined_observation = response[start_idx:].strip()
        response = refined_observation
    response = response.replace('\n', '\\n')
    return response

def parse_action(response: str) -> str:
    """Parse action from a model response."""
    # Find string between <action> and </action>
    match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if match:
        action = match.group(1).strip()
        return action
    else:
        return response

def get_paired_history_paths(base_dir: str, task_id_to_result: Dict[str, bool]) -> List[Tuple[str, str]]:
    """
    Get paired context optimizer and llm history paths for single app tasks that passed.
    
    Returns:
        List of tuples (co_history_path, llm_history_path)
    """
    co_paths = get_history_file_paths(base_dir, "context_optimizer_history")
    llm_paths = get_history_file_paths(base_dir, "llm_history")
    
    # Create mapping from task_id-run_id to paths
    co_path_map = {}
    llm_path_map = {}
    
    for path in co_paths:
        task_info = extract_task_info(path)
        key = f"{task_info['task_id']}-{task_info['run_id']}"
        co_path_map[key] = path
    
    for path in llm_paths:
        task_info = extract_task_info(path)
        key = f"{task_info['task_id']}-{task_info['run_id']}"
        llm_path_map[key] = path
    
    # Find matching pairs for single app tasks that passed
    paired_paths = []
    for key in co_path_map:
        if key in llm_path_map:
            task_info = extract_task_info(co_path_map[key])
            if is_single_app_task(task_info["task_id"]) and is_pass_task(task_info, task_id_to_result):
                paired_paths.append((co_path_map[key], llm_path_map[key]))
    
    return paired_paths

def process_history_file(file_path: str, file_type: str, replace_jinja_path: str = None) -> List[Dict[str, Any]]:
    """Process a single history file and return formatted data for fine-tuning."""
    with open(file_path, 'r', encoding='utf-8') as f:
        history_data = json.load(f)
        
    processed_data = []

    # Feasibility check: ensure the history ends with finish_task (only for officebench)
    # if file_type == "llm_history":
    #     last_action = json.loads(parse_action(history_data[-1][-1]))
    #     if last_action["action"] != "finish_task":
    #         print(f"Skipping {file_path} as it does not end with finish_task")
    #         return []
        
    # load jinja template if provided
    if replace_jinja_path:
        env = Environment(loader=FileSystemLoader(os.path.dirname(replace_jinja_path)))
        template = env.get_template(os.path.basename(replace_jinja_path))

    for step_idx, step_data in enumerate(history_data):
        if len(step_data) >= 3:
            if type(step_data[0]) == str:
                system_prompt = step_data[0]
                user_prompt = step_data[1]
                assistant_response = step_data[2]

                if replace_jinja_path and len(step_data) > 3:
                    # Replace Jinja template placeholders if provided
                    prompt_args = step_data[3]
                    user_prompt = template.render(**prompt_args)
            
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            else:
                messages = step_data
            # Create conversation format for fine-tuning
            conversation = {
                "messages": messages,
                "metadata": {
                    "file_type": file_type,
                    "step_idx": step_idx,
                    "source_file": file_path
                }
            }
            
            processed_data.append(conversation)
    
    return processed_data


def get_train_passed_paths(task: str, base_dir: str, file_type: str, task_id_to_result: Dict[str, bool]) -> List[str]:
    """
    Get file paths for single app tasks that passed evaluation.
    
    Args:
        base_dir: Base directory containing the files
        file_type: Type of history files to get
        task_id_to_result: Dictionary mapping task_id-run_id to pass/fail status
    
    Returns:
        List of file paths for single app tasks that passed
    """
    file_paths = get_history_file_paths(base_dir, file_type)
    
    # Filter for single app tasks that passed
    train_passed_paths = []
    for file_path in file_paths:
        # TODO: we gather only train tasks before entering this function...
        task_info = extract_task_info(task, file_path)
        # if is_single_app_task(task_info["task_id"]) and is_pass_task(task_info, task_id_to_result):
        if is_pass_task(task, task_info, task_id_to_result):
            train_passed_paths.append(file_path)
    
    return train_passed_paths

def compute_io_token_stats(dataset, tokenizer, input_roles=("system", "user"), output_role="assistant", max_samples: int = None) -> Dict[str, Any]:
    """Compute token length statistics for input (prompt) and output (assistant) parts.

    Args:
        dataset: HF Dataset where each item has a "messages" list of {role, content} dicts.
        tokenizer: HuggingFace / Unsloth tokenizer.
        input_roles: Roles counted as input.
        output_role: Role counted as output (typically last assistant message).
        max_samples: If provided, limit computation to first N samples.

    Returns:
        Dictionary with statistics printed to stdout and also returned.
    """
    if len(dataset) == 0:
        print("[compute_io_token_stats] Empty dataset.")
        return {}

    # Optionally slice
    if max_samples is not None and max_samples < len(dataset):
        ds_iter = (dataset[i] for i in range(max_samples))
    else:
        ds_iter = (dataset[i] for i in range(len(dataset)))

    input_lengths = []
    output_lengths = []
    total_lengths = []

    for ex in ds_iter:
        if "messages" not in ex:
            # Skip if already transformed (no messages field)
            continue
        msgs = ex["messages"]
        input_text_parts = []
        output_text_parts = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "") if isinstance(m, dict) else ""
            if role in input_roles:
                input_text_parts.append(content)
            elif role == output_role:
                output_text_parts.append(content)
        input_text = "\n".join(input_text_parts)
        output_text = "\n".join(output_text_parts)

        # Tokenize without adding special tokens to get raw token count
        input_tokens = tokenizer(input_text, add_special_tokens=False).input_ids if input_text else []
        output_tokens = tokenizer(output_text, add_special_tokens=False).input_ids if output_text else []

        input_len = len(input_tokens)
        output_len = len(output_tokens)
        total_len = input_len + output_len
        input_lengths.append(input_len)
        output_lengths.append(output_len)
        total_lengths.append(total_len)

    def _build_stats(arr: List[int]) -> Dict[str, float]:
        if not arr:
            return {k: 0 for k in ["mean", "p50", "p90", "p95", "max"]}
        a = np.array(arr)
        return {
            "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "max": int(a.max()),
        }

    stats = {
        "num_samples": len(input_lengths),
        "input_tokens": _build_stats(input_lengths),
        "output_tokens": _build_stats(output_lengths),
        "total_tokens": _build_stats(total_lengths),
    }

    print("[Token Stats] Computed over", stats["num_samples"], "samples")
    for section in ["input_tokens", "output_tokens", "total_tokens"]:
        sec = stats[section]
        print(f"  {section}: mean={sec['mean']:.1f} p50={sec['p50']:.1f} p90={sec['p90']:.1f} p95={sec['p95']:.1f} max={sec['max']}")

    return stats