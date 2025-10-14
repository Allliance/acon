"""
Task Discovery module for Blueprint task evaluation.

This module provides functionality to discover and sort evaluation tasks.
"""

import glob
from typing import List, Tuple


class TaskDiscovery:
    """Handles discovery and sorting of tasks."""
    
    @staticmethod
    def discover_tasks(task_dir="tasks") -> List[Tuple[str, str]]:
        """
        Find all tasks and subtasks in the workspace.
        
        Returns:
            List of (task_id, subtask_id) tuples, sorted by task ID
        """
        all_tasks_info = []
        all_config_filepaths = glob.glob(f"./{task_dir}/*/subtasks/*.json")
        for config_filepath in all_config_filepaths:
            # ./tasks/1-1/subtasks/0.json
            task_id = config_filepath.split('/')[2]
            subtask_id = config_filepath.split('/')[4].split('.')[0]
            all_tasks_info.append((task_id, subtask_id))
        
        # Sort by task ID and subtask ID
        return sorted(all_tasks_info, 
                    key=lambda x: tuple(map(int, x[0].split('-'))) + (int(x[1]),))