from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class OfficeBenchEnvConfig:
    """Configuration for OfficeAgent environment"""
    env_id: str = 'null'
    local_workdir: str = "./local_workdir"
    output_dir: str = "./officebench_output"
    task: str = None
    task_dir: str = "tasks/officebench/tasks/1-1"
    prompt_file: str = "src/productive_agents/env/officebench/prompts_v2.json"
    app_root_dir: str = "src/productive_agents/env/officebench/apps"
    terminate_token_len: int = 7500