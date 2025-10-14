from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from productive_agents.env.base import BaseEnvConfig


@dataclass
class AppWorldEnvConfig(BaseEnvConfig):
    """Configuration for the AppWorld environment."""
    
    # Environment settings
    env_id: str = "appworld"
    experiment_name: str = "minimal_react_agent"
    max_interactions: int = 50
    
    # Task settings
    dataset_split: str = "train"  # train, dev, test_normal, test_challenge
    task_id: Optional[str] = None
    
    # Directory settings
    local_workdir: str = "./workdir"
    output_dir: str = "./outputs"
    
    # Environment behavior
    verbose: bool = True
    timeout_duration: int = 30
    
    # Debug settings
    debug_mode: bool = False
    
    def __post_init__(self):
        super().__init__()
        # Set invalid action defaults
        self.invalid_act = "Invalid action"
        self.invalid_act_score = -1.0


# Factory functions for creating configs
def create_appworld_config(**kwargs) -> AppWorldEnvConfig:
    """Create AppWorld environment config from parameters."""
    return AppWorldEnvConfig(**kwargs)
