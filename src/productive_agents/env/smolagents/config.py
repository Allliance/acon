from dataclasses import dataclass
from productive_agents.env.base import BaseEnvConfig


@dataclass
class SmolagentsEnvConfig(BaseEnvConfig):
    # Core settings
    env_id: str = "smolagents"
    experiment_name: str = "smolagents_agent"
    max_interactions: int = 50

    # Directories
    local_workdir: str = "./workdir"
    output_dir: str = "./outputs"

    # Runtime behavior
    verbose: bool = True
    timeout_duration: int = 30
    debug_mode: bool = False

    def __post_init__(self):
        # Initialize base config fields
        super().__init__()
        self.invalid_act = "Invalid action"
        self.invalid_act_score = -1.0


def create_smolagents_config(**kwargs) -> SmolagentsEnvConfig:
    return SmolagentsEnvConfig(**kwargs)
