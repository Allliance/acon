from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AppWorldAgentConfig:
    """Configuration for the AppWorld agent."""
    
    # Basic agent settings
    name: str = "appworld_agent"
    description: str = "Agent for the AppWorld environment"
    default_timeout: int = 60
    error_message: str = "Operation timed out in AppWorld agent"
    
    # Prompt configuration
    prompt_file: Optional[str] = None
    
    # Model configuration
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Agent behavior
    use_thinking: bool = True
    max_code_length: int = 2000
    
    # Debug settings
    debug_mode: bool = False
    verbose: bool = True
    
    # Workflow memory configuration
    workflow_memory: Optional[str] = None
    
    # Context optimization configuration
    co_config: Optional[Dict[str, Any]] = None
    
    # Additional configuration parameters
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing for additional config."""
        # Handle any extra configuration that doesn't fit standard fields
        pass


# Factory function for creating configs from different sources
def create_appworld_agent_config(**kwargs) -> AppWorldAgentConfig:
    """Create AppWorld agent config from parameters."""
    # Separate known parameters from extra config
    known_fields = set(AppWorldAgentConfig.__dataclass_fields__.keys())
    config_params = {}
    extra_config = {}
    
    for key, value in kwargs.items():
        if key in known_fields:
            config_params[key] = value
        else:
            extra_config[key] = value
    
    # Set extra_config if there are any extra parameters
    if extra_config:
        config_params['extra_config'] = extra_config
    
    return AppWorldAgentConfig(**config_params)
