
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ExperimentConfig:
    """Configuration for experiment-specific settings."""
    use_workflow_memory: bool = False
    use_scratchpad: bool = False
    use_thinking_tokens: bool = False
    workflow_memory_file: Optional[str] = None
    
    def __post_init__(self):
        # Set default workflow memory file if using workflow memory but no file specified
        if self.use_workflow_memory and self.workflow_memory_file is None:
            self.workflow_memory_file = "workflows/azure_gpt-4o_test-baseline-newapps-all_single_app.txt"


@dataclass
class OfficeBenchAgentConfig:
    """Configuration for the OfficeBench agent."""
    name: str = "officebench_agent"
    description: str = "Agent for the OfficeBench environment"
    default_timeout: int = 60  # Default timeout in seconds
    error_message: str = "Operation timed out in OfficeBench agent"
    co_config: str = None
    
    # Prompt configuration
    prompt_file: Optional[str] = None
    
    # Experiment configuration
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Additional configuration parameters
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, **kwargs):
        """Initialize config with keyword arguments for backward compatibility."""
        # Set defaults
        self.name = kwargs.get('name', "officebench_agent")
        self.description = kwargs.get('description', "Agent for the OfficeBench environment")
        self.default_timeout = kwargs.get('default_timeout', 60)
        self.error_message = kwargs.get('error_message', "Operation timed out in OfficeBench agent")
        self.prompt_file = kwargs.get('prompt_file', None)
        self.co_config = kwargs.get('co_config', None)
        
        # Handle experiment config
        if 'experiment' in kwargs:
            if isinstance(kwargs['experiment'], ExperimentConfig):
                self.experiment = kwargs['experiment']
            elif isinstance(kwargs['experiment'], dict):
                self.experiment = ExperimentConfig(**kwargs['experiment'])
            else:
                self.experiment = ExperimentConfig()
        else:
            # Extract experiment-related kwargs
            exp_kwargs = {}
            for key in ['use_workflow_memory', 'use_scratchpad', 'use_thinking_tokens', 'workflow_memory_file']:
                if key in kwargs:
                    exp_kwargs[key] = kwargs.pop(key)
            self.experiment = ExperimentConfig(**exp_kwargs)
        
        # Store any additional configuration
        self.extra_config = kwargs
        
        # For backward compatibility, also set attributes directly
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


# Factory function for creating configs from different benchmarks
def create_agent_config(benchmark_name: str, **kwargs) -> OfficeBenchAgentConfig:
    """
    Factory function for creating agent configurations for different benchmarks.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., 'officebench', 'webarena', etc.)
        **kwargs: Additional configuration parameters
        
    Returns:
        Agent configuration object
    """
    if benchmark_name.lower() == 'officebench':
        return OfficeBenchAgentConfig(**kwargs)
    else:
        # For future benchmarks, can extend this
        raise ValueError(f"Unsupported benchmark: {benchmark_name}")