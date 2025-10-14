"""
Minimal configuration for OfficeBench experiments.

This module provides a simplified configuration class with only the parameters
that are actually used by the agent and environment.
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from pathlib import Path

from productive_agents.env.officebench.config import OfficeBenchEnvConfig
from productive_agents.agents.officebench.config import OfficeBenchAgentConfig, ExperimentConfig


@dataclass
class OfficeBenchExperimentConfig:
    """
    Minimal configuration for OfficeBench experiments.
    
    Contains only the parameters that are actually used by the agent and environment.
    """
    
    # Essential experiment metadata
    exp_id: str = "default_experiment"
    model_name: str = "gpt-4o"
    tag: Optional[str] = None
    debug_mode: bool = True
    
    # Essential environment configuration  
    local_workdir: str = "./local_workdir"
    task_dir: str = 'tasks/1-1'
    task: Optional[str] = None
    
    # Essential agent configuration
    prompt_file: Optional[str] = "src/productive_agents/env/officebench/prompts_v2.json"
    max_iter: int = 20
    
    # Workflow memory settings (actually used)
    use_workflow_memory: bool = False
    use_thinking_tokens: bool = True
    workflow_memory_file: Optional[str] = None
    
    # Optional parameters for advanced usage
    lora_name: Optional[str] = None
    llm_cache: Optional[Dict] = None
    
    # Context optimization config (important for core experiments)
    co_config: Optional[Dict] = None
    
    
    def to_env_config(self) -> OfficeBenchEnvConfig:
        """Convert to environment configuration with minimal required parameters."""
        return OfficeBenchEnvConfig(
            task=self.task,
            local_workdir=self.local_workdir,
            task_dir=self.task_dir
        )
    
    def to_agent_config(self) -> OfficeBenchAgentConfig:
        """Convert to agent configuration with minimal required parameters."""
        experiment_config = ExperimentConfig(
            use_workflow_memory=self.use_workflow_memory,
            use_thinking_tokens=self.use_thinking_tokens,
            workflow_memory_file=self.workflow_memory_file
        )
        
        return OfficeBenchAgentConfig(
            name=f"officebench_agent_{self.exp_id}",
            description=f"Agent for OfficeBench experiment {self.exp_id}",
            prompt_file=self.prompt_file,
            experiment=experiment_config,
            co_config=self.co_config
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OfficeBenchExperimentConfig':
        """Create configuration from dictionary."""
        # Filter out unknown parameters to keep only what we actually use
        known_fields = set(cls.__dataclass_fields__.keys())
        filtered_config = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_config)
    
    @classmethod
    def from_yaml_file(cls, yaml_file: Union[str, Path]) -> 'OfficeBenchExperimentConfig':
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(yaml_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, Path]) -> 'OfficeBenchExperimentConfig':
        """Load configuration from JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def load_experiment_config(
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> OfficeBenchExperimentConfig:
    """
    Load experiment configuration from file or create from parameters.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        **kwargs: Additional configuration parameters to override
    
    Returns:
        OfficeBenchExperimentConfig instance
    """
    if config_path:
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            base_config = OfficeBenchExperimentConfig.from_yaml_file(config_path)
        elif config_path.suffix.lower() == '.json':
            base_config = OfficeBenchExperimentConfig.from_json_file(config_path)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    else:
        base_config = OfficeBenchExperimentConfig()
    
    # Override with any provided kwargs
    if kwargs:
        # Only update known fields to maintain minimal config
        known_fields = set(base_config.__dataclass_fields__.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        config_dict = base_config.__dict__.copy()
        config_dict.update(filtered_kwargs)
        base_config = OfficeBenchExperimentConfig(**config_dict)
    
    return base_config
