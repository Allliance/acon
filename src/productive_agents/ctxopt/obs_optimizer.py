# Observation Optimizer Module
# This module optimizes observations to reduce context length while preserving important information.

import os
import json
import logging
from typing import Dict, Any, List, Optional
from copy import deepcopy

try:
    from termcolor import cprint
except ImportError:
    def cprint(text, color=None):
        print(text)

# TODO: maybe llm manager can be defined in better way...
from .base import BaseContextOptimizer


class ObservationOptimizer(BaseContextOptimizer):
    """
    Observation optimizer that compresses observations to reduce token usage
    while preserving task-relevant information.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        prompt_dir: str = None, 
        debug_mode: bool = False,
        llm: Optional[Any] = None,  # only for local vLLM model
    ):
        """
        Initialize the observation optimizer.
        
        Args:
            config: Configuration dictionary containing:
                - prompts: Dictionary of prompt template names (optional)
                - model: Model name for LLM backend
                - temperature: Temperature for LLM generation
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        # Use default prompt directory if not specified
        prompt_dir = config.get("obs_prompt_dir", None)
        if prompt_dir is None:
            prompt_dir = 'prompts/context_opt'
            
        super().__init__(config, prompt_dir, debug_mode)
        
        # Set up prompt template names
        self.prompt_config = config.get("prompts", {})
        self.system_template = self.prompt_config.get("prompt_system", "system_prompt")
        self.user_template = self.prompt_config.get("prompt_user", "prompt_user")
        
        # Construct system message
        self.system_message = self.construct_system_message(self.system_template)
        self.model_name = config.get("model", "gpt-4o")
        if "lora_name_obs" in config:
            self.lora_name = config["lora_name_obs"]
        else:
            # Fallback to a general Lora name if not specified
            self.lora_name = config.get("lora_name", None)
        self.temperature = config.get("temperature", 0.0)

        self.obs_summarization_threshold = config.get("obs_summarization_threshold", -1)

        # Initialize LLM (lazy import to avoid circular dependency at module import time)
        if llm is not None:
            self.llm = llm
        else:
            from productive_agents.agents.utils import LLMManager  # local import
            self.llm = LLMManager.create_llm(
                self.model_name, '', self.system_message, self.lora_name
            )
        self.use_llmlingua = config.get("use_llmlingua", False)
        
    def process(self, task: str, observation: str, history: str, raw_history: List, opt_args: Dict, **kwargs) -> str:
        """
        Optimize an observation to reduce context length.
        
        Args:
            task: The task given to the agent
            observation: The current observation to optimize
            history: The history of optimizer's inputs and outputs
            current_app: The current app being used (optional)
            available_apps: Dictionary of available apps
            
        Returns:
            Optimized observation string
        """
        # Build the optimization prompt
        prompt, prompt_args = self._build_optimization_prompt(
            task, observation, history, opt_args
        )
        
        # Generate optimized observation
        if self.use_llmlingua:
            raw_response = self.llmlingua_compress_context(observation, ratio=0.3)
        else:
            raw_response = self.llm.generate(prompt)
            raw_response = raw_response.strip()
        
        # Save interaction to history
        self.add_to_history(self.system_message, prompt, raw_response, prompt_args)
        
        # Parse the response
        keyword = "# Refined Observation"
        optimized_observation = self.parse_output(raw_response, keyword)
        
        if self.debug_mode:
            cprint("Obs. Compression", 'red')
            cprint(raw_response, 'blue')
            
        return optimized_observation
    
    def _build_optimization_prompt(self, task: str, observation: str, history: str, opt_args: Dict) -> str:
        """
        Build the prompt for observation optimization.
        
        Args:
            observation: The observation to optimize
            opt_args: additional arguments for optimization, including:
                - task: The task given to the agent
                - history: The history of optimizer's inputs and outputs
                - current_app: The current app being used (optional)
            
        Returns:
            Formatted prompt string
        """
        # Prepare template arguments
        prompt_args = {
            "task": task,
            "observation": observation,
            "history": history,
        }
        prompt_args.update(opt_args)

        # Render the prompt
        prompt = self.render_template(self.user_template, **prompt_args)
        
        if self.debug_mode:
            logging.debug("Observation Optimizer Prompt:\n" + prompt)
            
        return prompt, prompt_args
    
    def check_summarization_needed(self, observation: str) -> bool:
        """
        Check if observation summarization is needed based on token threshold.
        
        Args:
            observation: The observation to check
            
        Returns:
            True if summarization is needed, False otherwise
        """
        if self.obs_summarization_threshold == -1:
            # Always summarize if threshold is -1
            return True
        
        # Calculate token count
        n_tokens = self.count_tokens(observation)
        
        if self.debug_mode:
            print(" ########### Observation summarization criterion check ###########")
            print(f"        Observation length: {n_tokens}")
            print(f"        History summarization threshold: {self.obs_summarization_threshold}")
            
        return n_tokens > self.obs_summarization_threshold

    def dump_history(self, output_dir: str):
        """
        Dump the observation optimizer's history to a JSON file.
        
        Args:
            output_dir: Directory to save the history file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        history_file = os.path.join(output_dir, "obs_optimizer_history.json")
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Observation optimizer history dumped to {history_file}")

class ObservationOptimizerV2(ObservationOptimizer):
    def __init__(
        self, 
        config: Dict[str, Any], 
        prompt_dir: str = None, 
        debug_mode: bool = False,
        llm: Optional[Any] = None,  # only for local vLLM model
    ):
        """
        Initialize the observation optimizer.
        
        Args:
            config: Configuration dictionary containing:
                - prompts: Dictionary of prompt template names (optional)
                - model: Model name for LLM backend
                - temperature: Temperature for LLM generation
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        super().__init__(config, prompt_dir, debug_mode, llm)
        # The rest is handled in the parent class.
        self.system_message = ''

    
    def process(self, task: str, observation: str, history: str, raw_history: List, opt_args: Dict, **kwargs) -> str:
        """
        Optimize an observation to reduce context length.
        
        Args:
            task: The task given to the agent
            observation: The current observation to optimize
            history: The history of optimizer's inputs and outputs
            current_app: The current app being used (optional)
            available_apps: Dictionary of available apps
            
        Returns:
            Optimized observation string
        """
        # Build the optimization prompt
        prompt, prompt_args = self._build_optimization_prompt(
            task, raw_history, observation, opt_args
        )
        # Generate optimized observation
        raw_response = self.llm.generate(prompt)
        raw_response = raw_response.strip()
        
        # Just for logging
        prompt_args["observation"] = observation
        prompt_args["history"] = history

        # Save interaction to history
        self.add_to_history(self.system_message, prompt, raw_response, prompt_args)
        
        # Parse the response
        keyword = "# Refined Observation"
        optimized_observation = self.parse_output(raw_response, keyword)
        
        if self.debug_mode:
            cprint("Obs. Compression", 'red')
            cprint(raw_response, 'blue')
            
        return optimized_observation
    
    def _build_optimization_prompt(self, task: str, raw_history: List, observation: str, opt_args: Dict) -> str:
        """
        Build the prompt for observation optimization.
        
        Args:
            observation: The observation to optimize
            opt_args: additional arguments for optimization, including:
                - task: The task given to the agent
                - history: The history of optimizer's inputs and outputs
                - current_app: The current app being used (optional)
            
        Returns:
            Formatted prompt string
        """
        # Prepare template arguments
        prompt_args = {
            "task": task
        }
        # prompt_args.update(opt_args)

        # Render the prompt
        prompt = self.render_template(self.user_template, **prompt_args)

        whole_prompt = deepcopy(raw_history)
        whole_prompt.append(
            {
                "role": "user",
                "content": observation
            }
        )
        whole_prompt[-1]["content"] += "\n[COMPRESSION START]\n" + prompt
        
        if self.debug_mode:
            logging.debug("Observation Optimizer Prompt:\n" + prompt)
            
        return whole_prompt, prompt_args