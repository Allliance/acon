"""
Unified Agent Class for Task-Based Benchmarks

This is a unified base agent that consolidates common patterns from AppWorld 
and OfficeBench agents, providing a flexible foundation that can be specialized
for different task environments.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from jinja2 import Template, Environment, FileSystemLoader

from productive_agents.agents.base import BaseAgent, BasePromptBuilder
from productive_agents.agents.memory import MemoryManager
from productive_agents.agents.utils import LLMManager, LLMOutput


class UnifiedPromptBuilder(BasePromptBuilder):
    """Unified prompt builder that can be specialized for different tasks."""
    
    def __init__(self, prompt_dict: Optional[Dict[str, str]] = None, working_dir: str = "."):
        if prompt_dict is None:
            prompt_dict = {}
        super().__init__(prompt_dict)
        self.working_dir = working_dir
        self._template_cache = {}
    
    def _load_jinja_template(self, template_path: str) -> Template:
        """Load and cache a Jinja template from file."""
        if template_path in self._template_cache:
            return self._template_cache[template_path]
        
        # Convert relative path to absolute path based on working directory
        if not os.path.isabs(template_path):
            template_path = os.path.join(self.working_dir, template_path)
        
        # Set up Jinja environment with the template directory
        template_dir = os.path.dirname(template_path)
        template_name = os.path.basename(template_path)
        
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_name)
        
        self._template_cache[template_path] = template
        return template
    
    def build_system_message(self, config: Dict[str, Any], available_apps: Dict) -> str:
        """Build system message for LLM - to be implemented by subclasses."""
        return self.prompt_dict.get("system_message", 
            "You are an AI assistant that helps complete tasks.")
    
    @abstractmethod
    def build_prompt(self, env, context_sections: List[str]) -> str:
        """Build complete prompt for current environment state."""
        pass


class UnifiedActionProcessor(ABC):
    """Unified action processor that can handle different action formats."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    @abstractmethod
    def extract_action(self, response: str) -> str:
        """Extract action from LLM response - implemented by subclasses."""
        pass
    
    def validate_action(self, action: str) -> bool:
        """Validate action format - can be overridden by subclasses."""
        return True


class UnifiedAgent(BaseAgent):
    """
    Unified agent class that consolidates common patterns from different benchmark agents.
    
    This class provides:
    - Common initialization patterns
    - Memory management with optimization support
    - LLM interaction with caching
    - Flexible prompt building
    - Configurable action processing
    - Standard run loop with error handling
    
    Task-specific agents should inherit from this and implement:
    - _create_prompt_builder()
    - _create_action_processor()
    - _process_response()
    """
    
    def __init__(
        self, 
        model_name: str, 
        key: str, 
        env, 
        task_config: Dict[str, Any], 
        llm_cache: Optional[Dict] = None,
        debug_mode: bool = False,
        exp_config: Optional[Any] = None,
        lora_name: Optional[str] = None,
        model_ctxopt: Optional[Any] = None,
        **kwargs
    ):
        """Initialize unified agent with common setup."""
        super().__init__(model_name, key, env, task_config, **kwargs)
        
        # Core attributes
        self.exp_config = exp_config
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(__name__)
        self.llm_cache = llm_cache
        self.lora_name = lora_name
        self.model_ctxopt = model_ctxopt
        self.stop_sequences = None
        # Seed for reproducible LLM outputs if provided in config
        self.seed = None
        try:
            if exp_config is not None:
                if hasattr(exp_config, 'seed') and getattr(exp_config, 'seed') is not None:
                    self.seed = getattr(exp_config, 'seed')
                elif hasattr(exp_config, 'extra_config') and isinstance(exp_config.extra_config, dict):
                    self.seed = exp_config.extra_config.get('seed')
        except Exception:
            self.seed = None
        
        # Initialize all components
        self._initialize_components()
        
        self.logger.info(f"Unified agent initialized with model: {model_name}")
        # optional rich console for pretty debug output
        self._debug_console = None
        if self.debug_mode:
            try:  # pragma: no cover
                from rich.console import Console  # type: ignore
                self._debug_console = Console(highlight=False, soft_wrap=True)
            except Exception:
                self._debug_console = None
        self._system_prompt_printed = False
    
    def _initialize_components(self):
        """Initialize all agent components in the correct order."""
        self._initialize_prompt_builder()
        self._initialize_system_message()
        self._initialize_llm()
        self._initialize_action_processor()
        self._initialize_memory_manager()
        self._post_initialization_hook()
    
    def _initialize_prompt_builder(self):
        """Initialize prompt builder - delegates to subclass."""
        self.prompt_builder = self._create_prompt_builder()
    
    def _initialize_system_message(self):
        """Initialize system message using prompt builder."""
        self.system_message = self.prompt_builder.build_system_message(
            self.config, 
            getattr(self.env, 'available_apps', {})
        )
    
    def _initialize_llm(self):
        """Initialize LLM for generation."""
        self.llm = LLMManager.create_llm(
            model_name=self.model_name,
            key=self.key,
            system_message=self.system_message,
            lora_name=self.lora_name
        )
    
    def _initialize_action_processor(self):
        """Initialize action processor - delegates to subclass."""
        self.action_processor = self._create_action_processor()
    
    def _initialize_memory_manager(self):
        """Initialize memory manager with context optimization support."""
        self.memory_manager = MemoryManager(
            self.exp_config, 
            preinit_model=self.model_ctxopt,
        )
        self.memory_manager.add_system_prompt(self.system_message)
    
    def _post_initialization_hook(self):
        """Hook for subclasses to add custom initialization logic."""
        pass
    
    @abstractmethod
    def _create_prompt_builder(self) -> UnifiedPromptBuilder:
        """Create task-specific prompt builder."""
        pass
    
    @abstractmethod
    def _create_action_processor(self) -> UnifiedActionProcessor:
        """Create task-specific action processor."""
        pass
    
    def _build_context_sections(self, env) -> List[str]:
        """Build context sections for prompt."""
        sections = []
        
        # Add workflow memory if available
        workflow_memory = self._get_workflow_memory()
        if workflow_memory:
            sections.append(self._build_workflow_memory_content(workflow_memory))
        
        # Add any task-specific context sections
        additional_sections = self._get_additional_context_sections(env)
        sections.extend(additional_sections)
        
        return sections
    
    def _get_workflow_memory(self) -> Optional[str]:
        """Get workflow memory content - can be overridden by subclasses."""
        if hasattr(self.exp_config, 'workflow_memory') and self.exp_config.workflow_memory:
            return self.exp_config.workflow_memory
        return None
    
    def _build_workflow_memory_content(self, workflow_memory: str) -> str:
        """Build workflow memory content section."""
        return (f"# Workflow memory\n"
                f"These are common, reusable workflows for this type of task. "
                f"They represent proven patterns and sequences of actions that can be referenced "
                f"and adapted for similar tasks.\n\n{workflow_memory}\n\n")
    
    def _get_additional_context_sections(self, env) -> List[str]:
        """Get additional context sections - can be overridden by subclasses."""
        return []
    
    def build_prompt(self, env) -> str:
        """Build complete prompt for current environment state."""
        context_sections = self._build_context_sections(env)
        return self.prompt_builder.build_prompt(env, context_sections)
    
    def forward(self, prompt) -> LLMOutput:
        """Generate response for the given prompt."""
        try:
            cache_key = str(prompt) if isinstance(prompt, str) else str(prompt)
            if self.llm_cache and cache_key in self.llm_cache:
                if self.debug_mode:
                    print("LLM Cache Hit!")
                response = self.llm_cache[cache_key]
            else:
                response = self._generate_with_fallback(prompt)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            response = f"Error: {e}"
        
        if self.debug_mode:
            self._debug_print(prompt, response)
            if self._debug_console:
                try:  # pragma: no cover
                    from rich.panel import Panel  # type: ignore
                    # Pretty conversation rendering
                    if isinstance(prompt, list):
                        conv_txt = []
                        for m in prompt:
                            role = m.get('role', '').upper()
                            content = m.get('content', '')
                            conv_txt.append(f"[{role}]\n{content}\n")
                        prompt_str = "\n".join(conv_txt)
                    else:
                        prompt_str = str(prompt)
                    self._debug_console.print(Panel(prompt_str, title="LLM Prompt", border_style="cyan", padding=(0,1)))
                    self._debug_console.print(Panel(str(response), title="LLM Raw Response", border_style="green", padding=(0,1)))
                except Exception:
                    pass
        
        # Process response to extract action
        action = self._process_response(response)
        
        return LLMOutput(action=action, response=response)

    def _generate_with_fallback(self, prompt):
        """Try generation with stop sequences; if model rejects 'stop', retry without."""
        # Hard override: certain models (e.g., gpt-5-mini) reject stop parameters.
        if getattr(self, "model_name", "") == "gpt-5-mini":
            return self.llm.generate(prompt, seed=self.seed) if self.seed is not None else self.llm.generate(prompt)
        if self.stop_sequences:
            try:
                if self.seed is not None:
                    return self.llm.generate(prompt, stop_sequences=self.stop_sequences, seed=self.seed)
                return self.llm.generate(prompt, stop_sequences=self.stop_sequences)
            except Exception as e:
                msg = str(e)
                if ("unsupported" in msg.lower() or "not supported" in msg.lower()) and "stop" in msg.lower():
                    self.logger.warning("Model rejected stop parameter; retrying without stop sequences.")
                    self.stop_sequences = None
                else:
                    raise
        if self.seed is not None:
            return self.llm.generate(prompt, seed=self.seed)
        return self.llm.generate(prompt)
    
    @abstractmethod
    def _process_response(self, response: str) -> str:
        """Process LLM response into action - implemented by subclasses."""
        pass
    
    def run(self, env, max_iter: int = 50) -> Dict[str, Any]:
        """
        Execute the agent in the environment for a complete task.
        
        This provides a unified run loop that works for different task types.
        """
        self.logger.info(f"Starting task execution with max_iter={max_iter}")
        
        results = {
            'success': False,
            'iterations': 0,
            'final_reward': 0.0,
            'done': False,
            'info': {},
            'termination_reason': 'unknown',
            'task_id': getattr(env, 'task_id', None),
        }
        
        # Add task instruction if available
        if hasattr(env, 'task'):
            if hasattr(env.task, 'instruction'):
                results['task_instruction'] = env.task.instruction
            elif hasattr(env, 'task') and isinstance(env.task, str):
                results['task_instruction'] = env.task

        try:
            self.logger.info(f"Starting task: {results.get('task_instruction', 'No task available')}")
            
            # Initialize environment state
            done = False
            n_iter = 0
            obs = getattr(env, 'observation', None)
            
            while not done and n_iter < max_iter:
                n_iter += 1
                
                if self.debug_mode:
                    print(f"\n--- Iteration {n_iter}/{max_iter} ---")
                    if obs:
                        obs_display = obs[:200] + "..." if len(str(obs)) > 200 else obs
                        print(f"Current observation: {obs_display}")
                    if not self._system_prompt_printed and self._debug_console and getattr(self, 'system_message', None):
                        try:  # pragma: no cover
                            from rich.panel import Panel  # type: ignore
                            self._debug_console.print(Panel(self.system_message, title="System Prompt", border_style="yellow", padding=(0,1)))
                        except Exception:
                            pass
                        self._system_prompt_printed = True
                
                # Build and add user prompt
                user_prompt = self.build_prompt(env)
                self.memory_manager.add_user_prompt(user_prompt)
                
                # History optimization if enabled
                if self.memory_manager.do_history_optimization:
                    try:
                        task_desc = results.get('task_instruction', '')
                        self.memory_manager.optimize_history(
                            task=task_desc,
                            opt_args={}
                        )
                    except Exception as e:
                        self.logger.warning(f"History optimization failed: {e}")

                # Get conversation history as prompt
                prompt = self.memory_manager.get_conversation_history(exclude_system=True)

                # Generate action
                llm_output = self.forward(prompt)
                action = llm_output.action
                raw_response = llm_output.response
                
                if self.debug_mode:
                    action_display = action[:200] + "..." if len(str(action)) > 200 else action
                    print(f"Generated action: {action_display}")
                
                # Add assistant response to memory
                self.memory_manager.add_assistant_response(raw_response)
                
                # Execute action in environment
                obs, reward, done, info = env.step(action)
                
                # Observation optimization if enabled
                if (not done and obs is not None and 
                    self.memory_manager.do_observation_optimization):
                    try:
                        task_desc = results.get('task_instruction', '')
                        refined_obs = self.memory_manager.optimize_observation(
                            task=task_desc,
                            observation=obs,
                            opt_args={}
                        )
                        env.observation = refined_obs
                        if self.debug_mode:
                            print(f"Observation optimized: {len(str(obs))} -> {len(str(refined_obs))} chars")
                    except Exception as e:
                        self.logger.warning(f"Observation optimization failed: {e}")

                if self.debug_mode:
                    print(f"Reward: {reward}, Done: {done}")
                
                # Update results
                results['iterations'] = n_iter
                results['final_reward'] = reward
                results['done'] = done
                results['info'] = info
                
                # Check for completion
                if done:
                    success = self._determine_success(env, reward, info)
                    results['success'] = success
                    results['termination_reason'] = 'task_completed'
                    self.logger.info(f"Task completed - Success: {success}")
                    break
                
                # Check for early termination conditions
                if self._should_terminate_early(env, n_iter, max_iter):
                    results['termination_reason'] = self._get_early_termination_reason(env)
                    break
            
            if n_iter >= max_iter and not done:
                results['termination_reason'] = 'max_iterations_reached'
                self.logger.warning(f"Reached maximum iterations ({max_iter}) without completion")
                
        except KeyboardInterrupt:
            results['termination_reason'] = 'user_interrupted'
            self.logger.info("Execution interrupted by user")
        except Exception as e:
            results['termination_reason'] = 'error'
            results['error'] = str(e)
            self.logger.error(f"Error during execution: {e}")
        
        return results
    
    def _determine_success(self, env, reward: float, info: Dict) -> bool:
        """Determine if task was successful - can be overridden by subclasses."""
        # Try different ways to determine success
        if hasattr(env, 'task_completed') and callable(env.task_completed):
            return env.task_completed()
        if 'success' in info:
            return info['success']
        return reward > 0
    
    def _should_terminate_early(self, env, n_iter: int, max_iter: int) -> bool:
        """Check if should terminate early - can be overridden by subclasses."""
        # Check for context length issues
        if hasattr(env, 'terminate_due_to_length') and env.terminate_due_to_length:
            return True
        return False
    
    def _get_early_termination_reason(self, env) -> str:
        """Get reason for early termination - can be overridden by subclasses."""
        if hasattr(env, 'terminate_due_to_length') and env.terminate_due_to_length:
            return 'context_length_exceeded'
        return 'early_termination'
    
    def _debug_print(self, prompt, response: str):
        """Print debug information."""
        print('\n\n' + '>' * 20)
        print(f'System: {self.system_message[:30]}')
        if isinstance(prompt, list):
            last_turn = prompt[-1] if prompt else {}
            if isinstance(last_turn, dict):
                content = last_turn.get('content', str(last_turn))
            else:
                content = str(last_turn)
            print('Prompt: ' + content[:500] + ("..." if len(content) > 500 else ""))
        else:
            prompt_str = str(prompt)
            print(f'Prompt: {prompt_str[:500]}{"..." if len(prompt_str) > 500 else ""}')
        print(f'Response: {response}')
        print('<' * 20 + '\n\n')
    
    def dump_history(self, output_dir: str):
        """Save conversation history to file."""
        self.memory_manager.dump_history(output_dir)
        
        # Also save token usage and cost information
        if hasattr(self.llm, 'get_cost_breakdown'):
            cost_info = self.llm.get_cost_breakdown()
            cost_file = os.path.join(output_dir, 'token_usage_and_cost.json')
            with open(cost_file, 'w') as f:
                json.dump(cost_info, f, indent=2)
    
    def get_token_usage_summary(self):
        """Get summary of token usage and costs."""
        if hasattr(self.llm, 'get_cost_breakdown'):
            return self.llm.get_cost_breakdown()
        return {}


# Utility functions for creating unified configs
def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged


def load_prompt_dict_from_file(prompt_file: str, working_dir: str = ".") -> Dict[str, str]:
    """Load prompt dictionary from JSON file."""
    prompt_path = prompt_file if os.path.isabs(prompt_file) else os.path.join(working_dir, prompt_file)
    try:
        with open(prompt_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load prompt file {prompt_path}: {e}")
        return {}
