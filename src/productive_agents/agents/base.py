from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import logging
import json


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""
    
    def __init__(self, prompt_dict: Dict[str, str]):
        self.prompt_dict = prompt_dict
    
    @abstractmethod
    def build_system_message(self, config: Dict[str, Any], available_apps: Dict[str, Any]) -> str:
        """Build system message from config and available apps."""
        pass
    
    @abstractmethod
    def build_prompt(self, env, obs) -> str:
        """
        Build complete prompt for the current environment state.
        
        Args:
            env: Environment instance
            obs: Current observation string
            
        Returns:
            Complete prompt string
        """
        pass 

class BaseAgent(ABC):
    """
    Abstract base class for benchmark agents.
    
    This provides a common framework that can be extended for different benchmarks
    while maintaining consistency in core functionality.
    """
    
    def __init__(self, model_name: str, key: str, env, config: Dict[str, Any], **kwargs):
        """
        Initialize base agent.
        
        Args:
            model_name: Name of the LLM model to use
            key: API key for the model
            env: Environment instance
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.model_name = model_name
        self.key = key
        self.env = env
        self.config = config
        self.kwargs = kwargs
        
        # Common attributes
        self.debug_mode = kwargs.get('debug_mode', False)
        self.logger = logging.getLogger(__name__)
        self.llm_cache = kwargs.get('llm_cache', None)
        
        # Initialize components (to be implemented by subclasses)
        self.llm = None
        self.prompt_builder = None
        self.action_processor = None
        self.system_message = None
        self.memory_manager = None
    
    @abstractmethod
    def _initialize_llm(self):
        """Initialize the LLM instance."""
        pass
    
    @abstractmethod
    def _initialize_prompt_builder(self):
        """Initialize the prompt builder."""
        pass
    
    @abstractmethod
    def _initialize_action_processor(self):
        """Initialize the action processor."""
        pass
    
    @abstractmethod
    def _build_context_sections(self, env) -> List[str]:
        """Build context sections for prompt (scratchpad, workflow memory, etc.)."""
        pass

    @abstractmethod
    def _initialize_memory_manager(self):
        """Initialize the memory manager"""
        pass
    
    def forward(self, env) -> str:
        """
        Generate next action based on current environment state.
        
        This method provides the core interaction loop and can be overridden
        if different behavior is needed.
        """
        prompt = self.build_prompt(env)
        
        # Generate response with caching support
        try:
            if self.llm_cache and prompt in self.llm_cache:
                if self.debug_mode:
                    print('LLM Cache Hit!')
                response = self.llm_cache[prompt]
            else:
                response = self.llm.generate(prompt)
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            response = f"Error: {str(e)}"
        
        # Log interaction
        self.llm_history.append((self.system_message, prompt, response))
        
        if self.debug_mode:
            self._debug_print(prompt, response)
        
        # Process action
        action = self._process_response(response)
        
        return action
    
    def build_prompt(self, env) -> str:
        """Build complete prompt for current environment state."""
        context_sections = self._build_context_sections(env)
        return self.prompt_builder.build_prompt(env, context_sections)
    
    def run(self, env, max_iter: int = 20) -> Dict[str, Any]:
        """
        Execute the agent in the environment for a complete task.
        
        This is a default implementation that can be overridden by subclasses
        for benchmark-specific behavior.
        
        Args:
            env: Environment instance
            max_iter: Maximum number of iterations
            
        Returns:
            Dictionary containing execution results
        """
        results = {
            'success': False,
            'iterations': 0,
            'final_reward': 0.0,
            'done': False,
            'info': {},
            'termination_reason': 'unknown'
        }
        
        try:
            obs = env.observation if hasattr(env, 'observation') else env.render()
            done = False
            n_iter = 0
            
            while not done and n_iter < max_iter:
                n_iter += 1
                action = self.forward(env)
                obs, reward, done, info = env.step(action)
                
                results['iterations'] = n_iter
                results['final_reward'] = reward
                results['done'] = done
                results['info'] = info
                
                if done:
                    results['success'] = info.get('success', reward > 0)
                    results['termination_reason'] = 'task_completed'
                    break
            
            if n_iter >= max_iter and not done:
                results['termination_reason'] = 'max_iterations_reached'
                
        except KeyboardInterrupt:
            results['termination_reason'] = 'user_interrupted'
        except Exception as e:
            results['termination_reason'] = 'error'
            results['error'] = str(e)
        
        return results
    
    @abstractmethod
    def _process_response(self, response: str) -> str:
        """Process LLM response into action."""
        pass
    
    def _debug_print(self, prompt: str, response: str):
        """Print debug information."""
        print('\n\n' + '>' * 20)
        print(f'System: {self.system_message}')
        print(f'Prompt: {prompt}')
        print(f'Response: {response}')
        print('<' * 20 + '\n\n')
    
    def dump_history(self, output_dir: str):
        """Save LLM conversation history to file."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/llm_history.json', 'w') as f:
            json.dump(self.llm_history, f, indent=2)
