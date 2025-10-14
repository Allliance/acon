import json
import logging
import os
from typing import Dict, List, Optional, Any
from jinja2 import Template

from productive_agents.agents.unified_agent import UnifiedAgent, UnifiedPromptBuilder, UnifiedActionProcessor
from productive_agents.agents.utils import LLMOutput
from .config import OfficeBenchAgentConfig


# Default prompts - can be overridden by config
DEFAULT_PROMPT_DICT = {} # Should not use the default prompt directly, but rather load from a config or file
PROMPT_UNDECIDED_APP = """\
##Available apps: {{ available_apps }}
##Instruction:
 - choose an app from the available apps: {"app": "system", "action": "switch_app", "target_app": [THE_APP_YOU_CHOOSE]}
##Command:"""

PROMPT_DECIDED_APP = """\
##Current apps: {{ current_app }}
##Instruction: Choose one action from the list as the next step. Use the JSON schema provided to format your response. You may optionally include your thinking process.
{{ detailed_instruction }} - switch to another app among {{ available_apps }}: {"app": "system", "action": "switch_app", "target_app": [THE_APP_YOU_CHOOSE] }
 - finish the task with your answer as None if the task is not a question: <think>I'm finished the task.</think><action>{"app": "system", "action": "finish_task", "answer": "None"}</action>
 - finish the task with your answer if the task is a question: <think>I'm finished and the answer is [answer]</think><action>{"app": "system", "action": "finish_task", "answer": [ANSWER]}</action>
##Command:"""


class OfficeBenchPromptBuilder(UnifiedPromptBuilder):
    """Handles prompt construction logic for OfficeBench."""
    
    def build_system_message(self, config: Dict[str, Any], available_apps: Dict[str, Any]) -> str:
        """Build system message from config and available apps."""
        app_introduction = ''
        for app_name, app_module in available_apps.items():
            if hasattr(app_module, 'INTRO'):
                app_introduction += f' - {app_module.INTRO}\n'
        
        return self.prompt_dict['system_message'].format_map({
            'username': config.get('username', 'user'),
            'date': config.get('date', ''),
            'weekday': config.get('weekday', ''),
            'time': config.get('time', ''),
            'app_introduction': app_introduction,
            'testbed_data_path': config.get('testbed_data_path', '')
        }).strip()
    
    def _build_undecided_app_prompt(self, env, context_prefix: str) -> str:
        """Build prompt when no app is currently selected."""
        available_apps = list(env.available_apps.keys())
        
        if not env.observation:
            first_prompt = self.prompt_dict['prompt_undecided_app'].format_map({
                'task': env.task,
                'available_apps': available_apps,
            })
            first_prompt = context_prefix + first_prompt
            return first_prompt
        else:
            postfix_prompt = Template(PROMPT_UNDECIDED_APP.lstrip()).render(
                {
                    'available_apps': available_apps,
                }
            )
            prompt = env.observation + '\n' + postfix_prompt # Add to last user prompt
            return prompt
    
    def build_prompt(self, env, context_sections: List[str]) -> str:
        """Build complete prompt with context sections."""
        context_prefix = ''.join(context_sections)
        
        if env.current_app is None:
            return self._build_undecided_app_prompt(env, context_prefix)
        else:
            return self._build_decided_app_prompt(env, context_prefix)

    def _build_decided_app_prompt(self, env, context_prefix: str) -> str:
        """Build prompt when an app is currently selected."""
        obs = env.observation

        detailed_instruction = self._build_detailed_instruction(env)
        available_apps = [app for app in env.available_apps.keys() if app != env.current_app]
        
        # Add available apps and detailed instructions to the prompt
        postfix_prompt = Template(PROMPT_DECIDED_APP.lstrip()).render(
            {
                'current_app': env.current_app,
                'detailed_instruction': detailed_instruction,
                'available_apps': available_apps,
            }
        )
        prompt = obs + postfix_prompt # Add to last user prompt
        return prompt # Exclude the system message from the history
    
    def _build_detailed_instruction(self, env) -> str:
        """Build detailed instruction text for current app."""
        detailed_instruction = ''
        
        try:
            # Import apps module locally to avoid global dependency
            from productive_agents.env.officebench import apps
            
            for action in env.get_available_actions():
                if (env.current_app in apps.AVAILABLE_ACTIONS and 
                    action in apps.AVAILABLE_ACTIONS[env.current_app]):
                    action_module = apps.AVAILABLE_ACTIONS[env.current_app][action]
                    if hasattr(action_module, 'DEMO'):
                        detailed_instruction += f" - {action_module.DEMO}\n"
                    else:
                        detailed_instruction += f" - {action}: (no description available)\n"
                else:
                    detailed_instruction += f" - {action}: (action details unavailable)\n"
        except Exception as e:
            logging.warning(f"Failed to build detailed instruction: {e}")
            detailed_instruction = " - (Action details unavailable)\n"
        
        return detailed_instruction


class OfficeBenchActionProcessor(UnifiedActionProcessor):
    """Handles action processing and validation for OfficeBench."""
    
    def __init__(self, logger: logging.Logger, action_window_size: int = 5):
        super().__init__(logger)
        self.action_window = []
        self.action_window_size = action_window_size
    
    def extract_action(self, response: str) -> str:
        """Extract and process action from LLM response."""
        action = self._process_action_format(response)
        return self._check_stuck_action(action)
    
    def _process_action_format(self, action: str) -> str:
        """Process and validate action format."""
        if '{' not in action or '}' not in action:
            self.logger.warning(f"Invalid action format: {action}")
            return action
        
        left = action.find('{')
        right = action.rfind('}') + 1
        return action[left:right]
    
    def _check_stuck_action(self, action: str) -> str:
        """Check if agent is stuck repeating the same action."""
        self.action_window.append(action)
        self.action_window = self.action_window[-self.action_window_size:]
        
        if (len(self.action_window) >= self.action_window_size and 
            all(act == self.action_window[0] for act in self.action_window)):
            self.logger.warning(f"Action stuck in window: {action}")
            return json.dumps({'app': 'system', 'action': 'got_stuck'})
        
        return action


class OfficeBenchAgent(UnifiedAgent):
    """
    OfficeBench Agent for multi-app task completion.
    
    This agent specializes the unified agent framework for OfficeBench tasks,
    handling JSON-based actions across multiple applications.
    """
    
    def __init__(
        self, 
        model_name: str, 
        key: str, 
        env, 
        task_config: Dict[str, Any], 
        llm_cache: Optional[Dict] = None, 
        debug_mode: bool = False, 
        exp_config: Optional[OfficeBenchAgentConfig] = None, 
        lora_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize OfficeBench agent."""
        # Validate and set config
        if exp_config is None:
            exp_config = OfficeBenchAgentConfig()
        assert isinstance(exp_config, OfficeBenchAgentConfig), \
            f"exp_config should be a OfficeBenchAgentConfig object, but got {type(exp_config)}"
        
        super().__init__(
            model_name=model_name,
            key=key,
            env=env,
            task_config=task_config,
            llm_cache=llm_cache,
            debug_mode=debug_mode,
            exp_config=exp_config,
            lora_name=lora_name,
            **kwargs
        )
    
    def _post_initialization_hook(self):
        """OfficeBench-specific post-initialization."""
        self._initialize_workflow_memory()
        self._validate_environment()
    
    def _initialize_workflow_memory(self):
        """Initialize workflow memory if enabled."""
        self.workflow_memory = None
        
        if (hasattr(self.exp_config, 'experiment') and 
            hasattr(self.exp_config.experiment, 'use_workflow_memory') and 
            self.exp_config.experiment.use_workflow_memory):
            
            workflow_file = self.exp_config.experiment.workflow_memory_file
            try:
                with open(workflow_file) as f:
                    lines = f.readlines()
                self.workflow_memory = '\n'.join([line.strip() for line in lines])
            except Exception as e:
                logging.warning(f"Failed to load workflow memory from {workflow_file}: {e}")
    
    def _validate_environment(self):
        """Validate environment setup."""
        scratchpad_file = f'{self.env.workdir}/scratchpad.txt'
        if os.path.exists(scratchpad_file):
            raise ValueError(f"Scratchpad file {scratchpad_file} already exists, please remove it before running")
    
    def _create_prompt_builder(self) -> OfficeBenchPromptBuilder:
        """Create OfficeBench-specific prompt builder."""
        prompt_dict = DEFAULT_PROMPT_DICT.copy()
        
        if hasattr(self.exp_config, 'prompt_file') and self.exp_config.prompt_file is not None:
            logging.info(f"Loading prompt file: {self.exp_config.prompt_file}")
            try:
                with open(self.exp_config.prompt_file, 'r') as f:
                    custom_prompts = json.load(f)
                prompt_dict.update(custom_prompts)
            except Exception as e:
                logging.warning(f"Failed to load prompt file {self.exp_config.prompt_file}: {e}")
        else:
            logging.info("Using default prompts")
        
        return OfficeBenchPromptBuilder(prompt_dict)
    
    def _create_action_processor(self) -> OfficeBenchActionProcessor:
        """Create OfficeBench-specific action processor."""
        return OfficeBenchActionProcessor(self.logger)
    
    def _process_response(self, response: str) -> str:
        """Process LLM response into action."""
        # Handle thinking tokens configuration
        if (hasattr(self.exp_config, 'experiment') and 
            hasattr(self.exp_config.experiment, 'use_thinking_tokens') and 
            self.exp_config.experiment.use_thinking_tokens):
            action = response
        else:
            action = self.action_processor.extract_action(response)
        
        if action == '':
            action = response
        
        return action
    
    def _get_workflow_memory(self) -> Optional[str]:
        """Get workflow memory content specific to OfficeBench."""
        return getattr(self, 'workflow_memory', None)
    
    def _build_workflow_memory_content(self, workflow_memory: str) -> str:
        """Build workflow memory content section for OfficeBench."""
        return (f"# Workflow memory\n"
                f"These are common, reusable workflows extracted from successful task completions. "
                f"They represent proven patterns and sequences of actions that can be referenced and adapted "
                f"for similar tasks. Each workflow shows the typical steps, actions, and app usage patterns "
                f"for specific types of tasks.\n\n{workflow_memory}\n\n")
    
    def _should_optimize_observation(self, env, obs) -> bool:
        """OfficeBench-specific observation optimization logic."""
        # Don't optimize switch_app observations and only optimize long observations
        if hasattr(env, 'action') and env.action.get("action") == "switch_app":
            return False
        return len(str(obs)) > 100
    
    def _should_terminate_early(self, env, n_iter: int, max_iter: int) -> bool:
        """OfficeBench-specific early termination logic."""
        # Check for context length issues
        if hasattr(env, 'terminate_due_to_length') and env.terminate_due_to_length:
            return True
        return False
    
    def _determine_success(self, env, reward: float, info: Dict) -> bool:
        """Determine if OfficeBench task was successful."""
        if 'success' in info:
            return info['success']
        return reward > 0


# Factory function for creating OfficeBench agents
def create_officebench_agent(
    model_name: str, 
    key: str, 
    env, 
    task_config: Dict[str, Any], 
    **kwargs
) -> OfficeBenchAgent:
    """Factory function to create an OfficeBench agent."""
    exp_config = kwargs.pop('exp_config', OfficeBenchAgentConfig())
    return OfficeBenchAgent(
        model_name=model_name,
        key=key,
        env=env,
        task_config=task_config,
        exp_config=exp_config,
        **kwargs
    )
