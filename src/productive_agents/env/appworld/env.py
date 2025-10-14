import json
import os
import re
import traceback
from typing import Any, Dict, List, Tuple, Optional
import logging

from productive_agents.env.base import BaseLanguageBasedEnv
from productive_agents.utils import all_seed

from appworld import AppWorld, load_task_ids

from .config import AppWorldEnvConfig


class AppWorldTask:
    """Simple task class that mimics AppWorld Task interface"""
    
    def __init__(self, task_id: str, instruction: str, supervisor: str = ""):
        self.task_id = task_id
        self.instruction = instruction
        self.supervisor = supervisor


class AppWorldEnv(BaseLanguageBasedEnv):
    """
    AppWorld environment implementation following the productive-agents framework.
    
    This environment wraps the AppWorld benchmark to work with the base environment
    interface and follows the patterns established in OfficeBenchEnv.
    """
    
    name = "appworld"
    
    def __init__(self, config: Optional[AppWorldEnvConfig] = None, **kwargs):
        super().__init__()
        
        self.config = config or AppWorldEnvConfig()
        self.kwargs = kwargs
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.config.verbose:
            self.logger.disabled = True
        
        # Environment state
        self.task: Optional[AppWorldTask] = None
        self.task_id: Optional[str] = None
        self.observation: str = ""
        self.done: bool = False
        self.reward: float = 0.0
        self.info: Dict[str, Any] = {}
        self.trajectory: List[Dict[str, Any]] = []
        self.num_interactions: int = 0

        # AppWorld specific state  
        self.world: Optional[AppWorld] = None
        self.task_completed_flag = False
        
        # Initialize with default experiment name
        self.experiment_name = getattr(config, 'experiment_name', "minimal_test")
        
        self.logger.info(f"AppWorld environment initialized with config: {self.config}")
    
    def reset(self, seed: Optional[int] = None, task_id: Optional[str] = None, **kwargs) -> str:
        """
        Reset the environment for a new task.
        
        Args:
            seed: Random seed for reproducibility
            task_id: Specific task ID to load (required for AppWorld)
            **kwargs: Additional reset parameters
            
        Returns:
            Initial observation string
        """
        if seed is not None:
            all_seed(seed)
        
        if task_id is None:
            raise ValueError("task_id is required for AppWorld environment")
        
        self.task_id = task_id
        
        # Reset environment state
        self.observation = ""
        self.done = False
        self.reward = 0.0
        self.info = {}
        self.trajectory = []
        self.num_interactions = 0
        self.task_completed_flag = False
        
        # Initialize AppWorld environment context
        self.world = AppWorld(task_id=self.task_id, experiment_name=self.experiment_name)

        # Set task
        self.task = self.world.task
        
        # Set initial observation
        self.observation = f"Task: {self.task.instruction}\n\nYou can execute Python code to complete this task. What would you like to do?"
        
        self.logger.info(f"Environment reset with task: {self.task_id}")
        self.logger.info(f"Task instruction: {self.task.instruction}")
        
        return self.observation
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Code block to execute (should be Python code)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.num_interactions += 1
        
        # Check for max interactions
        if self.num_interactions >= self.config.max_interactions:
            self.done = True
            self.observation = f"Maximum interactions ({self.config.max_interactions}) reached."
            self.reward = 0.0
            self.info = {"success": False, "reason": "max_interactions"}
            return self.observation, self.reward, self.done, self.info
        
        try:
            # Execute the code block
            output = self._execute_code(action)
            
            # Check if task is completed
            is_completed = self._check_task_completion(action, output)
            
            if is_completed:
                self.done = True
                self.task_completed_flag = True
                self.reward = 1.0
                self.observation = output + "\n\nTask completed successfully!"
                self.info = {"success": True, "reason": "task_completed"}
                # evaluation: dict = self.world.evaluate().to_dict()
                # self.info["evaluation"] = evaluation
            else:
                self.done = False
                self.reward = 0.0
                self.observation = output
                self.info = {"success": False, "reason": "in_progress"}
                # self.info["evaluation"] = None
            
            # Record interaction
            self.trajectory.append({
                "step": self.num_interactions,
                "action": action,
                "output": output,
                "reward": self.reward,
                "done": self.done
            })
            
        except Exception as e:
            # Handle execution errors
            self.logger.error(f"Error executing action: {e}")
            self.observation = f"Error executing code: {str(e)}"
            self.reward = -0.1  # Small penalty for errors
            self.done = False
            self.info = {"success": False, "reason": "execution_error", "error": str(e)}
        
        return self.observation, self.reward, self.done, self.info
    
    def _execute_code(self, code: str) -> str:
        """
        Execute Python code and return output.
        
        In a real implementation, this would use the AppWorld execution environment.
        For now, we'll simulate code execution.
        """
        # Clean the code (remove markdown formatting if present)
        code = self._clean_code(code)
        
        output = self.world.execute(code)
        return output
    
    def _clean_code(self, code: str) -> str:
        """Clean code by removing markdown formatting."""
        # Remove ```python and ``` markers
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = code.strip()
        return code
    
    def _check_task_completion(self, action: str, output: str) -> bool:
        """
        Check if the task has been completed based on action and output.
        
        In a real implementation, this would check against AppWorld's task completion criteria.
        """
        if self.world.task_completed():
            return True
        else:
            return False
    
    def task_completed(self) -> bool:
        """Check if the current task has been completed."""
        return self.task_completed_flag
    
    def execute(self, code: str) -> str:
        """
        Execute code and return output (AppWorld-style interface).
        
        This method provides compatibility with the AppWorld interface.
        """
        obs, reward, done, info = self.step(code)
        return obs
    
    def render(self, mode: str = 'text') -> str:
        """Render the current environment state."""
        return self.observation
    
    def close(self):
        """Clean up environment resources."""
        if self.world:
            # In real implementation, close AppWorld instance
            self.world.close()
        self.logger.info("AppWorld environment closed")
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions (for compatibility)."""
        return ["execute_code", "complete_task"]
    
    def dump_history(self, output_dir: str):
        """Save trajectory history to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "appworld_trajectory.json"), 'w') as f:
            json.dump({
                "task_id": self.task_id if self.task else None,
                "task_instruction": self.task.instruction if self.task else None,
                "num_interactions": self.num_interactions,
                "completed": self.task_completed_flag,
                "final_reward": self.reward,
                "trajectory": self.trajectory
            }, f, indent=2)

        with open(os.path.join(output_dir, "env_history.json"), 'w') as f:
            json.dump(self.trajectory, f, indent=2)
        
        self.logger.info(f"Trajectory saved to {output_dir}/env_history.json")
