import json
import os
import re
import traceback
from typing import Any, Dict, List, Tuple, Optional
import logging

from smolagents import LocalPythonExecutor
from smolagents.local_python_executor import fix_final_answer_code
from smolagents.utils import parse_code_blobs, truncate_content, extract_code_from_text

from productive_agents.env.base import BaseLanguageBasedEnv
from productive_agents.utils import all_seed

from .config import SmolagentsEnvConfig
from .tool import WikipediaRetrieverTool, FinalAnswerTool


class SmolagentsEnv(BaseLanguageBasedEnv):
    """
    Smolagents environment implementation following the productive-agents framework.
    
    This environment wraps the Smolagents library (executor) to work with the base environment
    interface and follows the patterns established in OfficeBenchEnv.
    """
    
    name = "smolagents"
    
    def __init__(self, config: Optional[SmolagentsEnvConfig] = None, **kwargs):
        super().__init__()
        
        self.config = config or SmolagentsEnvConfig()
        self.kwargs = kwargs
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.config.verbose:
            self.logger.disabled = True

        # Silence noisy HTTP client logs (e.g., INFO:httpx:HTTP Request: ...)
        # Keep warnings/errors, but hide info-level request/response chatter.
        for _name in ("httpx", "httpcore"):
            _net_logger = logging.getLogger(_name)
            _net_logger.setLevel(logging.WARNING)
        
        # Environment state
        self.task: Optional[Dict[str, Any]] = None
        self.task_id: Optional[str] = None
        self.observation: str = ""
        self.done: bool = False
        self.reward: float = 0.0
        self.info: Dict[str, Any] = {}
        self.trajectory: List[Dict[str, Any]] = []
        self.num_interactions: int = 0

        # Smolagents specific state  
        self.additional_authorized_imports = []
        self.python_executor = LocalPythonExecutor(
            additional_authorized_imports=self.additional_authorized_imports,
        )
        tools = [WikipediaRetrieverTool()]
        self.tools = {tool.name: tool for tool in tools}
        self.tools.setdefault("final_answer", FinalAnswerTool())

        self.logger.info(f"Smolagents environment initialized with config: {self.config}")


    def reset(self, seed, task: str) -> str:
        """
        Reset the environment with a new task.
        
        Args:
            seed: Random seed for reproducibility.
            task: Task identifier or description.
        
        Returns:
            Initial observation string.
        """
        if seed is not None:
            all_seed(seed)
        self.task = task
        self.observation = ""
        self.done = False
        self.reward = 0.0
        self.info = {}
        self.trajectory = []
        self.num_interactions = 0

        self.task_completed_flag = False

        # Setup retriever tool in executor
        self.python_executor.send_tools({**self.tools})

        # TODO: set initial observation based on the task
        
        self.logger.info(f"Environment reset with task")
        self.logger.info(f"Task instruction: {self.task}")

        return self.observation

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute a step in the environment with the given action.
        
        Args:
            action: Action string to execute.
        
        Returns:
            Tuple of (observation, reward, done, info).
        """
        self.logger.debug(f"Executing action: {action}")

        self.num_interactions += 1

        # Check for max interactions
        if self.num_interactions >= self.config.max_interactions:
            self.done = True
            self.observation = f"Maximum interactions ({self.config.max_interactions}) reached."
            self.reward = 0.0
            self.info = {"success": False, "reason": "max_interactions"}
            return self.observation, self.reward, self.done, self.info
        
        action = self._clean_code(action)
        # Execute the action using the Python executor
        try:
            code_output = self.python_executor(action)
            if len(code_output.logs) > 0:
                self.logger.debug("Execution logs:\n%s", code_output.logs)
            observation = "Execution logs:\n" + code_output.logs

            truncated_output = truncate_content(str(code_output.output))
            observation += "Last output from code snippet:\n" + truncated_output

            if not code_output.is_final_answer:
                self.logger.debug("Out: %s", truncated_output)

            self.observation = observation

            if code_output.is_final_answer:
                self.done = True
                self.task_completed_flag = True
                self.reward = 1.0
                final_answer = observation.replace("Execution logs:\nLast output from code snippet:\n", "")
                self.info = {"success": True, "reason": "final_answer", "final_answer": final_answer}
            else:
                remaining_step = self.config.max_interactions - self.num_interactions
                self.observation += f"\n[Info] Remaining steps: {remaining_step}"
                self.done = False
                self.reward = 0.0
                self.info = {"success": False, "reason": "in_progress"}
            
            self.trajectory.append({
                "action": action,
                "observation": self.observation,
                "reward": self.reward,
                "done": self.done,
                "info": self.info
            })
                        
            return self.observation, self.reward, self.done, self.info
        
        except Exception as e:
            observation = ""
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    observation = "Execution logs:\n" + execution_logs
                    self.logger.info("Execution logs on error:\n%s", execution_logs)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.warning(
                    "Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent."
                )
            observation += f"\nError executing code: {error_msg}"
            # return observation, 0.0, True, {"error": str(e)} 
            return observation, 0.0, False, {"error": str(e)}
    
    def _clean_code(self, code: str) -> str:
        code_action = parse_code_blobs(code, ("```python", "```"))
        code_action = fix_final_answer_code(code_action)
        return code_action

    def task_completed(self) -> bool:
        """Check if the current task has been completed."""
        return self.task_completed_flag
    
    def execute(self, code: str) -> str:
        """
        Execute code and return output (smolagents-style interface).
        
        This method provides compatibility with the smolagents interface.
        """
        obs, reward, done, info = self.step(code)
        return obs
    
    def render(self, mode: str = 'text') -> str:
        """Render the current environment state."""
        return self.observation
    
    def close(self):
        """Clean up environment resources."""
        self.logger.info("Smolagents environment closed")

    def dump_history(self, output_dir: str):
        """Save trajectory history to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, "smolagents_trajectory.json"), 'w') as f:
            json.dump({
                "task_instruction": self.task,
                "num_interactions": self.num_interactions,
                "completed": self.task_completed_flag,
                "final_reward": self.reward,
                "trajectory": self.trajectory
            }, f, indent=2)

        with open(os.path.join(output_dir, "env_history.json"), 'w') as f:
            json.dump(self.trajectory, f, indent=2)
        
        self.logger.info(f"Trajectory saved to {output_dir}/env_history.json")