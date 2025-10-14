import json
import os
import re
import shutil
import traceback
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
from termcolor import cprint
import logging

from productive_agents.env.base import BaseEnv
from productive_agents.utils import all_seed

from .shell_executor import execute_shell_command
from .config import OfficeBenchEnvConfig
from . import apps

TIMEOUT_DURATION = 30
ACTION_EXEC = "action_is_effective"

class timeout:
    def __init__(self, seconds=10, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
        self.executor = None

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        return self

    def __exit__(self, type, value, traceback):
        if self.executor:
            self.executor.shutdown(wait=False)

    def run_with_timeout(self, func, *args, **kwargs):
        future = self.executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=self.seconds)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(self.error_message)


class OfficeBenchEnv(BaseEnv):
    """Gym environment for bash shell"""
    name = "officeagent_local_bash"

    def __init__(self, config=None, **kwargs):
        self.kwargs = kwargs
        self.container = None

        self.config = config or OfficeBenchEnvConfig()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.task_dir = getattr(config, "task_dir", None)
        self.workdir = getattr(config, "local_workdir", "./")
        self.exp_config = config

        self.available_apps = apps.AVAILABLE_APPS
        self.available_actions = apps.AVAILABLE_ACTIONS

        if "scratchpad" in apps.AVAILABLE_APPS: # and not self.exp_config.experiment.use_scratchpad:
            # Remove scratchpad app - it gets special treatment
            del apps.AVAILABLE_APPS["scratchpad"]
            del apps.AVAILABLE_ACTIONS["scratchpad"]

        if "llm" in self.available_apps: # remove llm app as it is distracted
            del self.available_apps["llm"]
            del self.available_actions["llm"]

        if "python" in self.available_apps: # remove python app as it is distracted
            del self.available_apps["python"]
            del self.available_actions["python"]
        
        if "verbose" not in self.kwargs or self.kwargs["verbose"] != True:
            self.logger.disabled = True

        self.tool_mode = False
        self.preprocess = None

        self.task = getattr(config, "task", '<undefined>')
        self.current_app = None
        self.history = [] # This is the history which can be used for llm input
        self.history_log = [] # This is the history which will be dumpped for logging
        if os.path.exists(os.path.join(self.workdir, "scratchpad.txt")):
            logging.info("Deleting scratchpad file")
            os.remove(os.path.join(self.workdir, "scratchpad.txt"))

        self.logger = logging.getLogger(__name__)

        BaseEnv.__init__(self)

    def reset(self, index: int = None) -> Tuple[str, Dict]:
        """
        Create new session and reset environment variables

        Args:
            index (`int`) - index of query, gold pair to use for new session. If None, random index is used.
        """
        # Reset instance variables
        self.info = {}
        self.trajectory = []
        self.observation = None
        
        # Set query, gold command
        if not self.tool_mode:
            self.logger.info("-------------\nNew task episode initialized")
            self.observation = None
            self.reward = None
        else:
            self.logger.info("-------------\nExecution Environment Reset")


        # Run preprocess function if provided
        if self.preprocess is not None:
            preprocess_cmds = self.preprocess(self.record)
            for cmd in preprocess_cmds:
                self.exec_action(cmd)
                if not self.info[ACTION_EXEC]:
                    raise RuntimeError(f"Preprocess command failed to execute successfully: {self.preprocess(self.record)}")
        
        return self.observation, self.info

    def add_to_history(self, action, observation, thinking_string=None):
        """
        Add action and observation to history.
        If thinking_string is provided, it will be added as well.
        """
        # For now, we'll store action and observation as a tuple
        # This follows the pattern from the blueprint local_env implementation
        self.history.append((action, observation))
        self.history_log.append((action, observation))

    def dump_history(self, output_dir):
        """
        Dump the action/observation history to a JSON file.
        """
        with open(f"{output_dir}/env_history.json", "w") as f:
            json.dump(self.history_log, f, indent=2)

    def render(self):
        return self.observation

    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        return self.exec_action(action)

    def exec_action(self, action_string: str) -> None:
        self.observation = None
        self.thinking_string = None
        try:
            # The new prompt requires the LLM to return <think>...</think><action>[JSON payload]</action>
            # parse out the thinking and action tokens. There may be whitespace between /think and action.
            if "<think>" in action_string:
                match = re.match(r"<think>(.*?)</think>\s*<action>(.*)</action>", action_string, re.DOTALL)
                if match:
                    thinking_string, action_string = match.groups()
                else:
                    raise ValueError(f"Malformed action string:{action_string}. Expected <think>...</think><action>[JSON payload]</action>")
            else:
                # Find { ... } in the action_string
                match = re.search(r'\{.*?\}', action_string, re.DOTALL)
                if not match:
                    raise ValueError(f"Malformed action string: {action_string}. Expected JSON payload in {{...}} format.")
                action_string = match.group(0)
                action_string = action_string.strip()
                thinking_string = ""
            # strip markdown if present
            if action_string.startswith("```json") and action_string.endswith("```"):
                action_string = re.match(r'```json(.*)```', action_string, re.DOTALL).group(1)
            action = json.loads(action_string)
            action = self._minor_action_fix(action)
            assert self.check_valid_action(action)
            self.action = action

            # special case for switch app
            if action['action'] == 'switch_app':
                action['app'] = 'system'
            elif len(action['action'].split('.')) > 1:
                action['app'] = action['action'].split('.')[0]
                action['action'] = action['action'].split('.')[1]
                self.current_app = action['app']
            is_cd_flag = False
            if action["app"] == "scratchpad":
                assert action["action"] == "write" and action["content"] is not None
                from apps.scratchpad_app.scratchpad import scratchpad_write
                scratchpad_write(self.workdir, action["content"])
                self.observation = f"Successfully wrote to scratchpad."
                command = None
            elif action["app"] == "shell":
                if 'command' not in action:
                    raise ValueError("Missing command in json request. Use the format specified: {\"app\": \"shell\", \"action\": \"command\", \"command\": \"<command>\"}")
                command = action["command"]
                if isinstance(command, list):
                    command = ' '.join(command)
                is_cd_flag = command.startswith("cd")
                if is_cd_flag:
                    # TODO: What if multiple commands on one line w/ `cd` as first one?
                    cd_arg = command[command.index("cd ")+3:].strip()
                    new_path = self.simplify_path(self.workdir, cd_arg)
                    command = f"cd {new_path}"
            elif action["app"] == "system":
                if action["action"] == "switch_app":
                    if "target_app" not in action:
                        raise ValueError("Missing target_app in json request. Use the json format specified: {\"app\": \"system\", \"action\": \"switch_app\", \"target_app\": \"<target_app>\"}")
                    if action["target_app"] not in self.available_apps:
                        raise ValueError(f"App {action['target_app']} not available")
                    self.current_app = action["target_app"]
                    def format_action_list():
                        return "\n".join([f"- {action}" for action in self.available_actions[self.current_app].keys()])
                    self.observation = f"Successfully switched to app: {self.current_app}. Available actions:\n{format_action_list()}"
                elif action["action"] == "finish_task":
                    answer = action.get("answer", 'None')                    
                    output_path = os.path.join(self.task_dir, "testbed", "data", "answer.txt")
                    self._write_answer(answer, output_path)
                    self.observation = "Task finished"
                elif action["action"] == 'got_stuck':
                    answer = 'None'
                    self.observation = "Task failed"
                command = None
            else:
                # TODO: config flag for quick actions
                if self.current_app != action["app"]:
                    command = None
                    self.observation = f"Error: you must switch to the {action['app']} app before executing this action. Use the switch_app action to switch apps."
                elif action["action"] in self.available_actions[action["app"]]:
                    action_module = self.available_actions[action["app"]][action["action"]]
                    command = action_module.construct_action(self.workdir, args=action)
                else:
                    command = None
                    self.observation = f"Error: Action {action['action']} not available in app {action['app']}"

            if command is not None:
                def execute_command():
                    cleaned_cmd = self.clean_cmd(command)
                    return execute_shell_command(cleaned_cmd, verbose=True)

                with timeout(seconds=TIMEOUT_DURATION) as t:
                    try:
                        exit_code, std_output, std_error = t.run_with_timeout(execute_command)
                        self.observation = std_output.decode("utf-8").split('OBSERVATION:')[-1].strip()

                        cprint(100*'+', color='yellow', attrs=['bold'])
                        cprint(f'OBSERVATION: {self.observation}', color='yellow', attrs=['bold'])
                        cprint(100*'+', color='yellow', attrs=['bold'])
                        self.info[ACTION_EXEC] = exit_code == 0
                    except TimeoutError:
                        self.observation = f"Command timed out after {TIMEOUT_DURATION} seconds: {command}"
                        self.info[ACTION_EXEC] = False

                if is_cd_flag and self.info[ACTION_EXEC]:
                    self.workdir = Path(new_path)

                if action["app"] == "shell":
                    if not self.info[ACTION_EXEC]:  # Command failed
                        self.observation = f"Command failed with exit code {exit_code}: {command}. The error was [{std_error.decode('utf-8')}]."
                    elif self.observation == "":  # Command succeeded but no output
                        self.observation = f"Successfully executed command: {command}. The output was [{std_output.decode('utf-8')}]."
            
            # Add action and observation to history
            self.add_to_history(action_string, self.observation)
        except Exception as e:
            cprint('!!!!!!!!', color='red')
            cprint(e, color='red')
            # print the stack trace            
            traceback.print_exc()
            cprint(f"Attempted action: {action_string}", color='red')
            cprint('!!!!!!!!', color='red')
            self.observation = f"Error: [{self.current_app}] {e}"
            if isinstance(e, json.JSONDecodeError):
                self.observation += " Malformed action! You must follow the given action JSON format!"
            self.info[ACTION_EXEC] = False
            if isinstance(e, TimeoutError) and self.current_app == "ocr": # bail aggressively on ocr timeouts which are likely auth failures
                raise e
        
        if self.observation == "Task finished":
            return self.observation, 1.0, True, self.info
        elif self.observation == "Task failed":
            return self.observation, 0.0, True, self.info

        return self.observation, 0.0, False, self.info
    
    def evaluate_output(self, output_dir):
        """
        Evaluate the output of a task according to its evaluation config.
        
        Args:
            output_dir: Directory containing the output to evaluate
            
        Returns:
            Whether the evaluation passed
        """
        
        try:
            eval_config = self.task_config['evaluation']
            for eval_item in eval_config:
                function = eval_item['function']
                args = eval_item['args']
                if not eval(f"{function}(output_dir, args)"):
                    return False
            return True
        except Exception as e:
            print(f"Error evaluating {self.task}.{self.subtask}: {e}")
            return False

    def _minor_action_fix(self, action):
        proc_action = {}
        for k, v in action.items():
            if isinstance(v, list) and len(v) == 1:
                proc_action[k] = v[0]
            else:
                proc_action[k] = v
        return proc_action

    def check_valid_action(self, action: dict) -> bool:
        """Checks if action is valid"""
        try:
            assert 'app' in action and 'action' in action
            return True
        except AssertionError:
            return False
    
    def get_available_actions(self) -> List:
        current_app = self.current_app
        available_actions = list(apps.AVAILABLE_ACTIONS[current_app].keys())
        return available_actions

    def simplify_path(self, current: str, changed: str) -> str:
        """Resolves path from current working directory path and the argument of the `cd` command"""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""

        path = []
        
        for segment in (current + "/" + changed).split("/"):
            if segment == "..":
                if path:
                    path.pop()
            elif segment and segment != ".":
                path.append(segment)

        return "/" + "/".join(path)
    
    def get_reward(self) -> Tuple[float, Dict]:
        
        print("======================================================")
        print('get_reward called')
        print("======================================================")
        return 0.0, {}

    def close(self):
        # Not using docker, so no need to stop container
        pass
    
    def clean_cmd(self, action: str) -> str:
        """Cleans action string"""
        entrypoint = "/bin/bash" 
        if self.current_app == "calendar" or self.current_app == "email":
            action = action.replace("\"", "\'")
            action += f" --workdir {self.task_dir}"
        command = '{} -c """ {} """'.format(entrypoint, action.strip())
        print('COMMAND:', command)
        return command
    
    def _write_answer(self, answer: str, file_path: str):
        answer = str(answer)
        answer = answer.replace('"', '').replace("'", '')
        
        try:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(answer)
                return 
        except Exception as e:
            print(f"Write Answer: Failed to write answer to container: {e}")