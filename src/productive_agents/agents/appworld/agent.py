import json
import logging
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from jinja2 import Template, Environment, FileSystemLoader

from productive_agents.agents.unified_agent import UnifiedAgent, UnifiedPromptBuilder, UnifiedActionProcessor
from productive_agents.agents.utils import LLMOutput
from .config import AppWorldAgentConfig

# AppWorld prompt template based on the minimal example
PROMPT_TEMPLATE = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

For example, consider the task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

What is the password for my Spotify account?

ASSISTANT:
# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

USER:
[
  {
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  },
  {
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  },
  ...
  {
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  },
  {
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  },
  ...
]


ASSISTANT:
# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))


USER:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]


ASSISTANT:
# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

USER:
{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {
    'success': [{'account_name': 'string', 'password': 'string'}],
    'failure': {'message': 'string'}
  }
}

ASSISTANT:
# Okay, it requires no arguments. So I can just call it directly.
print(apis.supervisor.show_account_passwords())

USER:
[
  {
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  },
  {
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  },
  ...
]


ASSISTANT:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)


USER:
dummy_spotify_pass

ASSISTANT:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)


USER:
Marked the active task complete.


----------------------------------------------

USER:
**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

{{ instruction }}
"""

# Default prompt dict - will be loaded from external files
DEFAULT_PROMPT_DICT = {
    "system_message": """You are an AI assistant that writes Python code to complete tasks. You should respond with clear, executable Python code to interact with APIs and solve the given task completely autonomously.""",
    "main_prompt": PROMPT_TEMPLATE
}

class AppWorldPromptBuilder(UnifiedPromptBuilder):
    """Prompt builder for AppWorld tasks."""
    
    def build_prompt(self, env, context_sections: List[str]) -> str:
        """Build complete prompt for AppWorld task."""
        context_prefix = ''.join(context_sections)
        
        # Build the first user prompt with task details
        if len(env.trajectory) == 0:
            task_instruction = env.task.instruction if env.task else "No task specified"
            supervisor = env.task.supervisor if env.task else {}
            
            # Load the main prompt template
            template_path = self.prompt_dict.get("main_prompt_template")
            if template_path:
                # Load from external Jinja template
                template = self._load_jinja_template(template_path)
                dictionary = {"supervisor": supervisor, "instruction": task_instruction}
                main_prompt = template.render(dictionary)
            else:
                # Fallback to inline template if available
                main_prompt_text = self.prompt_dict.get("main_prompt", "No task instruction provided.")
                dictionary = {"supervisor": supervisor, "instruction": task_instruction}
                main_prompt = Template(main_prompt_text.lstrip()).render(dictionary)
            
            return context_prefix + main_prompt
        else:
            # Return the latest observation for continuing conversation
            return env.observation if env.observation else ""


class AppWorldActionProcessor(UnifiedActionProcessor):
    """Handles processing of code responses for AppWorld."""
    
    def extract_action(self, response: str) -> str:
        """Extract Python code from LLM response.
        Removes optional <think>...</think> markers emitted by some models before further parsing.
        """
        # Strip any <think> tags while preserving their inner content
        response = self._remove_think_markers(response)
        
        # Remove any markdown formatting
        code = self._clean_markdown(response)
        
        # If the response is already clean code, return it
        if self._is_clean_code(code):
            return code
        
        # Try to extract code from various patterns
        patterns = [
            r'```python\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`(.*?)`',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if extracted:
                    return extracted
        
        # If no code blocks found, return the whole response cleaned
        return code.strip()
    
    def _remove_think_markers(self, text: str) -> str:
        """Remove <think> and </think> tags, keeping inner content intact.
        Handles variations in capitalization and any attributes (just in case), e.g. <think reason="x">.
        """
        # Remove opening <think ...> and closing </think> tags only
        text = re.sub(r'<\s*think[^>]*>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\s*/\s*think\s*>', '', text, flags=re.IGNORECASE)
        return text
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        # Remove markdown code block markers
        text = re.sub(r'^```python\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```.*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def _is_clean_code(self, text: str) -> bool:
        """Check if text appears to be clean Python code."""
        # Simple heuristic: if it contains Python keywords and no markdown
        python_keywords = ['def', 'import', 'print', 'if', 'for', 'while', 'class']
        has_keywords = any(keyword in text for keyword in python_keywords)
        no_markdown = '```' not in text
        return has_keywords and no_markdown


class AppWorldAgent(UnifiedAgent):
    """
    AppWorld Agent for code-based task completion.
    
    This agent specializes the unified agent framework for AppWorld tasks,
    handling Python code generation and execution in the AppWorld environment.
    """
    
    def __init__(
        self, 
        model_name: str, 
        key: str, 
        env, 
        task_config: Dict[str, Any], 
        llm_cache: Optional[Dict] = None,
        debug_mode: bool = False,
        exp_config: Optional[AppWorldAgentConfig] = None,
        lora_name: Optional[str] = None,
        model_ctxopt: Optional[Any] = None,
        **kwargs
    ):
        """Initialize AppWorld agent."""
        # Validate and set config
        if exp_config is None:
            exp_config = AppWorldAgentConfig()
        assert isinstance(exp_config, AppWorldAgentConfig), \
            f"exp_config should be an AppWorldAgentConfig object, but got {type(exp_config)}"
        
        super().__init__(
            model_name=model_name,
            key=key,
            env=env,
            task_config=task_config,
            llm_cache=llm_cache,
            debug_mode=debug_mode,
            exp_config=exp_config,
            lora_name=lora_name,
            model_ctxopt=model_ctxopt,  # Pass pre-initialized model for ctxopt
            **kwargs
        )
    
    def _create_prompt_builder(self) -> AppWorldPromptBuilder:
        """Create AppWorld-specific prompt builder."""
        prompt_dict = DEFAULT_PROMPT_DICT.copy()
        
        if hasattr(self.exp_config, 'prompt_file') and self.exp_config.prompt_file is not None:
            self.logger.info(f"Loading prompt file: {self.exp_config.prompt_file}")
            try:
                with open(self.exp_config.prompt_file, 'r') as f:
                    custom_prompts = json.load(f)
                prompt_dict.update(custom_prompts)
            except Exception as e:
                self.logger.warning(f"Failed to load prompt file {self.exp_config.prompt_file}: {e}")
        else:
            self.logger.info("Using default prompts")
            
        # Pass the current working directory to the prompt builder
        working_dir = os.getcwd()
        return AppWorldPromptBuilder(prompt_dict, working_dir)
    
    def _create_action_processor(self) -> AppWorldActionProcessor:
        """Create AppWorld-specific action processor."""
        return AppWorldActionProcessor(self.logger)
    
    def _process_response(self, response: str) -> str:
        """Process LLM response to extract code."""
        return self.action_processor.extract_action(response)
    
    def _get_workflow_memory(self) -> Optional[str]:
        """Get workflow memory content specific to AppWorld."""
        if hasattr(self.exp_config, 'workflow_memory') and self.exp_config.workflow_memory:
            return self.exp_config.workflow_memory
        return None
    
    def _build_workflow_memory_content(self, workflow_memory: str) -> str:
        """Build workflow memory content section for AppWorld."""
        return (f"# Workflow memory\n"
                f"These are common, reusable workflows for AppWorld tasks:\n\n"
                f"{workflow_memory}\n\n")
    
    def _determine_success(self, env, reward: float, info: Dict) -> bool:
        """Determine if AppWorld task was successful."""
        if hasattr(env, 'task_completed') and env.task_completed():
            return True
        return super()._determine_success(env, reward, info)
    

# Factory function for creating AppWorld agents
def create_appworld_agent(
    model_name: str, 
    key: str, 
    env, 
    task_config: Dict[str, Any], 
    **kwargs
) -> AppWorldAgent:
    """Factory function to create an AppWorld agent."""
    exp_config = kwargs.pop('exp_config', AppWorldAgentConfig())
    return AppWorldAgent(
        model_name=model_name,
        key=key,
        env=env,
        task_config=task_config,
        exp_config=exp_config,
        **kwargs
    )
