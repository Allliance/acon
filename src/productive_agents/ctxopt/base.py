# Base class for context optimization modules
# This base should be extended by observation and history optimizers

import os
import json
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
try:
    import tiktoken
except ImportError:
    tiktoken = None
from jinja2 import Environment, FileSystemLoader


class BaseContextOptimizer(ABC):
    """
    Base class for context optimization modules.
    Provides common functionality for observation and history optimizers.
    """
    
    def __init__(self, config: Dict[str, Any], prompt_dir: str, debug_mode: bool = False):
        """
        Initialize the context optimizer.
        
        Args:
            config: Configuration dictionary
            prompt_dir: Directory containing prompt templates
            debug_mode: Whether to enable debug mode
        """
        self.config = config
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(self.__class__.__name__)
        self.history = []
        
        self.prompt_dir = prompt_dir
        self.llm = None
        self.system_message = None
        self.temperature = config.get("temperature", 0.0)
        
        # Set up tokenizer for token counting (robust to missing/unknown models)
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o") if tiktoken else None
        except Exception:
            # Fallback to cl100k_base if available
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base") if tiktoken else None
            except Exception:
                self.tokenizer = None
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Main processing method that each module must implement.
        """
        pass
    
    def setup_llm_backend(self, llm_instance) -> None:
        """
        Set up the LLM backend for context optimization.
        
        Args:
            llm_instance: The LLM instance to use
        """
        self.llm = llm_instance
    
    def _load_prompt_templates(self) -> Dict[str, Any]:
        """
        Load Jinja2 prompt templates from the prompt directory.
        
        Returns:
            Dictionary of loaded prompt templates
        """
        if not os.path.exists(self.prompt_dir):
            self.logger.warning(f"Prompt directory {self.prompt_dir} does not exist")
            return {}
            
        prompt_env = Environment(loader=FileSystemLoader(self.prompt_dir))
        templates = {}
        
        # Load all .jinja files in the prompt directory
        for filename in os.listdir(self.prompt_dir):
            if filename.endswith('.jinja'):
                template_name = filename[:-6]  # Remove .jinja extension
                templates[template_name] = prompt_env.get_template(filename)
                
        return templates
    
    def construct_system_message(self, template_name: str = "system_prompt") -> str:
        """
        Construct the system message using a template.
        
        Args:
            template_name: Name of the template to use
            
        Returns:
            Rendered system message
        """
        if template_name in self.prompt_templates:
            return self.prompt_templates[template_name].render()
        else:
            self.logger.warning(f"System template {template_name} not found")
            return ""
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the given arguments.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Arguments to pass to the template
            
        Returns:
            Rendered template string
        """
        if template_name in self.prompt_templates:
            return self.prompt_templates[template_name].render(**kwargs)
        else:
            raise ValueError(f"Template {template_name} not found")
    
    def parse_output(self, response: str, keyword: str) -> str:
        """
        Parse LLM output to extract content after a specific keyword.
        
        Args:
            response: Raw LLM response
            keyword: Keyword to look for in the response
            
        Returns:
            Parsed content after the keyword
        """
        if keyword in response:
            start_idx = response.index(keyword) + len(keyword)
            parsed_content = response[start_idx:].strip()
            return parsed_content
        return response
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        # Fallback to approximate token count (roughly 4 chars per token)
        return max(1, len(text) // 4) if text else 0
    
    def add_to_history(self, system_message: str, prompt: str, response: str, prompt_args: Dict) -> None:
        """
        Add an interaction to the module's history.
        
        Args:
            system_message: System message used
            prompt: User prompt sent
            response: LLM response received
        """
        self.history.append((system_message, prompt, response, prompt_args))
    
    def build_history_text(self, history: List[Tuple], start_idx: int = 0) -> str:
        """
        Build history text from action-observation pairs.
        
        Args:
            history: List of (action, observation) tuples
            start_idx: Index to start building from
            
        Returns:
            Formatted history text
        """
        history_text = ''
        for i, (action, observation) in enumerate(history):
            if i < start_idx:
                continue
            if observation:
                observation = observation.replace('\n', '\\n')
                history_text += f' - Step {i}: {json.dumps(action)} -> [{observation}]\n'
            else:
                history_text += f' - Step {i}: {json.dumps(action)}\n'
        return history_text
    
    def convert_llm_history_to_text(self, history: List[Dict[str, str]]) -> str:
        """
        Convert a list of LLM messages into a formatted text string.
        
        Args:
            history: List of messages in the format [{'role': 'user', 'content': '...'}, ...]
        Returns:
            Formatted history text
        """
        history_text = ''
        for i, message in enumerate(history):
            # Exclude the first user message
            if i == 1 and message['role'] == 'user':
                continue
            role = message['role']
            content = message['content']
            # add as USER: or ASSISTANT: prefix
            if role == 'user':
                history_text += f'USER:\n{content}\n\n'
            elif role == 'assistant':
                history_text += f'ASSISTANT:\n{content}\n\n'
        return history_text.strip()
    
    def llmlingua_compress_context(self, prompt, ratio=0.33):
        r = requests.post("http://localhost:9999/compress", json={"prompt": prompt, "rate": ratio})
        r.raise_for_status()
        return r.json()["compressed_prompt"]    