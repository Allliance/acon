
from productive_agents.llm import ChatGPT, Gemini, vLLM, AzureOpenAIServerModel
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yaml
import os

def load_openai_key_from_config() -> Optional[str]:
    """Load OpenAI API key from private_config.yaml"""
    config_paths = [
        "configs/private_config.yaml",
        "../configs/private_config.yaml", 
        "../../configs/private_config.yaml",
        os.path.expanduser("~/Projects/productive-agents-submission/configs/private_config.yaml")
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and 'openai_key' in config:
                        return config['openai_key']
        except Exception as e:
            continue
    
    # Try environment variable as fallback
    return os.getenv('OPENAI_API_KEY')

class LLMManager:
    """Manages LLM initialization and inference."""
    
    @staticmethod
    def create_llm(model_name: str, key: str, system_message: str, lora_name: Optional[str] = None):
        """Create appropriate LLM instance based on model name."""
        if any(pattern in model_name for pattern in ['gpt', 'o1', 'o3', 'o4']):
            if model_name.startswith('azure/'):
                model_name = model_name.split('/')[1]
            
            # Load OpenAI key from config instead of using the key parameter
            openai_key = load_openai_key_from_config()
            if not openai_key:
                raise ValueError("OpenAI API key not found in config file or environment variable")
            
            return ChatGPT(model_name, openai_key, system_message)
        elif 'gemini' in model_name:
            return Gemini(model_name, key, system_message)
        else:
            return vLLM(model_name, system_message, lora_name=lora_name)

# dataclass for managing output from agent.forward
@dataclass
class LLMOutput:
    """Holds the output from an agent's forward method."""
    action: Any
    response: str
    metadata: Dict[str, Any] = None

