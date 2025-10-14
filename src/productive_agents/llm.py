"""
LLM Interface Module

This module provides a unified interface for various LLM services with a clean inheritance hierarchy:

Classes:
- BaseLLMModel: Abstract base class with common functionality for all LLM models
- SimpleLLM: Basic Azure-based LLM for simple tasks  
- AzureOpenAIServerModel: Azure OpenAI service via AzureApiWrapper
- ChatGPT: OpenAI API with API key authentication
- Gemini: Google's Gemini API
- vLLM: Local vLLM server interface
- vLLMLocal: Direct local vLLM interface

The base class handles:
- Logging configuration
- Temperature/token parameter management for different model types (o1, o3, o4 vs others)
- Message building (handling system messages appropriately for different models)
- Error handling and retry logic
- Consistent interface across all implementations

All classes now use a simple generate() method instead of the complex generate_plus_with_score().
"""

from openai import OpenAI
import time
import numpy as np
# import google.generativeai as genai
from mimetypes import guess_type
from .subtrate_api import AzureApiWrapper
try:
    from .subtrate_api import ResponsibleAIPolicyViolationError
except Exception:
    class ResponsibleAIPolicyViolationError(Exception):
        pass
import base64
import logging
import os

# Set up a dedicated logger for this module
logger = logging.getLogger('llm')
logger.setLevel(logging.INFO)

# Configure logger with file and console output
def configure_logger(log_file=None, log_level=logging.INFO):
    """
    Configure the logger with optional file output
    
    Args:
        log_file: Optional path to a log file
        log_level: Logging level (default: INFO)
    """
    global logger
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if a log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Add file handler (append mode)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

# Initialize with default console handler
if not logger.handlers:
    configure_logger()
    
# Function to set up logging from outside this module
def set_log_file(log_file, log_level=logging.INFO):
    """
    Set up logging to a file from outside the module.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (default: INFO)
    """
    configure_logger(log_file=log_file, log_level=log_level)
    return logger

# Model pricing information (per 1M tokens) - updated as of 2024
MODEL_PRICING = {
    'gpt-5': {'input': 1.25, 'output': 10.00},
    'gpt-5-mini': {'input': 0.25, 'output': 2.00},
    'gpt-4.1': {'input': 2.00, 'output': 8.00},
    'gpt-4.1-mini': {'input': 0.8, 'output': 3.2},
    'o3': {'input': 2.0, 'output': 8.0},
    'o4-mini': {'input': 1.10, 'output': 4.40},
}

def calculate_api_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the API cost for given token usage.
    
    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    # Find matching model pricing
    pricing = None
    for model_key in MODEL_PRICING:
        if model_key in model_name.lower():
            pricing = MODEL_PRICING[model_key]
            break
    
    if pricing is None:
        # Default to gpt-4o pricing if model not found
        logger.warning(f"No pricing found for model {model_name}, using gpt-4o pricing")
        pricing = MODEL_PRICING['gpt-4o']
    
    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing['input']
    output_cost = (output_tokens / 1_000_000) * pricing['output']
    total_cost = input_cost + output_cost
    
    return total_cost

def _setup_logging(log_file, log_level):
    """Helper method to configure logging from model classes"""
    if log_file:
        configure_logger(log_file=log_file, log_level=log_level)
        logger.info(f"LLM module logging to: {log_file}")
    
class BaseLLMModel:
    """Base class for all LLM models with common functionality"""
    
    def __init__(self, model_name, system_message=None, log_file=None, log_level=logging.INFO, temperature=0.0):
        # Configure logging
        _setup_logging(log_file, log_level)
        
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = self._get_temperature(model_name, temperature)
        
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
    
    def _get_temperature(self, model_name, temperature):
        """Get appropriate temperature for model"""
        if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
            return 1.0
        return temperature
    
    def _get_max_token_key(self, model_name):
        """Get appropriate max token key for model"""
        if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name or 'gpt-5' in model_name:
            return 'max_completion_tokens'
        return 'max_tokens'
    
    def _build_messages(self, prompt):
        """Build messages array for API call"""
        messages = []
        
        # Add system message for non-o1 models
        if 'o1' not in self.model_name and self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        # Add user message
        if type(prompt) == str:
            user_content = prompt
            if 'o1' in self.model_name and self.system_message:
                # For o1 models, prepend system message to user content
                user_content = f"{self.system_message}\n\n{prompt}"
        
            messages.append({
                "role": "user", 
                "content": user_content
            })
        elif isinstance(prompt, list):
            # append in user, assistant, user, assistant format
            if isinstance(prompt[0], str):
                for i, msg in enumerate(prompt):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({
                        "role": role,
                        "content": msg
                    })
            elif isinstance(prompt[0], dict):
                messages.extend(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings")
        
        return messages
    
    def _handle_api_error(self, error, retry_num, retry_limit=2):
        """Handle API errors with retry logic"""
        logger.error(f"{self.__class__.__name__}: {str(error)}")
        
        # No retries on Azure policy violations
        if isinstance(error, ResponsibleAIPolicyViolationError):
            return "RESPONSIBLE_AI_POLICY_VIOLATION"

        if "This model's maximum context length is" in str(error):
            logger.error(f"Context length exceeded: {error}")
            return "CONTEXT_LENGTH_EXCEEDED"
        elif "invalid_prompt" in str(error):
            return "INVALID_PROMPT"
        elif retry_num >= retry_limit:
            return f"MAX_RETRIES_EXCEEDED: {error}"
        else:
            time.sleep(60)
            return None  # Continue retrying
    
    def get_model_options(self, temperature=None, max_tokens=1024, top_p=1.0, n=1, seed=42):
        """Get model options for API call"""
        temp = temperature if temperature is not None else self.temperature
        max_token_key = self._get_max_token_key(self.model_name)
        
        return {
            max_token_key: max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "n": n,
            "seed": seed,
        }
    
    def generate(self, prompt, **kwargs):
        """Generate response - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate method")
    
    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        """
        Backward compatibility method that wraps generate().
        Returns [(response, fake_confidence_score)] for compatibility.
        """
        # Convert old-style options to new-style kwargs
        kwargs = {}
        if options:
            kwargs.update({
                'max_tokens': options.get('max_tokens', options.get('per_example_max_decode_steps', 1024)),
                'temperature': options.get('temperature', None),
                'n': options.get('n', options.get('n_sample', 1))
            })
        if end_str:
            kwargs['stop'] = end_str
            
        response = self.generate(prompt, **kwargs)
        # Return in the old format with fake confidence score
        return [(response, np.log(1.0))]
    
    def set_system_message(self, system_message):
        """Set system message"""
        self.system_message = system_message
    
    def _update_token_usage(self, input_tokens: int, output_tokens: int):
        """Update token usage statistics"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {self.get_total_tokens()}")
    
    def get_total_tokens(self):
        """Get total tokens used"""
        return self.total_input_tokens + self.total_output_tokens
    
    def get_token_usage_stats(self):
        """Get detailed token usage statistics"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.get_total_tokens(),
            'total_requests': self.total_requests,
            'model_name': self.model_name
        }
    
    def get_total_cost(self):
        """Get total API cost for this model instance"""
        return calculate_api_cost(self.model_name, self.total_input_tokens, self.total_output_tokens)
    
    def get_cost_breakdown(self):
        """Get detailed cost breakdown"""
        total_cost = self.get_total_cost()
        stats = self.get_token_usage_stats()
        
        return {
            **stats,
            'total_cost_usd': total_cost,
            'input_cost_usd': calculate_api_cost(self.model_name, self.total_input_tokens, 0),
            'output_cost_usd': calculate_api_cost(self.model_name, 0, self.total_output_tokens),
        }


class SimpleLLM(BaseLLMModel):
    def __init__(self, service='azure', model_name='dev-phi-35-mini-instruct', is_embedding=False, log_file=None, log_level=logging.INFO):
        super().__init__(model_name, log_file=log_file, log_level=log_level)
        
        self.is_embedding = is_embedding
        assert service == 'azure'
        
        sampling_params = self.get_model_options(max_tokens=4096)
        self.llm_model = AzureApiWrapper(model_name, sampling_params)

    def generate_response(self, prompt='hello'):
        """Generate response - kept for backward compatibility"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_model.get_response(messages)
        logger.info(f"SimpleLLM response: {response}")
        
        if self.is_embedding:
            return response['data'][0]['embedding']
        else:
            return response[1]
    
    def generate(self, prompt, **kwargs):
        """Generate response using the standard interface"""
        return self.generate_response(prompt)


class AzureOpenAIServerModel(BaseLLMModel):
    def __init__(self, model_name, system_message=None, log_file=None, log_level=logging.INFO, temperature=0.0):
        super().__init__(model_name, system_message, log_file, log_level, temperature)
        
        sampling_params = self.get_model_options(max_tokens=4096)
        self.llm_model = AzureApiWrapper(model_name, sampling_params)

    def generate(self, prompt, max_tokens=1024, temperature=None, seed=None, **kwargs):
        """Generate response using Azure API wrapper"""
        messages = self._build_messages(prompt)
        
        retry_num = 0
        retry_limit = 4

        while retry_num <= retry_limit:
            try:
                response = self.llm_model.get_response(messages, **kwargs)
                # AzureApiWrapper returns [response_object, text_content]
                if isinstance(response, (list, tuple)) and len(response) > 1:
                    response_obj, text_content = response[0], response[1]
                    
                    # Try to extract token usage from response object
                    if hasattr(response_obj, 'usage') and response_obj.usage:
                        input_tokens = response_obj.usage.prompt_tokens
                        output_tokens = response_obj.usage.completion_tokens
                        self._update_token_usage(input_tokens, output_tokens)
                    
                    return text_content
                else:
                    # Handle different response formats
                    if hasattr(response, 'choices') and response.choices:
                        # Try to extract token usage
                        if hasattr(response, 'usage') and response.usage:
                            input_tokens = response.usage.prompt_tokens
                            output_tokens = response.usage.completion_tokens
                            self._update_token_usage(input_tokens, output_tokens)
                        return response.choices[0].message.content
                    elif isinstance(response, dict) and 'choices' in response:
                        # Try to extract token usage from dict format
                        if 'usage' in response:
                            usage = response['usage']
                            if 'prompt_tokens' in usage and 'completion_tokens' in usage:
                                input_tokens = usage['prompt_tokens']
                                output_tokens = usage['completion_tokens']
                                self._update_token_usage(input_tokens, output_tokens)
                        return response['choices'][0]['message']['content']
                    else:
                        return str(response)
                        
            except Exception as e:
                error_result = self._handle_api_error(e, retry_num, retry_limit)
                if error_result:
                    if error_result == "RESPONSIBLE_AI_POLICY_VIOLATION":
                        # Surface the base message without further retries
                        raise Exception("ResponsibleAIPolicyViolation: content filtered by Azure policy")
                    if error_result.startswith("MAX_RETRIES_EXCEEDED"):
                        raise Exception(error_result)
                    return "Error generating response"
                retry_num += 1
        
        return "Error generating response after retries"


class ChatGPT(BaseLLMModel):
    def __init__(self, model_name, key, system_message=None, log_file=None, log_level=logging.INFO, temperature=0.0):
        super().__init__(model_name, system_message, log_file, log_level, temperature)
        
        self.key = key
        self.client = OpenAI(api_key=key)

    def generate(self, prompt, max_tokens=1024, temperature=None, stop=None, seed=None, **kwargs):
        """Generate response using OpenAI API"""
        messages = self._build_messages(prompt)
        options = self.get_model_options(temperature=temperature, max_tokens=max_tokens, n=1, seed=seed if seed is not None else 42)
        
        if stop:
            options['stop'] = stop
        
        retry_num = 0
        retry_limit = 2
        
        while retry_num <= retry_limit:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **options,
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    input_tokens = response.usage.prompt_tokens
                    output_tokens = response.usage.completion_tokens
                    self._update_token_usage(input_tokens, output_tokens)
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_result = self._handle_api_error(e, retry_num, retry_limit)
                if error_result:
                    if error_result == "RESPONSIBLE_AI_POLICY_VIOLATION":
                        raise Exception("ResponsibleAIPolicyViolation: content filtered by Azure policy")
                    if error_result.startswith("MAX_RETRIES_EXCEEDED"):
                        raise Exception(error_result)
                    return "PLACEHOLDER"
                retry_num += 1
        
        return "PLACEHOLDER"


class Gemini(BaseLLMModel):
    def __init__(self, model_name, key, system_message=None, log_file=None, log_level=logging.INFO):
        super().__init__(model_name, system_message, log_file, log_level, temperature=1.0)
        
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt, max_tokens=1024, **kwargs):
        """Generate response using Gemini API"""
        combined_prompt = (
            f"{self.system_message}\n\n{prompt}" if self.system_message else prompt
        )
        response = self.model.generate_content(
            combined_prompt,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=self.temperature,
            ),
        )
        return response.text


class vLLM(BaseLLMModel):
    def __init__(self, model_name, system_message=None, log_file=None, log_level=logging.INFO, lora_name=None):
        super().__init__(model_name, system_message, log_file, log_level)
        
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc",
        )
        self.lora_name = lora_name
        if self.lora_name and "=" in self.lora_name:
            self.lora_name = self.lora_name.split("=")[0] # Take the first part for vLLM
        if self.lora_name:
            self.model_name = self.lora_name

    def generate(self, prompt, n=1, max_tokens=2048, temperature=None, seed=None, **kwargs):
        """Generate response using vLLM server"""
        try:
            messages = self._build_messages(prompt)
            options = self.get_model_options(
                temperature=temperature, 
                max_tokens=max_tokens, 
                n=n,
                seed=seed if seed is not None else 42
            )
            
            if n > 1:
                options["temperature"] = 0.6  # Use higher temp for multiple samples

            options["extra_body"] = {
                "presence_penalty": 0.5, # Default presence penalty
                "chat_template_kwargs": {"enable_thinking": False},
            }  
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **options,
            )
            
            # Track token usage if available
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                self._update_token_usage(input_tokens, output_tokens)
            else:
                # If no usage info, at least track the request
                self.total_requests += 1
            
            if n > 1:
                return [choice.message.content for choice in completion.choices]
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"vLLM generation Error: {e}")
            return "None"

class vLLMLocal(BaseLLMModel):
    def __init__(self, model_name, system_message=None, lora_path=None):
        super().__init__(model_name, system_message)
        
        from transformers import AutoTokenizer
        from vllm import LLM
        self.model_name = model_name
        self.system_message = system_message
        self.llm = LLM(model=self.model_name, enable_lora=lora_path is not None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.lora_path = lora_path

    def register_system_message(self, system_message):
        """Register system message for vLLM"""
        self.system_message = system_message
    
    def get_model_options(
        self,
        temperature=0.7,
        per_example_max_decode_steps=512,
        per_example_top_p=1,
        n_sample=1,
    ):
        return dict(
            temperature=temperature,
            n=n_sample,
            top_p=per_example_top_p,
            max_tokens=per_example_max_decode_steps,
        )

    def generate(self, prompt, seed=None):
        try:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            options = self.get_model_options()
            sampling_params = SamplingParams(**options)
            messages = self._build_messages(prompt)
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            generate_inputs = {
                "prompts": inputs,
                "sampling_params": sampling_params,
            }
            if self.lora_path:
                generate_inputs["lora_request"] = LoRARequest(
                    "finetune",
                    1,
                    self.lora_path,
                )

            completion = self.llm.generate(**generate_inputs)
            return completion[0].outputs[0].text
        except Exception as e:
            print("Generation Error:", e)
            return "None"