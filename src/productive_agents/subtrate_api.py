'''
Substrate API for calling the LLM/SLM endpoints, used by the models to make requests
'''

import atexit
import json
import time
import requests
import os
from os.path import exists
import yaml
from msal import PublicClientApplication, SerializableTokenCache
import requests
import json
from abc import ABC, abstractmethod
import logging
from typing import Tuple
import base64
from mimetypes import guess_type
from openai import RateLimitError
try:
    from openai import BadRequestError, APIStatusError
except Exception:  # fallback for older SDKs
    BadRequestError = Exception
    APIStatusError = Exception

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get('DEFAULT_IDENTITY_CLIENT_ID'):
    os.environ["AZURE_CLIENT_ID"] = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')


class ResponsibleAIPolicyViolationError(Exception):
    """Raised when Azure content filter blocks a request with ResponsibleAIPolicyViolation."""
    pass

def get_private_config():
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    private_config_paths = [os.path.join(src_dir, "configs", f'{"private_config"}.yaml'), os.path.join(src_dir, "apps", f'{"private_config"}.yaml')]
    for private_config_path in private_config_paths:
        if exists(private_config_path):
            break
    else:
        raise FileNotFoundError("No private configuration file found.")
    
    with open(private_config_path, "r") as file:
        private_config = yaml.safe_load(file)  # loading the experiment, task and model config
    return private_config

# --- New: model-aware endpoint + api version resolution ---

def _resolve_azure_endpoint_and_version(model_name: str, cfg: dict) -> tuple[str, str]:
    """Return (endpoint, api_version) for a given model_name.
    Resolution priority:
      1. cfg['azure_models'][model_key]
      2. cfg['azure_endpoints'][model_key] + cfg['azure_api_versions'][model_key]
      3. Top-level 'azure_endpoint' + 'azure_api_version'
    Model key mapping:
      gpt-5 -> gpt5
      gpt-4.1-nano -> gpt41nano
      else -> default
    """
    if not cfg:
        raise ValueError("Config dict is empty")

    if 'gpt-5' in model_name:
        model_key = 'gpt5'
    elif 'gpt-4.1-nano' in model_name:
        model_key = 'gpt41nano'
    else:
        model_key = 'default'

    models_map = cfg.get('azure_models') or {}
    if model_key in models_map:
        entry = models_map[model_key] or {}
        endpoint = entry.get('endpoint') or cfg.get('azure_endpoint')
        api_version = entry.get('api_version') or cfg.get('azure_api_version')
        if endpoint and api_version:
            return endpoint, api_version

    endpoint_map = cfg.get('azure_endpoints') or {}
    version_map = cfg.get('azure_api_versions') or {}
    if model_key in endpoint_map or model_key in version_map:
        endpoint = endpoint_map.get(model_key, cfg.get('azure_endpoint'))
        api_version = version_map.get(model_key, cfg.get('azure_api_version'))
        if endpoint and api_version:
            return endpoint, api_version

    return cfg['azure_endpoint'], cfg['azure_api_version']


class AzureOpenAIEmbeddings:
    def __init__(self, model_name="text-embedding-3-large"):
        # Use AzureApiWrapper for embeddings (is_embedding=True)
        self.embedding_model = AzureApiWrapper(model_name, is_embedding=True)
        

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (list[str]): A list of texts to embed.

        Returns:
            list[list[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        msg = {"input": texts}
        response = self.embedding_model.get_response(msg)
        embeddings = [data['embedding'] for data in response['data']]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Generates an embedding for a single query.

        Args:
            text (str): The query to embed.

        Returns:
            list[float]: The embedding as a list of floats.
        """
        msg = {"input": [text]}
        response = self.embedding_model.get_response(msg)
        return response['data'][0]['embedding']
    
    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

class ApiWrapper:
    def __init__(self, model_name, sampling_params=None, is_embedding=False):
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.is_embedding = is_embedding
        self.max_token_key = 'max_completion_tokens' \
            if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name or 'gpt-5' in model_name \
            else 'max_tokens'
        if 'gpt-5' in model_name:
            self.sampling_params["temperature"] = 1.0

    def get_response(self, messages, **kwargs):
        # TODO: This is a format for Mistral, implement for other models
        if self.is_embedding:
            if isinstance(messages, list):
                messages = {"input": messages[-1]['content']}
            elif isinstance(messages, str):
                messages = {"input": messages}
            request_data = messages
            response, elapsed_seconds = self.llm_client.send_request(self.model_name, request_data)
            return response
        else:
            if "gpt-o1" in self.model_name or 'o1' in self.model_name or 'o3' in self.model_name or 'o4' in self.model_name:
                # o1 doesn't support sampling parameters
                # o1 doesn't support system prompts
                for msg in messages:
                    if msg['role'] == 'system':
                        msg['role'] = 'user'
                request_data = {
                    "messages": messages,
                }
            elif "mistral" in self.model_name:
                # mistral doesn't support system messages, however, it can take user, assistant, user .. messages
                # merge all system and user messages into one, keep the user msg as is
                new_msgs = []
                temp_user_msg = []
                temp_assistant_msg = []
                for msg in messages:
                    if msg['role'] in ['user', 'system']:
                        temp_user_msg.append(msg['content'])
                        if len(temp_assistant_msg) > 0:
                            new_msgs.append({'role': 'assistant', 'content': '\n'.join(temp_assistant_msg)})
                            temp_assistant_msg = []
                    elif msg['role'] == 'assistant':
                        temp_assistant_msg.append(msg['content'])
                        if len(temp_user_msg) > 0:
                            new_msgs.append({'role': 'user', 'content': '\n'.join(temp_user_msg)})
                            temp_user_msg = []
                    else:
                        raise ValueError(f"Invalid role {msg['role']}")
                if len(temp_user_msg) > 0:
                    new_msgs.append({'role': 'user', 'content': '\n'.join(temp_user_msg)})
                if len(temp_assistant_msg) > 0:
                    new_msgs.append({'role': 'assistant', 'content': '\n'.join(temp_assistant_msg)})
                messages = new_msgs
                request_data = {
                    "messages": messages,
                    self.max_token_key: self.sampling_params[self.max_token_key],
                    "temperature": self.sampling_params["temperature"],
                    "top_p": self.sampling_params["top_p"],
                    "stream": False
                }
            else:
                request_data = {
                    "messages": messages,
                    self.max_token_key: self.sampling_params[self.max_token_key],
                    "temperature": self.sampling_params["temperature"],
                    "top_p": self.sampling_params["top_p"],
                    "stream": False,
                    "seed": self.sampling_params.get("seed", 42)
                }
            logger.debug(f"Sending request to LLM API with endpoint {self.llm_client._endpoint}, model {self.model_name}")
            if "stop_sequences" in kwargs:
                request_data["stop"] = kwargs["stop_sequences"]
            response, elapsed_seconds = self.llm_client.send_request(self.model_name, request_data)
            return response, response['choices'][0]['message']['content'], elapsed_seconds


class AzureApiWrapper(ApiWrapper):
    # wrapper to make calls to the AzureApiClient
    def __init__(self, model_name, sampling_params=None, is_embedding=False):
        super().__init__(model_name, sampling_params, is_embedding)
        private_config = get_private_config()
        endpoint, api_version = _resolve_azure_endpoint_and_version(model_name, private_config)
        if is_embedding:
            self.llm_client = AzureEmbeddingApiClient(endpoint, api_version)
        else:
            self.llm_client = AzureChatApiClient(endpoint, api_version)


class ApiClient(ABC):
    def __init__(self, endpoint: str, client_id: str):
        self._endpoint = endpoint
        self._client_id = client_id
        
    @abstractmethod
    def send_request(self, model_name: str, request: dict) -> dict:
        pass

    @abstractmethod
    def send_stream_request(self, model_name: str, request: dict) -> dict:
        pass

class AzureApiClient(ApiClient):
    def __init__(self, endpoint: str, api_version: str):
        super().__init__(endpoint, None)
        self._endpoint = endpoint
        self._api_version = api_version
        self.max_retries = 10
        from azure.identity import DefaultAzureCredential, AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider
        from openai import AzureOpenAI
        if os.environ.get('DEFAULT_IDENTITY_CLIENT_ID'):
            self._credential = DefaultAzureCredential()
        else:
            self._credential = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
        self._token_provider = get_bearer_token_provider(self._credential, "https://cognitiveservices.azure.com/.default")
        logger.info(f"Using Azure OpenAI with endpoint {self._endpoint} and API version {self._api_version}")
        self.client = AzureOpenAI(
            azure_endpoint= self._endpoint,
            api_version = self._api_version,
            azure_ad_token_provider=self._token_provider
        )

                
    @abstractmethod
    def send_request(self, model_name: str, request: dict) -> dict:
        pass

    @abstractmethod
    def send_stream_request(self, model_name: str, request: dict) -> dict:
        pass

class AzureChatApiClient(AzureApiClient):
    def __init__(self, endpoint: str, api_version: str):
        super().__init__(endpoint, api_version)

    def send_request(self, model_name: str, request: dict) -> Tuple[dict, float]:
        messages = request.get("messages", [])
        options = dict(request)
        if 'messages' in options:
            del options['messages']
        logger.debug(f"Sending request to Azure API with model {model_name} and options {options}")
        timestart = time.time()
        retries = 10
        while retries > 0:
            try:
                result = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **options
                )
                retries = 0
            except RateLimitError as e:
                logger.warning(f"Rate limit exceeded: {e}. Retrying in 30 seconds...")
                time.sleep(30)
                retries -= 1
                continue
            except (BadRequestError, APIStatusError) as e:
                # Detect ResponsibleAIPolicyViolation and do NOT retry
                def _is_policy_violation(err: Exception) -> bool:
                    try:
                        # Newer SDKs: err.response is httpx.Response-like
                        body = None
                        if hasattr(err, 'response') and getattr(err.response, 'json', None):
                            try:
                                body = err.response.json()
                            except Exception:
                                body = None
                        if body is None and hasattr(err, 'body'):
                            body = getattr(err, 'body')
                        if isinstance(body, dict):
                            inner = ((body.get('error') or {}).get('innererror')
                                     or (body.get('error') or {}).get('inner_error')
                                     or {})
                            code = inner.get('code') or (body.get('error') or {}).get('code')
                            if code == 'ResponsibleAIPolicyViolation':
                                return True
                            # Secondary signal
                            if (body.get('error') or {}).get('code') == 'content_filter':
                                return True
                        # Fallback: string search
                        s = str(err)
                        return ('ResponsibleAIPolicyViolation' in s) or ("'code': 'content_filter'" in s)
                    except Exception:
                        return 'ResponsibleAIPolicyViolation' in str(err)

                if _is_policy_violation(e):
                    # Raise a specific exception so upper layers can avoid retries
                    raise ResponsibleAIPolicyViolationError(str(e))
                # Otherwise, re-raise for normal handling
                raise
        elapsed_seconds = time.time() - timestart
        return result.to_dict(), elapsed_seconds   
    
    def send_stream_request(self, model_name: str, request: dict) -> dict:                           
        raise NotImplementedError("Streaming is not yet supported for Azure API client")

class AzureEmbeddingApiClient(AzureApiClient):
    def __init__(self, endpoint: str, api_version: str):
        # Reuse AzureApiClient init to set up credential + self.client
        super().__init__(endpoint, api_version)

    def send_request(self, model_name: str, request: dict) -> Tuple[dict, float]:
        logger.debug(f"Sending embedding request to Azure API with model {model_name} and input size {len(request['input'])}")
        timestart = time.time()
        result = self.client.embeddings.create(model=model_name, input=request['input'])
        elapsed_seconds = time.time() - timestart
        # Normalize to dict like chat client for downstream uniform access
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        else:  # fallback for SDK variants
            try:
                result_dict = json.loads(result.model_dump_json())
            except Exception:
                # Last resort: construct minimal dict form expected by callers
                result_dict = {"data": [{"embedding": d.embedding if hasattr(d, 'embedding') else d} for d in getattr(result, 'data', [])]}
        return result_dict, elapsed_seconds

    def send_stream_request(self, model_name: str, request: dict):
        raise NotImplementedError("Streaming is not supported for Azure Embedding API client")

def get_token_cache_file():
    for path in [".msal_tokens.json", "/apps/.msal_tokens.json"]:
        if exists(path):
            return path
    return ".msal_tokens.json"