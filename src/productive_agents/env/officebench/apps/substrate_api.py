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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_private_config():
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    private_config_paths = [os.path.join(src_dir, "configs", f'{"private_config"}.yaml'), os.path.join(src_dir, "apps", f'{"private_config"}.yaml')]
    for private_config_path in private_config_paths:
        if exists(private_config_path):
            break
    else:
        raise FileNotFoundError("No private configuration file found.")
    
    with open(private_config_path, "r") as file:
        private_config = yaml.safe_load(file)  # loading the experiment, task and model config
    return private_config



class vLLM_API:
    # for parsing image to text
    def __init__(self, model_name='dev-phi-35-vision-instruct'):
        max_token_key = 'max_completion_tokens' if 'o1' in model_name else 'max_tokens'
        self.sampling_params = {
            max_token_key: 1024,
            "temperature": 1.0 if 'o1' in model_name else 0.0,
            "top_p": 1.0,
        }        
        self.llm_model = LLMApiWrapper(model_name, self.sampling_params)
        
    def get_response(self, image_path):
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        mime_type, _ = guess_type(image_path)
        messages = [{ "role": "user",
            "content": [
                {"type": "text", "text": "List the text in the message as a table, separate the columns with a comma and rows by newline."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}]}]
        response_text = self.llm_model.get_response(messages)[1]
        return response_text
    

class AzureOpenAIEmbeddings:
    def __init__(self, model_name="dev-text-embedding-3-small"):
        self.embedding_model = LLMApiWrapper(model_name, is_embedding=True)
        

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
        self.max_token_key = 'max_completion_tokens' if 'o1' in model_name else 'max_tokens'

    def get_response(self, messages):
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
            if "gpt-o1" in self.model_name or 'o1' in self.model_name:
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
                    "stream": False
                }
            logger.debug(f"Sending request to LLM API with endpoint {self.llm_client._endpoint}, model {self.model_name}")
            response, elapsed_seconds = self.llm_client.send_request(self.model_name, request_data)

            return response, response['choices'][0]['message']['content'], elapsed_seconds


class LLMApiWrapper(ApiWrapper):
    # wrapper to make calls to the LLMApiClient
    def __init__(self, model_name, sampling_params=None, is_embedding=False):
        super().__init__(model_name, sampling_params, is_embedding)        
        private_config = get_private_config()
        endpoint = private_config["endpoint"]
        if is_embedding:
            endpoint += "embeddings/"
        else:
            endpoint += "chat/completions/"
        self.llm_client = LLMApiClient(endpoint, private_config["tenant_id"], private_config["client_id"])
        
        self.sampling_params = sampling_params
        self.model_name = model_name
 

class AzureApiWrapper(ApiWrapper):
    # wrapper to make calls to the AzureApiClient
    def __init__(self, model_name, sampling_params=None, is_embedding=False):
        super().__init__(model_name, sampling_params, is_embedding)
        private_config = get_private_config()
        if is_embedding:
            self.llm_client = AzureEmbeddingApiClient(private_config["azure_endpoint"], private_config["azure_api_version"])
        else:
            self.llm_client = AzureChatApiClient(private_config["azure_endpoint"], private_config["azure_api_version"])


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
        options = request
        del options['messages']
        logger.info(f"Sending request to Azure API with model {model_name} and options {options}")
        timestart = time.time()
        retries = 10
        while retries > 0:
            try:
                result= self.client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            **options) 
                retries = 0
            except RateLimitError as e:
                    logger.warning(f"Rate limit exceeded: {e}. Retrying in 30 seconds...")
                    time.sleep(30)
                    retries -= 1
                    continue
        elapsed_seconds = time.time() - timestart
        return result.to_dict(), elapsed_seconds   
    
    def send_stream_request(self, model_name: str, request: dict) -> dict:                           
        raise NotImplementedError("Streaming is not yet supported for Azure API client")

class AzureEmbeddingApiClient(ApiClient):
    def __init__(self, endpoint: str, api_version: str):
        super().__init__(endpoint, api_version)

    def send_request(self, model_name: str, request: dict) -> Tuple[dict,float]:
        logger.info(f"Sending request to Azure API with model {model_name} and input {request['input']}")
        timestart = time.time()
        result = self.client.embeddings.create(model=model_name, input=request['input'])
        elapsed_seconds = time.time() - timestart
        return result, elapsed_seconds
    
    def send_stream_request(self, model_name, request):
        raise NotImplementedError("Streaming is not yet supported for Azure Embedding API client")

def get_token_cache_file():
    for path in [".msal_tokens.json", "/apps/.msal_tokens.json"]:
        if exists(path):
            return path
    return ".msal_tokens.json"


class LLMApiClient(ApiClient):

    _TOKENS_CACHE_FILE = get_token_cache_file()
    assert _TOKENS_CACHE_FILE is not None, "Token cache file not found. Ensure the cache file is copied to /apps."

    def __init__(self, endpoint: str, tenant_id: str, client_id: str):
        self._endpoint = endpoint
        self._client_id = client_id
        self._scopes = ['https://substrate.office.com/llmapi/LLMAPI.dev']
        self.max_retries = 10
        
        self._cache = SerializableTokenCache()
        atexit.register(lambda: 
            open(self._TOKENS_CACHE_FILE, 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)
        
        if exists(self._TOKENS_CACHE_FILE):
            self._cache.deserialize(open(self._TOKENS_CACHE_FILE, 'r').read())

        authority = f"https://login.microsoftonline.com/{tenant_id}"
        self._app = PublicClientApplication(self._client_id, authority=authority, token_cache=self._cache)

        

    
    def _get_auth_token(self) -> str:
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            selected_account = accounts[0]
            result = self._app.acquire_token_silent(self._scopes, selected_account)

        if not result:
            flow = self._app.initiate_device_flow(self._scopes)

            if 'user_code' not in flow:
                raise ValueError(f'Fail to initialize device flow. Error: {json.dumps(flow)}')

            print(flow["message"])
            result = self._app.acquire_token_by_device_flow(flow)

        if not result:
            raise RuntimeError(f'Fail to authenticate')
        #print(dir(result))        
        logger.info(f"MSAL Token expires in {result['expires_in']//60}:{result['expires_in']%60} minutes")
        return result['access_token']
    
    def _get_headers(self, model_name: str) -> dict:
        auth_token = self._get_auth_token()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_token}',
            'X-ModelType': model_name
        }

        return headers
    
    def send_request(self, model_name: str, request: dict) -> Tuple[dict, float]:
        headers = self._get_headers(model_name)
        body = str.encode(json.dumps(request))
        max_retries_chunk_exception = 10
        wait_time = 60
        for chunk_attempt in range(max_retries_chunk_exception):
            try:
                for attempt in range(self.max_retries):
                    response = requests.post(self._endpoint, data=body, headers=headers)
                    if response.status_code >= 400:
                        print (f'retrying attempt {attempt} for model {model_name} in {wait_time} seconds')
                        print ('code', response.text)
                        time.sleep(wait_time)
                    else:
                        response_time = response.elapsed.total_seconds()
                        return response.json(), response_time
                raise RuntimeError(f'API Request failed with error for model {model_name}')
            except requests.exceptions.ChunkedEncodingError:
                print ('ChunkedEncodingError occured, retrying after 10 sec...')
                time.sleep(wait_time)
        raise RuntimeError(f'API Request failed with error for model {model_name}')
    
    def send_stream_request(self, model_name: str, request: dict) -> dict:
        headers = self._get_headers(model_name)
        body = str.encode(json.dumps(request))
        response = requests.post(self._endpoint, data=body, headers=headers, stream=True)
        
        if response.status_code >= 400:
            raise RuntimeError(f'Request failed with error: {response.content}')

        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data:'):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)