import fire
import time
import shlex
import os
import sys
from ruamel.yaml import YAML
import logging

yaml = YAML()
yaml.preserve_quotes = True  # Preserve the original quoting style
yaml.width = 4096


def setup_logging():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)
    azure_logger = logging.getLogger("azure.identity")
    azure_logger.setLevel(logging.WARNING)
    azure_core_logger = logging.getLogger("azure.core")
    azure_core_logger.setLevel(logging.WARNING)
    
setup_logging()

current_file_path = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_file_path, '..')
# Temporarily add substrate_api's directory to sys.path
sys.path.insert(0, app_path)
from substrate_api import LLMApiWrapper, AzureApiWrapper
sys.path.pop(0)  # Remove it immediately after importing


class ChatGPT:
    def __init__(self, model_name, service="azure", system_message=None):
        self.model_name = model_name
        self.temperature = 1.0 if 'o1' in model_name else 0.0
        if service == 'llmapi':
            self.llm_model = LLMApiWrapper(model_name, ChatGPT.get_model_options(temperature=self.temperature), is_embedding=False)
        else:
            assert service=='azure'
            self.llm_model = AzureApiWrapper(model_name, ChatGPT.get_model_options(temperature=self.temperature))
        
        
        self.system_message = system_message

    @classmethod
    def get_model_options(
        cls,
        temperature=0,
        per_example_max_decode_steps=2048,
        per_example_top_p=1,
        n_sample=1,
    ):
        return dict(
            temperature=temperature,
            n=n_sample,
            top_p=per_example_top_p,
            max_tokens=per_example_max_decode_steps,
        )

    def generate_plus_with_score(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options(temperature=self.temperature)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        gpt_responses = None
        retry_num = 0
        retry_limit = 2
        error = None
        while gpt_responses is None and retry_num < retry_limit:
            try:
                response, gpt_responses, _ = self.llm_model.get_response(
                    messages=messages
                )
                logging.info(f"gpt_responses: {gpt_responses}")
                error = None
            except Exception as e:
                error = e
                time.sleep(5)
                retry_num += 1
        if error:
            raise Exception(error)
        return gpt_responses

    

    def generate(self, prompt, options=None, end_str=None):
        if options is None:
            options = self.get_model_options()
        options["n"] = 1
        result = self.generate_plus_with_score(prompt, options, end_str)
        return result


DEMO = (
    'Query an LLM model for an answer to a given prompt: '
    '{"app": "llm", "action": "complete_text", "prompt": [PROMPT]}'
)


def construct_action(
    word_dir, args: dict, py_file_path="/apps/llm_app/llm_query.py"
):
    # return f'python3 {py_file_path} --prompt "{args["prompt"]}" '
    return "python3 {} --prompt \"{}\"".format(py_file_path, shlex.quote(shlex.quote(args["prompt"])))


def query(prompt, exp_path=None):
    """
    Query an LLM model for an answer to a given prompt.
    """
    try:
        # load the experiment config
        if exp_path is None:
            exp_path = 'apps/exp_config.yaml'
        with open(exp_path, "r") as exp_config_file:
            exp_config = yaml.load(exp_config_file)
        model_name = exp_config.get("model", {}).get("name", "azure/gpt-4o")
        service = 'llmapi'
        if 'azure' in model_name:
            service = 'azure'
            model_name = model_name.split('/')[1]

        llm = ChatGPT(model_name, service)
        
        response = llm.generate(prompt)
        return response
    except Exception as e:
        return f"Error: [llm] {e}"


def main(prompt):
    response = query(prompt)
    if response is not None:
        return f"OBSERVATION: {response}"

    return "OBSERVATION: No response from the model."


if __name__ == "__main__":
    fire.Fire(main)
