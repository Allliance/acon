import os
import sys
import fire
import pytesseract
from PIL import Image
import base64
from productive_agents.env.officebench.apps import APPS_ROOT
from ruamel.yaml import YAML
# from substrate_api import vLLM_API, AzureApiWrapper

current_file_path = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_file_path, '..')
# Temporarily add substrate_api's directory to sys.path
sys.path.insert(0, app_path)
sys.path.pop(0)  # Remove it immediately after importing


# DEMO = (
#     'You can recognize an image by calling `ocr_recognize_file` with 1 argument.\n'
#     '1. file_path: the path to the image file.\n'
#     "You can call it by generating command: {'app': 'ocr', 'action': 'ocr_recognize_file', 'file_path': ...}"
# )


'''
Substrate API for calling the LLM/SLM endpoints, used by the models to make requests
'''

yaml = YAML()
yaml.preserve_quotes = True  # Preserve the original quoting style
yaml.width = 4096


# vllm_image_parser = vLLM_API()
# openai_image_parser = AzureApiWrapper(
#     model_name="gpt-4o",
#     sampling_params={"temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": 4096},
# )

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

DEMO = (
    'recognize the text from an image file: '
    '{"app": "ocr", "action": "recognize_file", "file_path": [THE_PATH_TO_THE_IMAGE_FILE]}'
)

def construct_action(work_dir, args: dict, py_file_path=f'{APPS_ROOT}/ocr_app/ocr_recognize_file.py'):
    # TODO: not sure if we need to specify the file path with the current workdir
    return f'python3 {py_file_path} --file_path {args["file_path"]}'

def ocr_recognize_file(file_path, exp_path=None):
    try:
        # load the experiment config
        if exp_path is None:
            exp_path = 'apps/exp_config.yaml'
        with open(exp_path, "r") as exp_config_file:
            exp_config = yaml.load(exp_config_file)
            ocr_option = exp_config.get("model", {}).get("ocr", "phi35-vision-instruct")

        if ocr_option == "phi35-vision-instruct":
            text = vllm_image_parser.get_response(file_path)
        elif ocr_option == "pytesseract":
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        elif ocr_option == "gpt-4o":
            img = encode_image(file_path)
            response = openai_image_parser.get_response(
                # Build messages containing user prompt (text) as OCR instruction and image as input
                # images should be given in the messages
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "List the text in the message as a table, separate the columns with a comma and rows by newline."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}"
                            },
                        },
                    ],
                }],
            )
            print("        !!!!!!!!  RUN GPT OCR !!!!!!!!!         ")
            text = response[1]
        else:
            raise ValueError(f"OCR option {ocr_option} not recognized")
    except Exception as e:
        print ('error', e)
        text = None
    return text

def main(file_path, exp_path=None, debug=False):
    if not os.path.exists(file_path):
        return f'OBSERVATION: The file {file_path} does not exist. Failed to recognize text.'
    
    text = ocr_recognize_file(file_path, exp_path)
    if debug:
        print(text)
    if text:
        observation = f'OBSERVATION: The text from {file_path} is:\n{text}'
    else:
        observation = f'OBSERVATION: Failed to recognize text from {file_path}'
    return observation


if __name__ == '__main__':
    fire.Fire(main)
