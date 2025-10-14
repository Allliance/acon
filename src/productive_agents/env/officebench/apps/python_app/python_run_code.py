import os
import sys
import fire
import base64
from productive_agents.env.officebench.apps import APPS_ROOT


try:
    from local_python_executor import LocalPythonExecutor
except ModuleNotFoundError as e:
    from .local_python_executor import LocalPythonExecutor  # noqa: F401

DEMO = (
    'run the python code using a python interpreter: '
    '{"app": "python", "action": "run_code", "code": [THE_PYTHON_CODE (JSON-encoded string with all line breaks escaped as \\n and double quotes properly escaped.)]}'
)


additional_authorized_imports = [
    'pandas', 'numpy', 'matplotlib', 'scipy', 'sklearn',
    # Add any other libraries you want to allow here
]
EXECUTOR = LocalPythonExecutor(
    additional_authorized_imports=additional_authorized_imports,
)
EXECUTOR.send_tools({}) # will work with base python tools inside the local python executor

def construct_action(work_dir, args: dict, py_file_path=f'{APPS_ROOT}/python_app/python_run_code.py'):
    # Encode action into base64 to avoid any escape characters issue
    raw_code = args['code']
    encoded_bytes = base64.b64encode(raw_code.encode('utf-8'))
    encoded_code = encoded_bytes.decode('utf-8')
    return f'python3 {py_file_path} --b64code {encoded_code}'

def run_code(code, b64code=None):
    """
    Run the python code using a python interpreter.
    :param code: The python code to run.
    :param workdir: The working directory to run the code in.
    :return: The output of the code execution.
    """
    if b64code:
        decoded_code = base64.b64decode(b64code).decode("utf-8")
        code_to_run = decoded_code
    else:
        code_to_run = code

    result = EXECUTOR(code_to_run)
    output = result.output
    logs = result.logs if result.logs else 'No logs'
    observation = f"OBSERVATION: The output from code is:\n{output}\nThe stdout or stderr is:\n{logs}"
    return observation

def main(code=None, b64code=None):
    code_output = run_code(code, b64code)
    return code_output

if __name__ == '__main__':
    fire.Fire(main)
