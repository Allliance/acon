import fire
import os
import json
from datetime import datetime
import logging
from productive_agents.env.officebench.apps import APPS_ROOT

logging.basicConfig(level=logging.INFO)

# Demo provides documentation on how to use the scratchpad functionality
DEMO = (
    'write content to the scratchpad. Content can be notes, plans, or any text.\n'
    '{"app": "scratchpad", "action": "write", "content": [TEXT_CONTENT_TO_WRITE]}'
)

def construct_action(work_dir, args: dict, py_file_path=f'{APPS_ROOT}/scratchpad_app/scratchpad.py'):
    # Escape single quotes in content for shell command
    content = args.get("content", "").replace("'", "'\\''")
    return "python3 {} --workdir '{}' --content '''{}'''".format(
        py_file_path, work_dir, content
    )

def scratchpad_write(workdir, content):
    """
    Write content to a scratchpad file for notes, plans, or any text.
    
    Args:
        content (str): The text content to write to the scratchpad.
                                 
    Returns:
        str: A message confirming the content was written and the filename.
    """
    try:
        # Create directory if it doesn't exist
        #os.makedirs('scratchpad', exist_ok=True)
        
        # Generate filename based on timestamp if not provided
        filename = 'scratchpad.txt'
        
        # Full path to the scratchpad file
        filepath = os.path.join(workdir, filename)
        logging.info(f"Writing to scratchpad at {filepath}")
        # Write content to file
        with open(filepath, 'w') as f:
            f.write(content)
        return "OBSERVATION: Content written to scratchpad"

    except Exception as e:
        return f"Error: [scratchpad] {e}"

# this method is not exported as a tool but instead the scratchpad is injected into the prompt on every turn
def scratchpad_read(workdir):
    """
    Read content from the scratchpad file.
    
    Args:
        workdir (str): The directory where the scratchpad file is located.
        
    Returns:
        str: The content of the scratchpad file.
    """
    try:
        filepath = os.path.join(workdir, 'scratchpad.txt')
        logging.info(f"Reading scratchpad from {filepath}")
        if not os.path.exists(filepath):
            return "The scratchpad is empty."
        
        with open(filepath, 'r') as f:
            content = f.read()
            
        return content
        
    except Exception as e:
        return f"Error: [scratchpad] {e}"


if __name__ == "__main__":
    fire.Fire(scratchpad_write)
