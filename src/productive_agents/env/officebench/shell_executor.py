import subprocess
from typing import Tuple

SHELL_TIMEOUT = 120

def execute_shell_command(command: str, verbose: bool = False) -> Tuple[int, bytes, bytes]:
    """
    Execute a shell command and return the exit code and output.

    Args:
        command (str): The shell command to execute.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        int: The exit code of the command.
        str: The output of the command.
    """
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(timeout=SHELL_TIMEOUT)
    exit_code = process.returncode
    if verbose:
        print(f"\n{'='*20} Shell Command Execution {'='*20}")
        print(f"Command: {command}")
        print(f"Exit code: {exit_code}")
        print(f"STDOUT: {stdout.decode()}")
        print(f"STDERR: {stderr.decode()}")
        print(f"{50 * '='}\n")

    return exit_code, stdout, stderr