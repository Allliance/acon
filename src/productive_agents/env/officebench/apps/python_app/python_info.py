from apps.python_app.python_run_code import DEMO as run_code_demo

class PythonInfo:
    def __init__(self):
        self.name = "python"
        self.info = {
            "run_code": run_code_demo,
        }

    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)