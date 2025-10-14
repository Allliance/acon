from .command import DEMO as command_demo

class ShellInfo:
    def __init__(self):
        self.name = "shell"
        self.info = {
            "shell command": command_demo,
        }
    
    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)