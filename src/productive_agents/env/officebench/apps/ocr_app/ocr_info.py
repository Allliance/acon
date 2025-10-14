from .ocr_recognize_file import DEMO as recognize_file_demo

class OCRInfo:
    def __init__(self):
        self.name = "ocr"
        self.info = {
            "recognize_file": recognize_file_demo,
        }
        
    def get_instruction(self) -> str:
        instructions = [f"Command to perform function: {key}:\n{demo}" for key, demo in self.info.items()]
        return f"## How to use the {self.name} app:\n\n" + "\n\n".join(instructions)