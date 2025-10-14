from .copy import DEMO as copy_demo
from paste import DEMO as paste_demo

class SystemInfo:
    def __init__(self):
        self.name = "shell"
        self.info = {
            "copy": copy_demo,
            "paste": paste_demo,
        }