"""Thin Jinja + LLM wrapper base classes to reduce duplication."""
from __future__ import annotations
from pathlib import Path
import jinja2
import os
from typing import Any, Optional

from productive_agents.agents.utils import LLMManager


class JinjaLLMTemplate:
    def __init__(self, model_name: str, template_path: Path, llm_backend: str = 'azure'):
        self.model_name = model_name
        self.template_path = template_path
        self.llm_backend = llm_backend

        if not self.template_path:
            self.template = None
        else:
            if self.template_path is not None and not self.template_path.exists():
                raise FileNotFoundError(f'Template not found: {self.template_path}')
            self.template = jinja2.Template(self.template_path.read_text())
        
        self.llm = LLMManager.create_llm(
            model_name=self.model_name,
            key=None,  # Key is managed internally in LLMManager
            system_message='',
        )

    def render(self, **kwargs: Any) -> str:
        if self.template is None:
            raise ValueError("Template path is not set.")
        return self.template.render(**kwargs)

    def generate(self, prompt: str) -> str:
        return self.llm.generate(prompt, max_tokens=16384)
