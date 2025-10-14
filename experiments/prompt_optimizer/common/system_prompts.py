"""Helpers for copying co-located system prompt templates with an updated prompt."""
from __future__ import annotations
from pathlib import Path
from typing import List
import logging
import shutil


def copy_system_prompts_from_source(base_template_path: Path, dest_dir: Path) -> None:
    src_dir = base_template_path.parent
    candidates: List[Path] = []
    for pattern in ('system*.jinja', 'system*.txt', 'system*.md', 'system*.json'):
        candidates.extend(list(src_dir.glob(pattern)))
    if not candidates:
        logging.debug(f'No system prompt files found in {src_dir}')
        return
    for src in candidates:
        if not src.is_file():
            continue
        dest = dest_dir / src.name
        try:
            if dest.exists():
                logging.debug(f'System prompt already exists, skipping copy: {dest}')
            else:
                shutil.copy2(src, dest)
                logging.info(f'Copied system prompt: {src.name} -> {dest}')
        except Exception as e:  # pragma: no cover
            logging.warning(f'Failed to copy system prompt {src}: {e}')
