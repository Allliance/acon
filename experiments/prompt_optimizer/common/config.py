"""Config merge helpers (CLI args + optional YAML/JSON file)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_config_file_and_merge(cli_namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    config_path = getattr(cli_namespace, 'config', None)
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found: {config_path}')
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                if yaml is None:
                    raise ImportError('pyyaml not installed but YAML provided')
                cfg = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                cfg = json.load(f)
            else:
                raise ValueError('Unsupported config extension')
    merged = dict(cfg)
    for k, v in vars(cli_namespace).items():
        if v is not None:
            merged[k.replace('_', '-')] = v
    # Normalize keys to underscores
    return {k.replace('-', '_'): v for k, v in merged.items()}
