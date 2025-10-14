"""Agents package public API.

This module avoids importing heavy agent submodules at import time to prevent
side effects and circular imports. It exposes agent classes via lazy access and
provides a lazy registry mapping.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    'OfficeBenchAgent',
    'AppWorldAgent',
    'SmolagentsAgent',
    'REGISTERED_AGENTS',
]

_ATTR_TO_MODULE = {
    'OfficeBenchAgent': 'productive_agents.agents.officebench',
    'AppWorldAgent': 'productive_agents.agents.appworld',
    'SmolagentsAgent': 'productive_agents.agents.smolagents',
}

def __getattr__(name: str) -> Any:  # PEP 562 lazy import for top-level classes
    module_name = _ATTR_TO_MODULE.get(name)
    if not module_name:
        raise AttributeError(f"module 'productive_agents.agents' has no attribute {name!r}")
    mod = import_module(module_name)
    return getattr(mod, name)


class _LazyRegistry:
    def __init__(self, mapping: Dict[str, Tuple[str, str]]):
        self._map = mapping
        self._cache: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key not in self._map:
            raise KeyError(key)
        module_name, class_name = self._map[key]
        mod = import_module(module_name)
        cls = getattr(mod, class_name)
        self._cache[key] = cls
        return cls

    def keys(self):
        return self._map.keys()

    def items(self):
        # Materialize values lazily when iterated
        for k in self._map.keys():
            yield (k, self[k])

    def values(self):
        for _, v in self.items():
            yield v

    def __contains__(self, key: str) -> bool:
        return key in self._map

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


REGISTERED_AGENTS = _LazyRegistry({
    "officebench": ('productive_agents.agents.officebench', 'OfficeBenchAgent'),
    "appworld": ('productive_agents.agents.appworld', 'AppWorldAgent'),
    "smolagents": ('productive_agents.agents.smolagents', 'SmolagentsAgent'),
})