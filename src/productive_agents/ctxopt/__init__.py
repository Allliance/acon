"""Context optimization package for productive_agents.

This init avoids importing heavy submodules at import time to prevent circular
imports with agents that reference ctxopt. It exposes common symbols via
lazy-loading so ``from productive_agents.ctxopt import ObservationOptimizer``
continues to work without triggering the circular dependency.
"""

from importlib import import_module
from typing import Any

# Public API
__all__ = [
    'BaseContextOptimizer',
    'ObservationOptimizer',
    'HistoryOptimizer',
]

_ATTR_TO_MODULE = {
    'BaseContextOptimizer': '.base',
    'ObservationOptimizer': '.obs_optimizer',
    'HistoryOptimizer': '.history_optimizer',
}

def __getattr__(name: str) -> Any:  # PEP 562 lazy import
    module_path = _ATTR_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module 'productive_agents.ctxopt' has no attribute {name!r}")
    mod = import_module(module_path, __name__)
    try:
        return getattr(mod, name)
    except AttributeError as e:
        # Re-raise with clearer context
        raise AttributeError(f"Failed to load attribute {name!r} from {mod.__name__}") from e

